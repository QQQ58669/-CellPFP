import json
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass

import dgl
import torch
from torch.nn.utils.rnn import pad_sequence


def load_tensor_file(path):
    with path.open("rb") as stream:
        try:
            return torch.load(stream, map_location="cpu", weights_only=True)
        except TypeError:
            stream.seek(0)
            return torch.load(stream, map_location="cpu")


def gene_symbol(pid):
    return pid[:-6] if pid.endswith("_HUMAN") else pid


def read_ids(path):
    ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    genes = [gene_symbol(pid) for pid in ids]
    if not ids or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", gene) for gene in genes):
        raise ValueError("Provide one gene symbol or GENE_HUMAN identifier per line.")
    if len(genes) != len(set(genes)):
        raise ValueError("The protein list contains duplicate gene symbols.")
    return ids


@dataclass
class GraphGroup:
    cell_idx: torch.Tensor
    input_nodes: torch.Tensor
    batch_indices: torch.Tensor
    slot_indices: torch.Tensor
    blocks: list


@dataclass
class CellPFPInput:
    esm_tokens: torch.Tensor
    esm_mask: torch.Tensor
    graph_groups: list
    cell_mask: torch.Tensor
    cell_degrees: torch.Tensor
    cell_ranks: torch.Tensor
    cell_indices: torch.Tensor


class PredictionData:
    def __init__(self, root, embeddings_dir):
        self.root = root
        self.embeddings_dir = embeddings_dir
        self.interpro = load_tensor_file(root / "atgo_interpro_features.pt").float()
        interpro_ids = json.loads((root / "atgo_interpro_row_mapping.json").read_text(encoding="utf-8"))
        if len(interpro_ids) != self.interpro.shape[0]:
            raise ValueError("InterPro row mapping does not match the feature matrix.")
        with (root / "global_node_mapping.pkl").open("rb") as stream:
            self.node2idx = pickle.load(stream)["node2idx"]
        self.global_to_interpro = torch.full((len(self.node2idx),), -1, dtype=torch.long)
        for row, gene in enumerate(interpro_ids):
            node = self.node2idx.get(gene)
            if node is not None:
                self.global_to_interpro[node] = row
        self.cell_indices = {}
        self.graphs = {}
        with (root / "cell_name_mapping.txt").open(encoding="utf-8") as stream:
            for line in stream:
                name, index = line.rstrip("\n").split("\t")
                index = int(index)
                if not 0 <= index < 156:
                    continue
                path = root / "156_dgl_graphs_string_weighted" / f"{name}.dgl"
                if not path.is_file():
                    raise FileNotFoundError(path)
                graph = dgl.load_graphs(str(path))[0][0]
                if graph.num_nodes() != len(self.node2idx) or "ppi" not in graph.edata:
                    raise ValueError(f"Invalid weighted PPI graph: {path}")
                ppi = graph.edata["ppi"].float()
                graph.edata.clear()
                graph.edata["ppi"] = ppi
                self.cell_indices[name] = index
                self.graphs[index] = graph
        self.ranked_cells = self.rank_cells()

    def configure(self, config):
        self.top_k = config["top_k_cells"]
        self.max_length = config["max_esm_len"]
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler(config["fanouts"])

    def rank_cells(self):
        expressed = {}
        with (self.root / "protein_cell_expression.txt").open(encoding="utf-8") as stream:
            for line in stream:
                pid, cells = line.rstrip("\n").split("\t")
                expressed[gene_symbol(pid)] = set(cells.split(","))
        candidates = defaultdict(list)
        with (self.root / "protein_degree_in_cells.txt").open(encoding="utf-8") as stream:
            next(stream)
            for line in stream:
                pid, name, degree = line.rstrip("\n").split("\t")
                gene = gene_symbol(pid)
                if name in self.cell_indices and name in expressed.get(gene, set()):
                    candidates[gene].append((self.cell_indices[name], int(degree)))
        return {
            gene: sorted(cells, key=lambda item: item[1], reverse=True)
            for gene, cells in candidates.items()
        }

    def embedding(self, gene):
        path = self.embeddings_dir / f"{gene}.pt"
        value = load_tensor_file(path)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 1280 or not len(value):
            raise ValueError(f"Expected a nonempty residue-level [L, 1280] tensor: {path}")
        value = value[:self.max_length].float()
        if not torch.isfinite(value).all():
            raise ValueError(f"Embedding contains non-finite values: {path}")
        return value

    def batch(self, ids, device):
        genes = [gene_symbol(pid) for pid in ids]
        embeddings = [self.embedding(gene) for gene in genes]
        tokens = pad_sequence(embeddings, batch_first=True)
        lengths = torch.tensor([len(value) for value in embeddings])
        mask = torch.arange(tokens.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        shape = (len(genes), self.top_k)
        cell_mask = torch.zeros(shape, dtype=torch.bool)
        degrees = torch.zeros(shape)
        ranks = torch.zeros(shape)
        indices = torch.full(shape, -1, dtype=torch.long)
        groups = defaultdict(list)
        for row, gene in enumerate(genes):
            node = self.node2idx.get(gene)
            if node is None:
                continue
            cells = self.ranked_cells.get(gene, [])[:self.top_k]
            for slot, (cell, degree) in enumerate(cells):
                groups[cell].append((row, slot, node))
                cell_mask[row, slot] = True
                degrees[row, slot] = degree
                ranks[row, slot] = slot + 1
                indices[row, slot] = cell
        graph_groups = []
        for cell, pairs in groups.items():
            rows, slots, nodes = zip(*pairs)
            input_nodes, _, blocks = self.sampler.sample_blocks(
                self.graphs[cell], torch.tensor(nodes, dtype=torch.long)
            )
            graph_groups.append(GraphGroup(
                torch.tensor(cell, device=device),
                input_nodes.long().to(device),
                torch.tensor(rows, device=device),
                torch.tensor(slots, device=device),
                [block.to(device) for block in blocks],
            ))
        return CellPFPInput(
            tokens.to(device), mask.to(device), graph_groups, cell_mask.to(device),
            degrees.to(device), ranks.to(device), indices.to(device),
        )
