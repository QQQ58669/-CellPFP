import argparse
import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault("DGLBACKEND", "pytorch")

import dgl
import numpy as np
import torch

from cellpfp.data import PredictionData, gene_symbol, load_tensor_file, read_ids
from cellpfp.model import CellPFPModel


def load_ensemble(ontology, models_dir, data, ids, device, batch_size, cache_interpro=False):
    expected_terms = None
    expected_config = None
    scores = None
    for slot in range(3):
        path = models_dir / f"{ontology}_{slot}.pt"
        checkpoint = load_tensor_file(path)
        if checkpoint["ontology"] != ontology or checkpoint["slot"] != slot:
            raise ValueError(f"Checkpoint identity does not match its filename: {path}")
        terms = checkpoint["terms"]
        config = checkpoint["config"]
        weights = checkpoint["model_state_dict"]
        if slot == 0:
            expected_terms = terms
            expected_config = config
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            dgl.seed(config["seed"])
            data.configure(config)
            model = CellPFPModel(
                labels_num=len(terms),
                interpro_dim=data.interpro.shape[1],
                cell_embedding=torch.zeros_like(weights["cell_embedding.weight"]),
                **config["model"],
            ).to(device).eval()
            interpro = data.interpro.to(device)
            mapping = data.global_to_interpro.to(device)
            scores = np.zeros((len(ids), len(terms)), dtype=np.float64)
        elif terms != expected_terms or config != expected_config:
            raise ValueError(f"GO terms or model settings differ across checkpoints: {path}")
        model.load_state_dict(weights, strict=True)
        with torch.inference_mode():
            node_interpro = None
            if cache_interpro:
                projected = model.interpro_encoder(interpro)
                missing_row = model.interpro_encoder(interpro.new_zeros((1, interpro.shape[1])))
                node_interpro = torch.cat((projected, missing_row))[mapping]
            for start in range(0, len(ids), batch_size):
                batch_ids = ids[start:start + batch_size]
                batch = data.batch(batch_ids, device)
                probabilities = torch.sigmoid(model(batch, interpro, mapping, node_interpro))
                if not torch.isfinite(probabilities).all():
                    raise ValueError(f"Non-finite scores from {path.name}")
                scores[start:start + len(batch_ids)] += probabilities.cpu().numpy()
        del checkpoint, weights
    scores /= 3
    return expected_terms, scores


def read_test_labels(path, ids, terms):
    proteins = {pid: row for row, pid in enumerate(ids)}
    columns = {term: column for column, term in enumerate(terms)}
    labels = np.zeros((len(ids), len(terms)), dtype=bool)
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            pid, term = fields[:2]
            if pid in proteins and term in columns:
                labels[proteins[pid], columns[term]] = True
    missing = [pid for pid, row in zip(ids, labels) if not row.any()]
    if missing:
        raise ValueError(f"Missing test annotations in the model GO vocabulary: {', '.join(missing[:10])}")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Predict GO scores with the trained CellPFP ensemble")
    parser.add_argument("-d", "--ontology", choices=["bp", "mf", "cc"], required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--proteins", type=Path)
    inputs.add_argument("--evaluate", action="store_true", help="Evaluate on the complete labeled test2 split")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--embeddings-dir", type=Path)
    parser.add_argument("--output", type=Path, help="TSV output for protein-list prediction")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--top-k", type=int, help="For TSV predictions, default 20. Use 0 for all GO terms")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.top_k is not None and args.top_k < 0):
        parser.error("--batch-size must be positive and --top-k must be nonnegative")
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.evaluate:
        if args.top_k is not None:
            parser.error("--evaluate always evaluates all GO terms")
        if args.output is not None:
            if args.output.suffix != ".npz":
                parser.error("Optional evaluation output must have an .npz extension")
            metrics_path = args.output.with_suffix(".metrics.json")
            if metrics_path.exists():
                parser.error(f"Output already exists: {metrics_path}")
        protein_file = args.data_dir / f"{args.ontology}_test2_pid_list.txt"
        annotation_file = args.data_dir / f"{args.ontology}_test2_go.txt"
        go_file = args.data_dir / "go.obo"
        for path in (protein_file, annotation_file, go_file):
            if not path.is_file():
                parser.error(f"Missing benchmark file: {path}")
    else:
        if args.output is None:
            parser.error("--proteins requires --output")
        protein_file = args.proteins
        top_k = 20 if args.top_k is None else args.top_k
    try:
        ids = read_ids(protein_file)
        embeddings_dir = args.embeddings_dir or args.data_dir / "esm2_embeddings"
        missing = [pid for pid in ids if not (embeddings_dir / f"{gene_symbol(pid)}.pt").is_file()]
        if missing:
            raise FileNotFoundError(f"Missing embeddings for {len(missing)} protein(s): {', '.join(missing[:10])}")
        for slot in range(3):
            path = args.models_dir / f"{args.ontology}_{slot}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
        data = PredictionData(args.data_dir, embeddings_dir)
        terms, scores = load_ensemble(
            args.ontology, args.models_dir, data, ids, torch.device(args.device), args.batch_size,
            cache_interpro=args.evaluate,
        )
        if args.evaluate:
            from cellpfp.metrics import Evaluator

            labels = read_test_labels(annotation_file, ids, terms)
            metrics = Evaluator(go_file, args.ontology, terms)(labels, scores)
            if args.output is not None:
                metrics.update(
                    ontology=args.ontology, split="test2", protein_count=len(ids),
                    go_term_count=len(terms), batch_size=args.batch_size, device=args.device,
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("xb") as stream:
                    np.savez_compressed(stream, protein_ids=ids, terms=terms, scores=scores, labels=labels)
                with metrics_path.open("x", encoding="utf-8") as stream:
                    json.dump(metrics, stream, indent=2)
                    stream.write("\n")
            print(f"Fmax: {metrics['fmax']:.4f}", flush=True)
            print(f"AUPR: {metrics['aupr']:.4f}", flush=True)
            return
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["protein_id", "ontology", "go_id", "score"])
            for pid, row in zip(ids, scores):
                order = np.argsort(-row, kind="stable")
                if top_k:
                    order = order[:top_k]
                writer.writerows((pid, args.ontology, terms[i], f"{row[i]:.10g}") for i in order)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Prediction failed: {error}\n")
    print(f"Predicted {len(ids)} proteins for {args.ontology}: {args.output}")


if __name__ == "__main__":
    main()
