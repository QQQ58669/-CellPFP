from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from .data import CellPFPInput

class TransformerBlock(nn.Module):

    def __init__(self, input_dim: int=1280, nhead: int=8, num_layers: int=1, dropout: float=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=2048, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens, src_key_padding_mask=~mask)

class SelfAttentionPooling(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int=256):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1000000000.0)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(weights * x, dim=1)

class ESMBranch(nn.Module):

    def __init__(self, output_dim: int=512, dropout: float=0.25):
        super().__init__()
        self.transformer = TransformerBlock(input_dim=1280, dropout=0.1)
        self.pool = SelfAttentionPooling(1280)
        self.mlp = nn.Sequential(nn.Linear(1280, 2048), nn.LeakyReLU(), nn.Linear(2048, 1024), nn.Dropout(dropout), nn.LeakyReLU(), nn.Linear(1024, output_dim), nn.Dropout(dropout), nn.LeakyReLU())

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.transformer(tokens, mask)
        pooled = self.pool(encoded, mask)
        return self.mlp(pooled)

class WeightedGCN(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        self.activation = nn.LeakyReLU()
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            self.layers.append(GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
            if in_dim == hidden_dim:
                self.residual_projs.append(nn.Identity())
            else:
                self.residual_projs.append(nn.Linear(in_dim, hidden_dim, bias=False))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, features: torch.Tensor) -> torch.Tensor:
        h = features
        for layer, norm, residual_proj, block in zip(self.layers, self.norms, self.residual_projs, blocks):
            edge_weight: Optional[torch.Tensor] = block.edata.get('ppi')
            if edge_weight is not None:
                edge_weight = edge_weight.float()
            residual = h[:block.num_dst_nodes()]
            h = layer(block, h, edge_weight=edge_weight)
            h = self.activation(h)
            h = self.dropout(h)
            h = norm(h)
            h = h + residual_proj(residual)
        return h

class GOHead(nn.Module):

    def __init__(self, input_dim: int, labels_num: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LeakyReLU(), nn.Dropout(dropout), nn.Linear(input_dim, labels_num))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CellFiLMModulation(nn.Module):

    def __init__(self, cell_dim: int, fused_dim: int, dropout: float, hidden_dim: int=256):
        super().__init__()
        self.norm = nn.LayerNorm(fused_dim)
        self.context = nn.Sequential(nn.Linear(cell_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout))
        self.to_scale_shift = nn.Linear(hidden_dim, fused_dim * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, cell_vectors: torch.Tensor, cell_context: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
        scale_shift = self.to_scale_shift(self.context(cell_context))
        scale, shift = torch.chunk(scale_shift, chunks=2, dim=-1)
        modulated = cell_vectors + self.norm(cell_vectors) * scale + shift
        return torch.where(cell_mask.unsqueeze(-1), modulated, torch.zeros_like(modulated))

class CellTrustHead(nn.Module):

    def __init__(self, input_dim: int, dropout: float):
        super().__init__()
        hidden_dim = max(input_dim // 2, 1)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class CellPFPModel(nn.Module):

    def __init__(self, labels_num: int, interpro_dim: int, cell_embedding: torch.Tensor, interpro_hidden: int=256, graph_hidden: int=384, fused_dim: int=512, num_gcn_layers: int=2, dropout: float=0.25, graph_dropout: float=0.25, trust_alpha: float=0.5, freeze_cell_embedding: bool=True):
        super().__init__()
        self.fused_dim = fused_dim
        self.trust_alpha = float(trust_alpha)
        self.esm_branch = ESMBranch(output_dim=fused_dim, dropout=dropout)
        self.interpro_encoder = nn.Sequential(nn.Linear(interpro_dim, interpro_hidden), nn.LayerNorm(interpro_hidden), nn.LeakyReLU(), nn.Dropout(dropout))
        self.cell_embedding = nn.Embedding.from_pretrained(cell_embedding.float(), freeze=freeze_cell_embedding)
        graph_input_dim = interpro_hidden + cell_embedding.shape[1]
        self.gcn = WeightedGCN(graph_input_dim, graph_hidden, num_gcn_layers, graph_dropout)
        self.graph_to_fused = nn.Sequential(nn.Linear(graph_hidden, fused_dim), nn.LayerNorm(fused_dim), nn.LeakyReLU(), nn.Dropout(dropout))
        cell_context_dim = cell_embedding.shape[1] + 2
        self.cell_modulator = CellFiLMModulation(cell_dim=cell_context_dim, fused_dim=fused_dim, dropout=dropout)
        self.seq_go_head = GOHead(fused_dim, labels_num, dropout)
        self.cell_go_head = GOHead(fused_dim, labels_num, dropout)
        self.seq_gate_head = GOHead(fused_dim, labels_num, dropout)
        self.cell_gate_head = GOHead(fused_dim, labels_num, dropout)
        self.graph_gate_head = GOHead(fused_dim, labels_num, dropout)
        self.cell_trust_head = CellTrustHead(fused_dim, dropout)

    def _interpro_for_nodes(self, input_nodes: torch.Tensor, interpro_features: torch.Tensor, global_to_interpro: torch.Tensor) -> torch.Tensor:
        row_idx = global_to_interpro[input_nodes]
        out = torch.zeros((input_nodes.shape[0], interpro_features.shape[1]), dtype=interpro_features.dtype, device=interpro_features.device)
        valid = row_idx >= 0
        if valid.any():
            out[valid] = interpro_features[row_idx[valid]]
        return out

    def _cell_context(self, batch: CellPFPInput) -> torch.Tensor:
        safe_cell_indices = batch.cell_indices.clamp(min=0)
        cell_context = self.cell_embedding(safe_cell_indices)
        degree_values = torch.log1p(torch.clamp(batch.cell_degrees.float(), min=0.0))
        degree_values = degree_values.masked_fill(~batch.cell_mask, 0.0)
        degree_scale = degree_values.max(dim=1, keepdim=True).values.clamp(min=1.0)
        degree_feature = degree_values / degree_scale
        rank_values = torch.clamp(batch.cell_ranks.float(), min=1.0)
        rank_feature = torch.where(batch.cell_mask, 1.0 / rank_values, torch.zeros_like(rank_values))
        return torch.cat([cell_context, degree_feature.unsqueeze(-1), rank_feature.unsqueeze(-1)], dim=-1)

    def forward(
        self, batch: CellPFPInput, interpro_features: torch.Tensor,
        global_to_interpro: torch.Tensor, node_interpro: Optional[torch.Tensor] = None,
    ):
        esm_vec = self.esm_branch(batch.esm_tokens, batch.esm_mask)
        batch_size = esm_vec.shape[0]
        num_cells = batch.cell_mask.shape[1]
        device = esm_vec.device
        cell_vectors = torch.zeros((batch_size, num_cells, self.fused_dim), device=device)
        for group in batch.graph_groups:
            if node_interpro is None:
                raw_interpro = self._interpro_for_nodes(group.input_nodes, interpro_features, global_to_interpro)
                interpro_vec = self.interpro_encoder(raw_interpro)
            else:
                interpro_vec = node_interpro[group.input_nodes]
            cell_vec = self.cell_embedding(group.cell_idx).expand(interpro_vec.shape[0], -1)
            gcn_input = torch.cat([interpro_vec, cell_vec], dim=-1)
            gcn_out = self.gcn(group.blocks, gcn_input)
            fused = self.graph_to_fused(gcn_out)
            cell_vectors[group.batch_indices, group.slot_indices] = fused
        cell_context = self._cell_context(batch)
        cell_vectors = self.cell_modulator(cell_vectors, cell_context, batch.cell_mask)
        cell_trust_scores = self.cell_trust_head(cell_vectors)
        cell_trust_scores = cell_trust_scores.masked_fill(~batch.cell_mask, 0.0)
        seq_logits = self.seq_go_head(esm_vec)
        cell_logits = self.cell_go_head(cell_vectors)
        cell_gate_scores = self.cell_gate_head(cell_vectors)
        cell_gate_scores = cell_gate_scores + self.trust_alpha * cell_trust_scores.unsqueeze(-1)
        cell_mask = batch.cell_mask.unsqueeze(-1)
        cell_gate_scores = cell_gate_scores.masked_fill(~cell_mask, -1000000000.0)
        cell_source_weights = F.softmax(cell_gate_scores, dim=1)
        cell_source_weights = cell_source_weights * cell_mask.float()
        cell_source_weights = cell_source_weights / cell_source_weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
        graph_logits = torch.sum(cell_source_weights * cell_logits, dim=1)
        valid_count = batch.cell_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        graph_summary = torch.sum(cell_vectors * batch.cell_mask.unsqueeze(-1).float(), dim=1)
        graph_summary = graph_summary / valid_count
        seq_branch_scores = self.seq_gate_head(esm_vec).unsqueeze(1)
        graph_branch_scores = self.graph_gate_head(graph_summary).unsqueeze(1)
        branch_scores = torch.cat([seq_branch_scores, graph_branch_scores], dim=1)
        branch_mask = torch.cat([torch.ones((batch_size, 1), dtype=torch.bool, device=device), batch.cell_mask.any(dim=1, keepdim=True)], dim=1)
        branch_scores = branch_scores.masked_fill(~branch_mask.unsqueeze(-1), -1000000000.0)
        branch_source_weights = F.softmax(branch_scores, dim=1)
        final_logits = branch_source_weights[:, 0, :] * seq_logits + branch_source_weights[:, 1, :] * graph_logits
        return final_logits
