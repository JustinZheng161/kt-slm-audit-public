"""Optional DKT improvements for the controlled student-disjoint protocol.

The module keeps the original interaction-token interface: token ids are
skill_id + outcome * skill_count and outputs are per-skill logits.  It does
not change the target, split, or evaluation metric.

Variants:
* LayerNormResidualDKT: GRU states are normalized and passed through a gated
  residual projection before the per-skill head.  This follows the general
  stabilization logic used in Pre-LN residual sequence models.
* TemporalAttentionDKT: a small causal self-attention block is added after the
  GRU.  Padding masks are accepted so no future interaction is visible.

These variants are implementation candidates; their performance must be
reported only after being run under the same fixed split and seeds as DKT.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class LayerNormResidualDKT(nn.Module):
    """GRU DKT with post-recurrent normalization and a gated residual block."""

    def __init__(self, skill_count: int, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(skill_count * 2, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, skill_count)

    def forward(self, tokens: Tensor) -> Tensor:
        states, _ = self.gru(self.embedding(tokens))
        normalized = self.norm(states)
        residual = self.residual(normalized)
        mixed = normalized + torch.sigmoid(self.gate) * residual
        return self.output(self.dropout(mixed))


class TemporalAttentionDKT(nn.Module):
    """GRU plus a causal, padding-aware attention refinement block."""

    def __init__(self, skill_count: int, embedding_dim: int, hidden_dim: int, dropout: float, heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.embedding = nn.Embedding(skill_count * 2, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, skill_count)

    def forward(self, tokens: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        states, _ = self.gru(self.embedding(tokens))
        length = states.shape[1]
        causal_mask = torch.triu(torch.ones(length, length, device=states.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(self.norm(states), self.norm(states), self.norm(states),
                                     attn_mask=causal_mask, key_padding_mask=padding_mask, need_weights=False)
        refined = states + attended
        refined = refined + self.ffn(self.norm(refined))
        return self.output(self.dropout(refined))
