import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.optimized_dkt import LayerNormResidualDKT, TemporalAttentionDKT


def test_layernorm_residual_dkt_shape_and_gradients():
    model = LayerNormResidualDKT(skill_count=11, embedding_dim=16, hidden_dim=16, dropout=0.1)
    tokens = torch.randint(0, 22, (3, 7))
    logits = model(tokens)
    assert logits.shape == (3, 7, 11)
    logits.mean().backward()
    assert model.output.weight.grad is not None


def test_temporal_attention_dkt_shape_with_padding_mask():
    model = TemporalAttentionDKT(skill_count=11, embedding_dim=16, hidden_dim=16, dropout=0.1, heads=4)
    tokens = torch.randint(0, 22, (3, 7))
    padding = torch.zeros((3, 7), dtype=torch.bool)
    padding[:, -1] = True
    logits = model(tokens, padding_mask=padding)
    assert logits.shape == (3, 7, 11)
    assert torch.isfinite(logits).all()
