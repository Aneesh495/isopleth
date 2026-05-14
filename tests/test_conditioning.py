"""Tests for parameter and lead-time conditioning (I14)."""

import pytest
import torch

from isopleth.models.conditioning import (
    FiLMBlock,
    FourierEmbedding,
    PhysicalConditioningEncoder,
)

def test_fourier_embedding():
    emb = FourierEmbedding(embed_dim=32)
    dt = torch.tensor([0.01, 0.05, 0.1])
    out = emb(dt)
    assert out.shape == (3, 32)
    assert torch.all(torch.isfinite(out))

def test_film_block_1d_and_2d():
    film_1d = FiLMBlock(feature_dim=16, condition_dim=32)
    x_1d = torch.randn(2, 16, 64)
    cond = torch.randn(2, 32)
    y_1d = film_1d(x_1d, cond)
    assert y_1d.shape == (2, 16, 64)

    film_2d = FiLMBlock(feature_dim=16, condition_dim=32)
    x_2d = torch.randn(2, 16, 32, 32)
    y_2d = film_2d(x_2d, cond)
    assert y_2d.shape == (2, 16, 32, 32)

def test_physical_conditioning_encoder():
    enc = PhysicalConditioningEncoder(num_physical_params=3, embed_dim=32, out_dim=64)
    params = torch.tensor([[0.01, 9.81, 0.4], [0.02, 9.81, 0.5]])
    dt = torch.tensor([0.005, 0.01])
    latent = enc(params, dt)
    assert latent.shape == (2, 64)
    assert torch.all(torch.isfinite(latent))
