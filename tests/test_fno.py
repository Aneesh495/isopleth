"""Tests for Fourier Neural Operator baseline (I08)."""

import pytest
import torch

from isopleth.models.fno import FNO1D, FNO2D

def test_fno_1d_forward_and_resolutions():
    model = FNO1D(in_channels=1, out_channels=1, hidden_channels=32, modes=8, num_layers=2)
    
    # Train grid resolution 64
    x_64 = torch.randn(2, 1, 64)
    out_64 = model(x_64)
    assert out_64.shape == (2, 64)

    # Zero-shot cross-grid evaluation on resolution 128
    x_128 = torch.randn(2, 1, 128)
    out_128 = model(x_128)
    assert out_128.shape == (2, 128)

def test_fno_2d_forward_and_resolutions():
    model = FNO2D(in_channels=3, out_channels=3, hidden_channels=24, modes_x=8, modes_y=8, num_layers=2)
    
    # Resolution 32x32
    x_32 = torch.randn(2, 3, 32, 32)
    out_32 = model(x_32)
    assert out_32.shape == (2, 3, 32, 32)

    # Resolution 48x48
    x_48 = torch.randn(2, 3, 48, 48)
    out_48 = model(x_48)
    assert out_48.shape == (2, 3, 48, 48)
