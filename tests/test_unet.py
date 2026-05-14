"""Tests for U-Net baseline (I09)."""

import pytest
import torch

from isopleth.models.unet import UNet1D, UNet2D

def test_unet_1d_forward_and_odd_grids():
    model = UNet1D(in_channels=1, out_channels=1, base_channels=16)
    
    # Even grid 64
    x_64 = torch.randn(2, 1, 64)
    out_64 = model(x_64)
    assert out_64.shape == (2, 64)

    # Odd grid 65
    x_65 = torch.randn(2, 1, 65)
    out_65 = model(x_65)
    assert out_65.shape == (2, 65)

def test_unet_2d_forward():
    model = UNet2D(in_channels=3, out_channels=3, base_channels=16)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 3, 32, 32)
