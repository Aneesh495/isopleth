"""Tests for conservative depth limiter and diagnostics (I13)."""

import pytest
import torch

from isopleth.numerics.limiters import ConservativeDepthLimiter

def test_conservative_depth_limiter_mass_preservation():
    limiter = ConservativeDepthLimiter(h_min=1e-4)
    # 2D field with positive background and local negative dips
    h = torch.ones(1, 16, 16) * 0.1
    h[0, 5, 5] = -0.02  # Negative dip!
    h[0, 8, 8] = -0.01  # Another negative dip!

    orig_mass = torch.sum(h)
    h_lim, _, _, diag = limiter(h)

    assert diag.activations_count == 2
    assert diag.is_safe is True
    # Guaranteed non-negativity: h >= h_min everywhere
    assert torch.all(h_lim >= 1e-4)

    # Strict conservation of total mass
    final_mass = torch.sum(h_lim)
    assert torch.abs(final_mass - orig_mass) < 1e-6

def test_limiter_velocity_desaturation():
    limiter = ConservativeDepthLimiter(h_min=1e-4)
    h = torch.ones(1, 8, 8) * 0.1
    h[0, 2, 2] = 1e-6  # Extremely shallow dry cell!
    hu = torch.ones(1, 8, 8) * 0.5
    hv = torch.ones(1, 8, 8) * 0.5

    h_lim, hu_lim, hv_lim, _ = limiter(h, hu, hv)
    # Momentum in near-dry cell should be desaturated towards zero
    assert hu_lim[0, 2, 2] < 0.1
    assert hv_lim[0, 2, 2] < 0.1
