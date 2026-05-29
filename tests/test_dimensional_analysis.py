"""Tests for dimensional analysis and similitude transforms."""

from __future__ import annotations

import pytest
import torch

from isopleth.numerics.dimensional_analysis import (
    BurgersDimensionalScales,
    BurgersSimilitude,
    GrayScottDimensionalScales,
    GrayScottSimilitude,
    ShallowWaterDimensionalScales,
    ShallowWaterSimilitude,
)


def test_burgers_similitude() -> None:
    scales = BurgersSimilitude.compute_scales(velocity_scale=2.0, length_scale=1.0, viscosity=0.02)
    assert isinstance(scales, BurgersDimensionalScales)
    assert scales.reynolds_number == pytest.approx(100.0)
    assert scales.advective_time == pytest.approx(0.5)

    u = torch.tensor([1.0, 2.0, 3.0])
    x = torch.tensor([0.0, 0.5, 1.0])
    u_star, x_star, t_star = BurgersSimilitude.to_dimensionless(u, x, t=0.25, scales=scales)
    assert torch.allclose(u_star, torch.tensor([0.5, 1.0, 1.5]))
    assert torch.allclose(x_star, x)

    u_restored = BurgersSimilitude.to_dimensional(u_star, scales=scales)
    assert torch.allclose(u_restored, u)


def test_shallow_water_similitude() -> None:
    scales = ShallowWaterSimilitude.compute_scales(
        depth_scale=1.0, velocity_scale=2.0, length_scale=10.0, gravity=9.81, coriolis_param=1e-4
    )
    assert isinstance(scales, ShallowWaterDimensionalScales)
    assert scales.froude_number > 0.0
    assert scales.wave_celerity == pytest.approx(3.132, rel=1e-2)

    h = torch.tensor([1.2, 1.0])
    hu = torch.tensor([2.4, 2.0])
    hv = torch.tensor([0.0, 0.1])

    h_star, hu_star, hv_star = ShallowWaterSimilitude.to_dimensionless(h, hu, hv, scales)
    h_rec, hu_rec, hv_rec = ShallowWaterSimilitude.to_dimensional(h_star, hu_star, hv_star, scales)
    assert torch.allclose(h_rec, h)
    assert torch.allclose(hu_rec, hu)
    assert torch.allclose(hv_rec, hv)


def test_gray_scott_similitude() -> None:
    scales = GrayScottSimilitude.compute_scales(
        domain_size=1.0,
        diffusion_u=2e-5,
        diffusion_v=1e-5,
        feed_rate=0.03,
        kill_rate=0.06,
    )
    assert isinstance(scales, GrayScottDimensionalScales)
    assert scales.damkoehler_number > 0.0
    assert scales.diffusive_time_u > 0.0
