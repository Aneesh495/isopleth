"""Tests for spatial spectral analysis, structure functions, and enstrophy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from isopleth.numerics.spectral_analysis import (
    AutocorrelationResult,
    EnstrophyDissipationAnalyzer2D,
    SpatialAutocorrelationAnalyzer,
    SpatialStructureFunctionAnalyzer,
    StructureFunctionResult,
)


def test_spatial_structure_function_analyzer() -> None:
    analyzer = SpatialStructureFunctionAnalyzer(domain_length=1.0)
    x = np.linspace(0.0, 1.0, 64, endpoint=False)
    u = np.sin(2.0 * np.pi * x)

    res = analyzer.compute_structure_functions_1d(u, orders=(1, 2))
    assert isinstance(res, StructureFunctionResult)
    assert len(res.separation_distances) > 0
    assert 1 in res.structure_functions
    assert 2 in res.structure_functions


def test_spatial_autocorrelation_analyzer() -> None:
    analyzer = SpatialAutocorrelationAnalyzer(domain_length=1.0)
    x = np.linspace(0.0, 1.0, 64, endpoint=False)
    u = np.cos(4.0 * np.pi * x)

    res = analyzer.compute_autocorrelation_1d(u)
    assert isinstance(res, AutocorrelationResult)
    assert res.autocorrelation[0] == pytest.approx(1.0, abs=1e-5)
    assert res.integral_length_scale >= 0.0
    assert res.taylor_microscale > 0.0


def test_enstrophy_dissipation_analyzer_2d() -> None:
    analyzer = EnstrophyDissipationAnalyzer2D(domain_length=1.0)
    ny, nx = 32, 32
    y, x = np.mgrid[0:1:32j, 0:1:32j]
    u = -np.sin(2.0 * np.pi * y)
    v = np.cos(2.0 * np.pi * x)

    metrics = analyzer.compute_vorticity_and_enstrophy(u, v, viscosity=1e-3)
    assert metrics["mean_enstrophy"] > 0.0
    assert metrics["peak_vorticity"] > 0.0
    assert metrics["palinstrophy"] > 0.0
    assert metrics["enstrophy_dissipation_rate"] > 0.0
