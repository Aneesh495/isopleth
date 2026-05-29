"""Tests for spectral dissipation, drift spectrum, and hypothesis testing diagnostics."""

import numpy as np
import pytest
import torch

from isopleth.evaluation.diagnostics import (
    ConservationDriftPowerSpectrum,
    SpectralDissipationAnalyzer,
    StatisticalHypothesisTester,
    SurrogateHessianDistortionAnalyzer,
)


def test_spectral_dissipation_analyzer_1d_and_2d():
    analyzer = SpectralDissipationAnalyzer(domain_length=1.0)

    # 1D
    x = np.linspace(0, 1.0, 64)
    u_1d = np.sin(2.0 * np.pi * x) + 0.5 * np.sin(6.0 * np.pi * x)
    m_1d = analyzer.compute_spectrum_1d(u_1d)
    assert len(m_1d.wavenumbers) == 33
    assert m_1d.total_kinetic_energy > 0.0
    assert m_1d.enstrophy > 0.0

    # 2D
    u_2d = np.random.randn(32, 32)
    m_2d = analyzer.compute_spectrum_2d(u_2d)
    assert len(m_2d.energy_spectrum) > 0
    assert m_2d.total_kinetic_energy > 0.0


def test_conservation_drift_power_spectrum():
    # Construct synthetic drift with secular slope and 5 Hz oscillation
    t = np.linspace(0, 1.0, 100)
    drift = 0.02 * t + 0.005 * np.sin(2.0 * np.pi * 5.0 * t)

    metrics = ConservationDriftPowerSpectrum.analyze_drift_series(drift, dt=0.01)
    assert abs(metrics.secular_drift_rate - 0.02) < 0.01
    assert metrics.oscillatory_drift_fraction > 0.0


def test_statistical_hypothesis_tester():
    errors_a = [0.01, 0.015, 0.012, 0.020, 0.018, 0.014]
    errors_b = [0.05, 0.048, 0.055, 0.060, 0.052, 0.058]

    report = StatisticalHypothesisTester.compare_models(errors_a, errors_b, "ModelA", "ModelB")
    assert report.is_significant
    assert report.wilcoxon_p_value < 0.05

    # Conformal exact binomial test
    hits = [True] * 92 + [False] * 8  # 92% coverage out of 100
    emp_cov, p_val, passed = StatisticalHypothesisTester.test_conformal_coverage(hits, nominal_coverage=0.90)
    assert emp_cov == 0.92
    assert passed


def test_surrogate_hessian_distortion_analyzer():
    def simple_loss(w: torch.Tensor) -> torch.Tensor:
        return torch.sum(2.0 * w**2 + 0.5 * w**4)

    w = torch.tensor([0.5, -0.2], dtype=torch.float64)
    h_surr = SurrogateHessianDistortionAnalyzer.compute_hessian_matrix(simple_loss, w)
    assert h_surr.shape == (2, 2)
    # Symmetry check
    assert np.allclose(h_surr, h_surr.T)

    report = SurrogateHessianDistortionAnalyzer.compare_hessians(h_surr, h_surr)
    assert report.eigenvalue_relative_error < 1e-10
    assert report.surrogate_distortion_index < 1e-10
