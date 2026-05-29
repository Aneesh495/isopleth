"""Tests for Hamiltonian Monte Carlo and Bayesian inverse posterior sampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from isopleth.inverse.bayesian_mcmc import (
    BayesianSurrogateDistortionAnalyzer,
    HMCSampleChain,
    HamiltonianMonteCarloSampler,
    SurrogatePosteriorDistortionReport,
)


def quadratic_potential(q: torch.Tensor) -> torch.Tensor:
    """Standard harmonic oscillator potential U(q) = 0.5 * ||q||^2."""
    return 0.5 * torch.sum(q**2)


def test_hmc_symplectic_leapfrog_energy_conservation() -> None:
    sampler = HamiltonianMonteCarloSampler(
        potential_fn=quadratic_potential,
        step_size=0.01,
        num_leapfrog_steps=20,
        seed=42,
    )
    q0 = torch.tensor([1.5, -2.0], dtype=torch.float64)
    p0 = torch.tensor([0.5, 1.0], dtype=torch.float64)

    h0 = quadratic_potential(q0).item() + 0.5 * torch.sum(p0**2).item()
    q_prop, p_prop = sampler.leapfrog(q0, p0)
    h_prop = quadratic_potential(q_prop).item() + 0.5 * torch.sum(p_prop**2).item()

    # Symplectic integrator should conserve energy within tight tolerance
    assert abs(h_prop - h0) < 1e-3


def test_hmc_sample_chain_gaussian_target() -> None:
    sampler = HamiltonianMonteCarloSampler(
        potential_fn=quadratic_potential,
        step_size=0.05,
        num_leapfrog_steps=10,
        seed=123,
    )
    q_init = torch.zeros(2, dtype=torch.float64)
    chain = sampler.sample_chain(q_init, num_samples=60, burn_in=10)

    assert isinstance(chain, HMCSampleChain)
    assert chain.samples.shape == (60, 2)
    assert 0.0 <= chain.acceptance_rate <= 1.0
    assert chain.effective_sample_size >= 1.0
    assert len(chain.log_posterior_history) == 60


def test_bayesian_surrogate_distortion_analyzer() -> None:
    rng = np.random.default_rng(42)
    samples_ref = rng.normal(loc=0.0, scale=1.0, size=(50, 4))
    samples_surr = rng.normal(loc=0.1, scale=1.05, size=(50, 4))

    chain_ref = HMCSampleChain(
        samples=samples_ref,
        acceptance_rate=0.85,
        hamiltonian_energy_drift=0.02,
        effective_sample_size=35.0,
        log_posterior_history=[-1.0] * 50,
    )
    chain_surr = HMCSampleChain(
        samples=samples_surr,
        acceptance_rate=0.80,
        hamiltonian_energy_drift=0.05,
        effective_sample_size=30.0,
        log_posterior_history=[-1.2] * 50,
    )

    report = BayesianSurrogateDistortionAnalyzer.compare_chains(chain_surr, chain_ref)
    assert isinstance(report, SurrogatePosteriorDistortionReport)
    assert report.wasserstein_distance >= 0.0
    assert report.acceptance_rate_ratio > 0.0
