"""Scientific diagnostics, spectral dissipation, and hypothesis testing.

Provides mathematical diagnostics for evaluating physics-informed neural operators:
1. Spectral energy dissipation and enstrophy cascade analysis.
2. Long-horizon conservation drift power spectrum (Fourier analysis of mass error).
3. Statistical hypothesis testing suite (paired Wilcoxon signed-rank, Kolmogorov-Smirnov).
4. Inverse surrogate Hessian distortion and eigenspectrum alignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn


@dataclass
class SpectralDissipationMetrics:
    """Quantitative metrics describing kinetic energy cascade and dissipation."""

    wavenumbers: np.ndarray
    energy_spectrum: np.ndarray
    spectral_slope: float
    total_kinetic_energy: float
    dissipation_rate: float
    enstrophy: float


@dataclass
class DriftPowerSpectrumMetrics:
    """Spectral decomposition of long-horizon cumulative conservation drift."""

    drift_frequencies: np.ndarray
    power_spectral_density: np.ndarray
    peak_resonance_frequency: float
    secular_drift_rate: float
    oscillatory_drift_fraction: float


@dataclass
class StatisticalComparisonReport:
    """Outcomes of rigorous hypothesis tests comparing competing models."""

    model_a_name: str
    model_b_name: str
    sample_size: int
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    is_significant: bool
    alpha: float
    ks_statistic: float
    ks_p_value: float
    coverage_violation_rate: float
    binomial_coverage_p_value: float


@dataclass
class HessianDistortionReport:
    """Eigenspectrum alignment metrics between surrogate and numerical Hessians."""

    surrogate_eigenvalues: np.ndarray
    reference_eigenvalues: np.ndarray
    eigenvalue_relative_error: float
    spectral_condition_number_surrogate: float
    spectral_condition_number_reference: float
    surrogate_distortion_index: float
    subspace_overlap_angle_deg: float


class SpectralDissipationAnalyzer:
    """Analyzes spatial kinetic energy spectra, spectral slopes, and enstrophy."""

    def __init__(self, domain_length: float = 1.0) -> None:
        self.L = float(domain_length)

    def compute_spectrum_1d(
        self, u: Union[np.ndarray, torch.Tensor], dx: Optional[float] = None
    ) -> SpectralDissipationMetrics:
        """Compute 1D spatial kinetic energy spectrum E(k).

        Args:
            u: 1D velocity/state array of shape (N,) or (B, N).
            dx: Spatial resolution step. If None, L / N is used.

        Returns:
            SpectralDissipationMetrics containing spectrum and fitted slope.
        """
        if isinstance(u, torch.Tensor):
            arr = u.detach().cpu().numpy()
        else:
            arr = np.asarray(u, dtype=np.float64)

        if arr.ndim == 1:
            arr = arr[None, :]

        batch_size, n_cells = arr.shape
        if dx is None:
            dx = self.L / float(n_cells)

        # FFT along spatial axis
        fft_vals = np.fft.rfft(arr, axis=-1)
        k_indices = np.fft.rfftfreq(n_cells, d=dx)

        # Kinetic energy per mode: 0.5 * |u_hat|^2 / n_cells^2
        energy_modes = 0.5 * np.mean(np.abs(fft_vals) ** 2, axis=0) / (n_cells**2)
        total_ke = float(np.sum(energy_modes))

        # Enstrophy in 1D: integral of (du/dx)^2 ~ sum(k^2 * E(k))
        enstrophy = float(np.sum((2.0 * np.pi * k_indices) ** 2 * energy_modes))

        # Fit spectral power-law slope log(E(k)) ~ slope * log(k) in inertial range
        valid_mask = (k_indices > 2.0) & (k_indices < (n_cells / 4.0)) & (energy_modes > 1e-15)
        if np.sum(valid_mask) > 3:
            log_k = np.log(k_indices[valid_mask])
            log_e = np.log(energy_modes[valid_mask])
            slope, _ = np.polyfit(log_k, log_e, deg=1)
        else:
            slope = 0.0

        dissipation_rate = float(np.mean(np.diff(energy_modes)))

        return SpectralDissipationMetrics(
            wavenumbers=k_indices,
            energy_spectrum=energy_modes,
            spectral_slope=float(slope),
            total_kinetic_energy=total_ke,
            dissipation_rate=dissipation_rate,
            enstrophy=enstrophy,
        )

    def compute_spectrum_2d(
        self,
        u: Union[np.ndarray, torch.Tensor],
        v: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> SpectralDissipationMetrics:
        """Compute isotropic radially averaged 2D energy spectrum."""
        if isinstance(u, torch.Tensor):
            arr_u = u.detach().cpu().numpy()
        else:
            arr_u = np.asarray(u, dtype=np.float64)

        if arr_u.ndim == 2:
            arr_u = arr_u[None, :, :]

        _, ny, nx = arr_u.shape
        fft_u = np.fft.fft2(arr_u, axes=(-2, -1))
        energy_2d = 0.5 * np.mean(np.abs(fft_u) ** 2, axis=0) / ((nx * ny) ** 2)

        if v is not None:
            arr_v = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            if arr_v.ndim == 2:
                arr_v = arr_v[None, :, :]
            fft_v = np.fft.fft2(arr_v, axes=(-2, -1))
            energy_2d += 0.5 * np.mean(np.abs(fft_v) ** 2, axis=0) / ((nx * ny) ** 2)

        # Radially average 2D spectrum into 1D wavenumber bins
        kx = np.fft.fftfreq(nx, d=self.L / nx)
        ky = np.fft.fftfreq(ny, d=self.L / ny)
        k_radius = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)

        k_bins = np.arange(0.5, min(nx, ny) // 2, 1.0)
        e_radial = np.zeros(len(k_bins) - 1, dtype=np.float64)

        for i in range(len(k_bins) - 1):
            mask = (k_radius >= k_bins[i]) & (k_radius < k_bins[i + 1])
            if np.any(mask):
                e_radial[i] = np.mean(energy_2d[mask])

        mid_k = 0.5 * (k_bins[:-1] + k_bins[1:])
        valid = (mid_k > 2.0) & (e_radial > 1e-15)
        if np.sum(valid) > 3:
            slope, _ = np.polyfit(np.log(mid_k[valid]), np.log(e_radial[valid]), deg=1)
        else:
            slope = 0.0

        total_ke = float(np.sum(energy_2d))
        enstrophy = float(np.sum((2.0 * np.pi * mid_k) ** 2 * e_radial))

        return SpectralDissipationMetrics(
            wavenumbers=mid_k,
            energy_spectrum=e_radial,
            spectral_slope=float(slope),
            total_kinetic_energy=total_ke,
            dissipation_rate=0.0,
            enstrophy=enstrophy,
        )


class ConservationDriftPowerSpectrum:
    """Fourier analyzes cumulative mass and momentum drift over long rollouts."""

    @staticmethod
    def analyze_drift_series(
        mass_drift_series: Union[np.ndarray, Sequence[float]], dt: float = 0.01
    ) -> DriftPowerSpectrumMetrics:
        """Decompose conservation drift into secular trend vs oscillatory resonance."""
        series = np.asarray(mass_drift_series, dtype=np.float64)
        n_steps = len(series)
        times = np.arange(n_steps) * dt

        # 1. Fit linear secular drift rate: drift(t) ~ secular_rate * t
        if n_steps > 2:
            secular_rate, _ = np.polyfit(times, series, deg=1)
        else:
            secular_rate = 0.0

        # 2. Detrend series to isolate periodic fluctuations
        detrended = series - secular_rate * times
        fft_vals = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(n_steps, d=dt)
        psd = (np.abs(fft_vals) ** 2) / float(n_steps)

        # Skip zero frequency (DC component)
        if len(psd) > 1:
            peak_idx = int(np.argmax(psd[1:])) + 1
            peak_freq = float(freqs[peak_idx])
            oscillatory_var = float(np.var(detrended))
            total_var = float(np.var(series) + 1e-15)
            osc_fraction = min(1.0, oscillatory_var / total_var)
        else:
            peak_freq = 0.0
            osc_fraction = 0.0

        return DriftPowerSpectrumMetrics(
            drift_frequencies=freqs,
            power_spectral_density=psd,
            peak_resonance_frequency=peak_freq,
            secular_drift_rate=float(secular_rate),
            oscillatory_drift_fraction=osc_fraction,
        )


class StatisticalHypothesisTester:
    """Conducts distribution-free paired hypothesis tests on rollout errors."""

    @staticmethod
    def compare_models(
        errors_model_a: Sequence[float],
        errors_model_b: Sequence[float],
        model_a_name: str = "MFFNO",
        model_b_name: str = "Baseline",
        alpha: float = 0.05,
    ) -> StatisticalComparisonReport:
        """Compare error distributions of two models across matched evaluation trajectories.

        Uses:
        - Wilcoxon signed-rank test for paired differences.
        - Two-sample Kolmogorov-Smirnov test for overall CDF equality.
        """
        arr_a = np.asarray(errors_model_a, dtype=np.float64)
        arr_b = np.asarray(errors_model_b, dtype=np.float64)

        if len(arr_a) != len(arr_b):
            raise ValueError(f"Sample sizes must match: {len(arr_a)} vs {len(arr_b)}")

        n_samples = len(arr_a)
        diff = arr_a - arr_b

        # Wilcoxon signed-rank test
        if np.all(diff == 0.0):
            w_stat, w_p = 0.0, 1.0
        else:
            w_res = stats.wilcoxon(arr_a, arr_b, alternative="two-sided")
            w_stat = float(w_res.statistic)
            w_p = float(w_res.pvalue)

        # Kolmogorov-Smirnov test
        ks_res = stats.ks_2samp(arr_a, arr_b)
        ks_stat = float(ks_res.statistic)
        ks_p = float(ks_res.pvalue)

        is_sig = w_p < alpha

        return StatisticalComparisonReport(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            sample_size=n_samples,
            wilcoxon_statistic=w_stat,
            wilcoxon_p_value=w_p,
            is_significant=is_sig,
            alpha=alpha,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            coverage_violation_rate=0.0,
            binomial_coverage_p_value=1.0,
        )

    @staticmethod
    def test_conformal_coverage(
        covered_indicator: Sequence[bool], nominal_coverage: float = 0.90
    ) -> Tuple[float, float, bool]:
        """Perform two-sided exact binomial test for conformal calibration validity."""
        k_success = int(np.sum(covered_indicator))
        n_trials = len(covered_indicator)
        emp_coverage = k_success / float(n_trials)

        res = stats.binomtest(k=k_success, n=n_trials, p=nominal_coverage, alternative="two-sided")
        p_val = float(res.pvalue)
        passed = p_val >= 0.05

        return emp_coverage, p_val, passed


class SurrogateHessianDistortionAnalyzer:
    """Evaluates surrogate vs true numerical objective curvature and Hessian eigenspectra."""

    @staticmethod
    def compute_hessian_matrix(
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        """Compute numerical Hessian via central finite differences of autograd gradients."""
        x_flat = x.detach().clone().requires_grad_(True).reshape(-1)
        dim = x_flat.numel()
        hessian = np.zeros((dim, dim), dtype=np.float64)

        for i in range(dim):
            # Forward perturbation
            x_plus = x_flat.clone().detach()
            x_plus[i] += epsilon
            x_plus.requires_grad_(True)
            loss_p = loss_fn(x_plus.view_as(x))
            grad_p = torch.autograd.grad(loss_p, x_plus)[0].reshape(-1)

            # Backward perturbation
            x_minus = x_flat.clone().detach()
            x_minus[i] -= epsilon
            x_minus.requires_grad_(True)
            loss_m = loss_fn(x_minus.view_as(x))
            grad_m = torch.autograd.grad(loss_m, x_minus)[0].reshape(-1)

            # Central difference of gradient
            col = (grad_p - grad_m) / (2.0 * epsilon)
            hessian[:, i] = col.detach().cpu().numpy()

        # Enforce exact symmetry
        return 0.5 * (hessian + hessian.T)

    @staticmethod
    def compare_hessians(
        hessian_surrogate: np.ndarray, hessian_reference: np.ndarray
    ) -> HessianDistortionReport:
        """Compare eigenspectrum and principal curvature directions."""
        evals_surr, evecs_surr = np.linalg.eigh(hessian_surrogate)
        evals_ref, evecs_ref = np.linalg.eigh(hessian_reference)

        # Sort eigenvalues descending
        evals_surr = np.sort(evals_surr)[::-1]
        evals_ref = np.sort(evals_ref)[::-1]

        rel_eig_err = float(
            np.linalg.norm(evals_surr - evals_ref) / (np.linalg.norm(evals_ref) + 1e-12)
        )

        cond_surr = float(
            (np.max(np.abs(evals_surr)) + 1e-12) / (np.min(np.abs(evals_surr)) + 1e-12)
        )
        cond_ref = float(
            (np.max(np.abs(evals_ref)) + 1e-12) / (np.min(np.abs(evals_ref)) + 1e-12)
        )

        # Principal eigenvector subspace overlap: angle between leading eigenvectors
        leading_surr = evecs_surr[:, -1]
        leading_ref = evecs_ref[:, -1]
        cos_angle = abs(np.dot(leading_surr, leading_ref))
        cos_angle = min(1.0, max(0.0, cos_angle))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))

        # Surrogate distortion index: 1 - cosine similarity of flattened Hessians
        flat_s = hessian_surrogate.ravel()
        flat_r = hessian_reference.ravel()
        denom = (np.linalg.norm(flat_s) * np.linalg.norm(flat_r)) + 1e-12
        cos_sim = float(np.dot(flat_s, flat_r) / denom)
        distortion_idx = max(0.0, 1.0 - cos_sim)

        return HessianDistortionReport(
            surrogate_eigenvalues=evals_surr,
            reference_eigenvalues=evals_ref,
            eigenvalue_relative_error=rel_eig_err,
            spectral_condition_number_surrogate=cond_surr,
            spectral_condition_number_reference=cond_ref,
            surrogate_distortion_index=distortion_idx,
            subspace_overlap_angle_deg=angle_deg,
        )
