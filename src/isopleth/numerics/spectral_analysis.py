"""Spatial spectral analysis, turbulence cascades, and structure functions.

Provides numerical tools for spatial energy spectra, structure functions,
enstrophy dissipation, and turbulence length scales in conservation laws:
1. 1D and 2D energy spectrum calculations.
2. Radially averaged isotropic spectra.
3. Spatial structure functions S_p(r) = <|u(x+r) - u(x)|^p>.
4. Spatial autocorrelation and integral / Taylor length scales.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


@dataclass
class StructureFunctionResult:
    """Outcomes of spatial structure function evaluation."""

    separation_distances: np.ndarray
    structure_functions: Dict[int, np.ndarray]
    scaling_exponents: Dict[int, float]


@dataclass
class AutocorrelationResult:
    """Outcomes of spatial two-point autocorrelation analysis."""

    lag_distances: np.ndarray
    autocorrelation: np.ndarray
    integral_length_scale: float
    taylor_microscale: float


class SpatialStructureFunctionAnalyzer:
    """Calculates spatial velocity structure functions and scaling exponents."""

    def __init__(self, domain_length: float = 1.0) -> None:
        self.L = float(domain_length)

    def compute_structure_functions_1d(
        self,
        u: Union[np.ndarray, torch.Tensor],
        orders: Sequence[int] = (1, 2, 3),
        max_lag_fraction: float = 0.5,
    ) -> StructureFunctionResult:
        """Compute structure functions S_p(r) = <|u(x+r) - u(x)|^p>.

        Args:
            u: 1D velocity field of shape (N,) or (B, N).
            orders: Sequence of structure function orders p.
            max_lag_fraction: Maximum separation distance as fraction of domain.

        Returns:
            StructureFunctionResult with S_p(r) curves and fitted scaling exponents.
        """
        arr = u.detach().cpu().numpy() if isinstance(u, torch.Tensor) else np.asarray(u, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]

        batch_size, n_cells = arr.shape
        dx = self.L / float(n_cells)
        max_lags = int(n_cells * max_lag_fraction)

        lag_indices = np.arange(1, max_lags + 1)
        r_distances = lag_indices * dx

        sf_dict: Dict[int, np.ndarray] = {p: np.zeros(len(lag_indices), dtype=np.float64) for p in orders}
        scaling_exponents: Dict[int, float] = {}

        for idx, lag in enumerate(lag_indices):
            # Difference with periodic wrap-around
            diff = np.roll(arr, shift=-lag, axis=-1) - arr
            abs_diff = np.abs(diff)

            for p in orders:
                sf_dict[p][idx] = float(np.mean(abs_diff**p))

        # Fit scaling exponents S_p(r) ~ r^(zeta_p) in inertial subrange
        # Typically between 3*dx and 0.2*L
        valid_mask = (r_distances >= 3.0 * dx) & (r_distances <= 0.25 * self.L)
        for p in orders:
            vals = sf_dict[p]
            if np.sum(valid_mask) > 3 and np.all(vals[valid_mask] > 1e-15):
                log_r = np.log(r_distances[valid_mask])
                log_sf = np.log(vals[valid_mask])
                slope, _ = np.polyfit(log_r, log_sf, deg=1)
                scaling_exponents[p] = float(slope)
            else:
                scaling_exponents[p] = 0.0

        return StructureFunctionResult(
            separation_distances=r_distances,
            structure_functions=sf_dict,
            scaling_exponents=scaling_exponents,
        )


class SpatialAutocorrelationAnalyzer:
    """Computes two-point spatial autocorrelations and characteristic scales."""

    def __init__(self, domain_length: float = 1.0) -> None:
        self.L = float(domain_length)

    def compute_autocorrelation_1d(
        self, u: Union[np.ndarray, torch.Tensor]
    ) -> AutocorrelationResult:
        """Compute spatial two-point autocorrelation function R(r) via Wiener-Khinchin theorem.

        Returns:
            AutocorrelationResult containing R(r), integral length scale, and Taylor microscale.
        """
        arr = u.detach().cpu().numpy() if isinstance(u, torch.Tensor) else np.asarray(u, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]

        batch_size, n_cells = arr.shape
        dx = self.L / float(n_cells)

        # Zero-mean fluctuations
        u_fluc = arr - np.mean(arr, axis=-1, keepdims=True)
        var = np.mean(u_fluc**2, axis=-1, keepdims=True) + 1e-15

        # FFT autocorrelation
        fft_u = np.fft.rfft(u_fluc, axis=-1)
        psd = np.abs(fft_u) ** 2
        autocorr_raw = np.fft.irfft(psd, n=n_cells, axis=-1)
        autocorr_norm = np.mean(autocorr_raw / (n_cells * var), axis=0)

        half_n = n_cells // 2
        r_lags = np.arange(half_n) * dx
        r_curve = autocorr_norm[:half_n]

        # 1. Integral length scale: area under autocorrelation curve until first zero crossing
        zero_crossings = np.where(r_curve <= 0.0)[0]
        trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        if len(zero_crossings) > 0:
            first_zero = zero_crossings[0]
            integral_scale = float(trap_fn(r_curve[:first_zero], r_lags[:first_zero]))
        else:
            integral_scale = float(trap_fn(r_curve, r_lags))

        integral_scale = max(0.0, integral_scale)

        # 2. Taylor microscale lambda: curvature at origin, R(r) ~ 1 - r^2 / (2 * lambda^2)
        # lambda = sqrt( -1 / R''(0) ) or lambda = sqrt( <u^2> / <(du/dx)^2> )
        du_dx = (np.roll(arr, shift=-1, axis=-1) - np.roll(arr, shift=1, axis=-1)) / (2.0 * dx)
        grad_var = np.mean(du_dx**2)
        mean_u_sq = np.mean(u_fluc**2)
        if grad_var > 1e-15:
            taylor_microscale = float(np.sqrt(mean_u_sq / grad_var))
        else:
            taylor_microscale = self.L

        return AutocorrelationResult(
            lag_distances=r_lags,
            autocorrelation=r_curve,
            integral_length_scale=integral_scale,
            taylor_microscale=taylor_microscale,
        )


class EnstrophyDissipationAnalyzer2D:
    """Computes 2D vorticity, enstrophy, and enstrophy dissipation rates."""

    def __init__(self, domain_length: float = 1.0) -> None:
        self.L = float(domain_length)

    def compute_vorticity_and_enstrophy(
        self,
        u: Union[np.ndarray, torch.Tensor],
        v: Union[np.ndarray, torch.Tensor],
        viscosity: float = 1e-3,
    ) -> Dict[str, float]:
        """Compute vorticity omega = dv/dx - du/dy and enstrophy Omega = 0.5 * integral(omega^2).

        Args:
            u: Zonal velocity array of shape (H, W).
            v: Meridional velocity array of shape (H, W).
            viscosity: Kinematic viscosity.

        Returns:
            Dictionary with mean_enstrophy, peak_vorticity, and dissipation_rate.
        """
        arr_u = u.detach().cpu().numpy() if isinstance(u, torch.Tensor) else np.asarray(u, dtype=np.float64)
        arr_v = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v, dtype=np.float64)

        ny, nx = arr_u.shape
        dx = self.L / float(nx)
        dy = self.L / float(ny)

        # Central difference gradients
        dv_dx = (np.roll(arr_v, shift=-1, axis=1) - np.roll(arr_v, shift=1, axis=1)) / (2.0 * dx)
        du_dy = (np.roll(arr_u, shift=-1, axis=0) - np.roll(arr_u, shift=1, axis=0)) / (2.0 * dy)

        # Vorticity omega = dv/dx - du/dy
        vorticity = dv_dx - du_dy
        enstrophy = 0.5 * np.mean(vorticity**2)

        # Palinstrophy P = 0.5 * integral(|grad omega|^2)
        domega_dx = (np.roll(vorticity, shift=-1, axis=1) - np.roll(vorticity, shift=1, axis=1)) / (2.0 * dx)
        domega_dy = (np.roll(vorticity, shift=-1, axis=0) - np.roll(vorticity, shift=1, axis=0)) / (2.0 * dy)
        palinstrophy = 0.5 * np.mean(domega_dx**2 + domega_dy**2)

        # Enstrophy dissipation rate = 2 * nu * P
        enstrophy_dissipation = 2.0 * viscosity * palinstrophy

        return {
            "mean_enstrophy": float(enstrophy),
            "peak_vorticity": float(np.max(np.abs(vorticity))),
            "palinstrophy": float(palinstrophy),
            "enstrophy_dissipation_rate": float(enstrophy_dissipation),
        }
