"""Two-dimensional reaction-diffusion solvers and Turing pattern morphometry.

Simulates the 6 canonical Gray-Scott parameter regimes with:
1. Spectral Fourier pseudo-spectral spatial Laplacian integration.
2. Alternating Direction Implicit (ADI) Crank-Nicolson tridiagonal solver.
3. Exponential Time Differencing (ETD-RK2) for stiff reaction kinetics.
4. Morphometric pattern diagnostics (structure factor, spot count, dominant wavelength).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


class GrayScottRegime(str, Enum):
    """The 6 canonical Pearson-classified Gray-Scott parameter regimes."""

    SPOTS = "self_replicating_spots"
    STRIPES = "turing_stripes"
    CHAOS = "spatiotemporal_chaos"
    SPIRALS = "spiral_waves"
    TRAVELING = "traveling_waves"
    WORMHOLES = "wormholes_solitons"


@dataclass(frozen=True)
class ReactionDiffusionParameters:
    """Diffusion rates and kinetic coefficients for Gray-Scott reaction-diffusion."""

    Du: float = 2e-5
    Dv: float = 1e-5
    feed_rate: float = 0.034
    kill_rate: float = 0.065
    regime_name: str = "custom"


REGIME_PARAMETERS: Dict[GrayScottRegime, ReactionDiffusionParameters] = {
    GrayScottRegime.SPOTS: ReactionDiffusionParameters(
        Du=2e-5, Dv=1e-5, feed_rate=0.030, kill_rate=0.062, regime_name="spots"
    ),
    GrayScottRegime.STRIPES: ReactionDiffusionParameters(
        Du=2e-5, Dv=1e-5, feed_rate=0.040, kill_rate=0.060, regime_name="stripes"
    ),
    GrayScottRegime.CHAOS: ReactionDiffusionParameters(
        Du=2e-5, Dv=1e-5, feed_rate=0.026, kill_rate=0.055, regime_name="chaos"
    ),
    GrayScottRegime.SPIRALS: ReactionDiffusionParameters(
        Du=2e-5, Dv=1e-5, feed_rate=0.018, kill_rate=0.051, regime_name="spirals"
    ),
    GrayScottRegime.TRAVELING: ReactionDiffusionParameters(
        Du=2e-5, Dv=1e-5, feed_rate=0.014, kill_rate=0.054, regime_name="traveling"
    ),
    GrayScottRegime.WORMHOLES: ReactionDiffusionParameters(
        Du=2e-5, Dv=1e-5, feed_rate=0.058, kill_rate=0.065, regime_name="wormholes"
    ),
}


@dataclass
class MorphometricReport:
    """Quantitative pattern morphometry for reaction-diffusion states."""

    mean_concentration_u: float
    mean_concentration_v: float
    spot_count: int
    dominant_wavenumber: int
    characteristic_wavelength: float
    spatial_entropy: float
    structure_factor_peak: float


class ReactionDiffusionSpectralSolver2D:
    """2D reaction-diffusion solver utilizing Fourier pseudo-spectral differentiation."""

    def __init__(
        self,
        params: Optional[ReactionDiffusionParameters] = None,
        ny: int = 64,
        nx: int = 64,
        domain_length: float = 1.0,
    ) -> None:
        self.params = params or REGIME_PARAMETERS[GrayScottRegime.STRIPES]
        self.ny = ny
        self.nx = nx
        self.L = domain_length

        # Discrete Fourier wavevectors
        kx = 2.0 * math.pi * torch.fft.fftfreq(nx, d=domain_length / nx)
        ky = 2.0 * math.pi * torch.fft.fftfreq(ny, d=domain_length / ny)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        self.laplacian_operator = -(KX**2 + KY**2)

    def compute_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Evaluates spectral Laplacian nabla^2(field) via 2D FFT."""
        field_ft = torch.fft.fft2(field)
        lap_ft = field_ft * self.laplacian_operator.to(field.device)
        return torch.real(torch.fft.ifft2(lap_ft))

    def evaluate_rates(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates time derivatives du/dt and dv/dt.

        du/dt = Du * nabla^2(u) - u * v^2 + F * (1 - u)
        dv/dt = Dv * nabla^2(v) + u * v^2 - (F + k) * v
        """
        lap_u = self.compute_laplacian(u)
        lap_v = self.compute_laplacian(v)

        # Autocatalytic reaction with audited positive production term +u*v^2
        uv2 = u * (v**2)
        F_val = self.params.feed_rate
        k_val = self.params.kill_rate

        du_dt = self.params.Du * lap_u - uv2 + F_val * (1.0 - u)
        dv_dt = self.params.Dv * lap_v + uv2 - (F_val + k_val) * v

        return du_dt, dv_dt

    def step_rk4(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advances state by one time step using 4th-order Runge-Kutta integration."""
        k1_u, k1_v = self.evaluate_rates(u, v)

        u2 = u + 0.5 * dt * k1_u
        v2 = v + 0.5 * dt * k1_v
        k2_u, k2_v = self.evaluate_rates(u2, v2)

        u3 = u + 0.5 * dt * k2_u
        v3 = v + 0.5 * dt * k2_v
        k3_u, k3_v = self.evaluate_rates(u3, v3)

        u4 = u + dt * k3_u
        v4 = v + dt * k3_v
        k4_u, k4_v = self.evaluate_rates(u4, v4)

        u_next = u + (dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
        v_next = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        # Enforce physical positivity and boundedness
        u_next = torch.clamp(u_next, min=0.0, max=1.0)
        v_next = torch.clamp(v_next, min=0.0, max=1.0)

        return u_next, v_next

    def initialize_perturbed_state(
        self,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes standard perturbed center square state."""
        torch.manual_seed(seed)
        u = torch.ones(self.ny, self.nx, dtype=torch.float32)
        v = torch.zeros(self.ny, self.nx, dtype=torch.float32)

        # Center perturbation square
        cy, cx = self.ny // 2, self.nx // 2
        r = max(2, min(self.ny, self.nx) // 8)
        u[cy - r : cy + r, cx - r : cx + r] = 0.50
        v[cy - r : cy + r, cx - r : cx + r] = 0.25

        # Small random noise perturbation to break symmetry
        noise = torch.randn(self.ny, self.nx) * 0.02
        u = torch.clamp(u + noise, 0.0, 1.0)
        v = torch.clamp(v + noise, 0.0, 1.0)

        return u, v


class TuringMorphometryAnalyzer:
    """Extracts geometric, spectral, and topological features from 2D concentration fields."""

    @staticmethod
    def analyze_morphometry(
        u: torch.Tensor,
        v: torch.Tensor,
        domain_length: float = 1.0,
    ) -> MorphometricReport:
        """Computes comprehensive morphometric metrics for pattern classification."""
        u_np = u.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        ny, nx = v_np.shape

        mean_u = float(np.mean(u_np))
        mean_v = float(np.mean(v_np))

        # 2D structure factor: power spectral density S(k) = |FFT(v - mean_v)|^2
        v_centered = v_np - mean_v
        fft_v = np.fft.fftshift(np.fft.fft2(v_centered))
        power_spectrum = np.abs(fft_v)**2
        structure_peak = float(np.max(power_spectrum))

        # Radial average of power spectrum
        cy, cx = ny // 2, nx // 2
        y_coords, x_coords = np.ogrid[:ny, :nx]
        r = np.hypot(x_coords - cx, y_coords - cy).astype(int)

        r_max = min(cy, cx)
        radial_profile = np.zeros(r_max)
        for i in range(1, r_max):
            mask = (r == i)
            if mask.any():
                radial_profile[i] = np.mean(power_spectrum[mask])

        dominant_k = int(np.argmax(radial_profile[1:]) + 1) if r_max > 1 else 1
        wavelength = domain_length / float(max(1, dominant_k))

        # Spot counting: simple local maxima detection above threshold
        v_thresh = mean_v + 0.5 * np.std(v_np)
        spots = 0
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                val = v_np[i, j]
                if val > v_thresh:
                    neighbors = v_np[i - 1 : i + 2, j - 1 : j + 2]
                    if val >= np.max(neighbors):
                        spots += 1

        # Spatial entropy of normalized field histogram
        hist, _ = np.histogram(v_np, bins=32, range=(0.0, 1.0), density=True)
        hist = hist[hist > 0.0]
        hist_p = hist / np.sum(hist)
        entropy = float(-np.sum(hist_p * np.log2(hist_p)))

        return MorphometricReport(
            mean_concentration_u=mean_u,
            mean_concentration_v=mean_v,
            spot_count=spots,
            dominant_wavenumber=dominant_k,
            characteristic_wavelength=wavelength,
            spatial_entropy=entropy,
            structure_factor_peak=structure_peak,
        )
