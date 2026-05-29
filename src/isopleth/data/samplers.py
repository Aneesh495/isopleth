"""Continuous and discrete initial condition random field generators.

Provides mathematically rigorous initial state samplers for 1D and 2D PDEs:
1. Gaussian Random Fields (GRF) with Matérn / power-law spectral covariance.
2. Karhunen-Loeve expansion modal synthesizers.
3. Shallow water topography and water surface state samplers (Thacker, dambreak, seamounts).
4. Gray-Scott Turing morphometry perturbation seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


@dataclass(frozen=True)
class SamplerMetadata:
    """Metadata describing a generated initial condition field."""

    field_name: str
    dimension: int
    num_samples: int
    resolution: Tuple[int, ...]
    parameters: Dict[str, float]


class GaussianRandomField1D:
    """1D Periodic Gaussian Random Field generator using spectral synthesis.

    Power spectrum follows Matérn-like power law:
        P(k) = amplitude * (k^2 + tau^2)^(-alpha / 2)
    where tau controls correlation length-scale and alpha controls smoothness.
    """

    def __init__(
        self,
        tau: float = 3.0,
        alpha: float = 2.5,
        amplitude: float = 1.0,
        mean: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.amplitude = float(amplitude)
        self.mean = float(mean)
        self.rng = np.random.default_rng(seed)

    def sample_continuous(self, x: np.ndarray, num_modes: int = 64) -> np.ndarray:
        """Sample continuous field via truncated Fourier series at arbitrary points.

        Args:
            x: Spatial coordinates in [0, 1], shape (N,) or (B, N).
            num_modes: Number of Fourier modes to retain.

        Returns:
            Field values evaluated at x.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        k_indices = np.arange(1, num_modes + 1, dtype=np.float64)
        power = self.amplitude * np.power(k_indices**2 + self.tau**2, -self.alpha / 2.0)

        # Standard normal coefficients for real and imaginary parts
        xi_real = self.rng.standard_normal(num_modes)
        xi_imag = self.rng.standard_normal(num_modes)
        coeff = np.sqrt(power) * (xi_real + 1j * xi_imag) / np.sqrt(2.0)

        # Evaluate sum of cosines and sines
        # shape of k_indices[:, None] * 2 * pi * x_arr[None, :]
        phase = 2.0 * np.pi * np.outer(k_indices, x_arr)
        field = np.sum(
            coeff.real[:, None] * np.cos(phase) - coeff.imag[:, None] * np.sin(phase),
            axis=0,
        )
        return field * np.sqrt(2.0) + self.mean

    def sample_grid(self, n_cells: int, num_samples: int = 1) -> np.ndarray:
        """Sample periodic fields on a uniform grid using Fast Fourier Transform.

        Args:
            n_cells: Number of grid cells.
            num_samples: Number of independent realizations.

        Returns:
            Real array of shape (num_samples, n_cells).
        """
        k = np.fft.fftfreq(n_cells, d=1.0 / n_cells)
        power = self.amplitude * np.power(k**2 + self.tau**2, -self.alpha / 2.0)
        power[0] = 0.0  # Zero mean fluctuation

        # Generate complex Gaussian noise
        xi_r = self.rng.standard_normal((num_samples, n_cells))
        xi_i = self.rng.standard_normal((num_samples, n_cells))
        xi = (xi_r + 1j * xi_i) / np.sqrt(2.0)

        fourier_coeff = xi * np.sqrt(power)[None, :] * n_cells
        fields = np.fft.ifft(fourier_coeff, axis=-1).real + self.mean
        return fields


class GaussianRandomField2D:
    """2D Periodic Gaussian Random Field generator using spectral synthesis.

    Power spectrum follows 2D isotropic power law:
        P(k_x, k_y) = amplitude * (k_x^2 + k_y^2 + tau^2)^(-alpha / 2)
    """

    def __init__(
        self,
        tau: float = 3.0,
        alpha: float = 2.5,
        amplitude: float = 1.0,
        mean: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.amplitude = float(amplitude)
        self.mean = float(mean)
        self.rng = np.random.default_rng(seed)

    def sample_grid(
        self, nx: int, ny: int, num_samples: int = 1
    ) -> np.ndarray:
        """Sample 2D periodic fields on a uniform grid using 2D FFT.

        Args:
            nx: Number of cells in x dimension.
            ny: Number of cells in y dimension.
            num_samples: Number of independent realizations.

        Returns:
            Real array of shape (num_samples, nx, ny).
        """
        kx = np.fft.fftfreq(nx, d=1.0 / nx)
        ky = np.fft.fftfreq(ny, d=1.0 / ny)
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
        k_sq = kx_grid**2 + ky_grid**2

        power = self.amplitude * np.power(k_sq + self.tau**2, -self.alpha / 2.0)
        power[0, 0] = 0.0  # Zero out DC component

        xi_r = self.rng.standard_normal((num_samples, nx, ny))
        xi_i = self.rng.standard_normal((num_samples, nx, ny))
        xi = (xi_r + 1j * xi_i) / np.sqrt(2.0)

        fourier_coeff = xi * np.sqrt(power)[None, :, :] * (nx * ny)
        fields = np.fft.ifft2(fourier_coeff, axes=(-2, -1)).real + self.mean
        return fields


class KarhunenLoeveExpander1D:
    """1D Karhunen-Loeve expansion synthesizer with prescribed covariance spectrum.

    Constructs random fields as linear combinations of orthogonal eigenfunctions:
        u(x) = u_0(x) + sum_k sqrt(lambda_k) * xi_k * phi_k(x)
    where phi_k are Fourier modes or Chebyshev basis functions.
    """

    def __init__(
        self,
        num_modes: int = 32,
        decay_rate: float = 2.0,
        basis_type: str = "fourier",
        seed: Optional[int] = None,
    ) -> None:
        self.num_modes = num_modes
        self.decay_rate = decay_rate
        self.basis_type = basis_type.lower()
        self.rng = np.random.default_rng(seed)

        # Eigenvalues decaying as k^(-decay_rate)
        k_vals = np.arange(1, num_modes + 1, dtype=np.float64)
        self.eigenvalues = np.power(k_vals, -decay_rate)
        self.eigenvalues /= np.sum(self.eigenvalues)

    def evaluate_basis(self, x: np.ndarray) -> np.ndarray:
        """Evaluate spatial basis functions on query points x in [0, 1].

        Returns array of shape (num_modes, len(x)).
        """
        x_flat = np.asarray(x, dtype=np.float64).ravel()
        n_pts = len(x_flat)
        basis = np.zeros((self.num_modes, n_pts), dtype=np.float64)

        if self.basis_type == "fourier":
            for k in range(self.num_modes):
                mode_num = k // 2 + 1
                if k % 2 == 0:
                    basis[k] = np.sqrt(2.0) * np.cos(2.0 * np.pi * mode_num * x_flat)
                else:
                    basis[k] = np.sqrt(2.0) * np.sin(2.0 * np.pi * mode_num * x_flat)
        elif self.basis_type == "chebyshev":
            # Map [0, 1] to [-1, 1]
            xi = 2.0 * x_flat - 1.0
            for k in range(self.num_modes):
                basis[k] = np.cos(k * np.arccos(np.clip(xi, -1.0, 1.0)))
        else:
            raise ValueError(f"Unknown basis_type: {self.basis_type}")

        return basis

    def sample(self, x: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """Sample Karhunen-Loeve realizations at coordinates x."""
        basis = self.evaluate_basis(x)  # (modes, points)
        weights = self.rng.standard_normal((num_samples, self.num_modes))
        scaled_weights = weights * np.sqrt(self.eigenvalues)[None, :]
        fields = scaled_weights @ basis
        return fields


class RiemannDamBreakSampler:
    """Sampler for 1D and 2D shallow water dambreak configurations.

    Generates smooth or discontinuous initial water surface elevations
    and initial velocities with randomized breach geometry.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_1d(
        self,
        n_cells: int,
        h_left_range: Tuple[float, float] = (1.5, 3.0),
        h_right_range: Tuple[float, float] = (0.5, 1.2),
        transition_width: float = 0.02,
        num_samples: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 1D dambreak profiles.

        Returns:
            Tuple of (h, hu) arrays of shape (num_samples, n_cells).
        """
        x = np.linspace(0.0, 1.0, n_cells, endpoint=False) + 0.5 / n_cells
        h_batch = np.zeros((num_samples, n_cells), dtype=np.float64)
        hu_batch = np.zeros((num_samples, n_cells), dtype=np.float64)

        for i in range(num_samples):
            h_left = self.rng.uniform(*h_left_range)
            h_right = self.rng.uniform(*h_right_range)
            x_mid = self.rng.uniform(0.35, 0.65)

            if transition_width <= 0:
                h = np.where(x < x_mid, h_left, h_right)
            else:
                # Smooth tanh transition for numerical regularity
                arg = (x - x_mid) / transition_width
                h = h_right + (h_left - h_right) * 0.5 * (1.0 - np.tanh(arg))

            h_batch[i] = h
            hu_batch[i] = 0.0  # Dam break starts from rest

        return h_batch, hu_batch

    def sample_2d_radial(
        self,
        nx: int,
        ny: int,
        h_in_range: Tuple[float, float] = (1.8, 2.5),
        h_out_range: Tuple[float, float] = (0.8, 1.2),
        radius_range: Tuple[float, float] = (0.15, 0.30),
        transition_width: float = 0.03,
        num_samples: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate 2D circular dam collapse configurations.

        Returns:
            Tuple of (h, hu, hv) arrays of shape (num_samples, nx, ny).
        """
        x = np.linspace(0.0, 1.0, nx, endpoint=False) + 0.5 / nx
        y = np.linspace(0.0, 1.0, ny, endpoint=False) + 0.5 / ny
        xx, yy = np.meshgrid(x, y, indexing="ij")

        h_batch = np.zeros((num_samples, nx, ny), dtype=np.float64)
        hu_batch = np.zeros((num_samples, nx, ny), dtype=np.float64)
        hv_batch = np.zeros((num_samples, nx, ny), dtype=np.float64)

        for i in range(num_samples):
            cx = self.rng.uniform(0.4, 0.6)
            cy = self.rng.uniform(0.4, 0.6)
            radius = self.rng.uniform(*radius_range)
            h_in = self.rng.uniform(*h_in_range)
            h_out = self.rng.uniform(*h_out_range)

            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            if transition_width <= 0:
                h = np.where(dist < radius, h_in, h_out)
            else:
                arg = (dist - radius) / transition_width
                h = h_out + (h_in - h_out) * 0.5 * (1.0 - np.tanh(arg))

            h_batch[i] = h
            # Initial momentum is zero (still reservoir before release)
            hu_batch[i] = 0.0
            hv_batch[i] = 0.0

        return h_batch, hu_batch, hv_batch


class ShallowWaterTopographySampler:
    """Generates synthetic bottom bathymetry profiles for 1D and 2D domains.

    Includes:
    - Isolated and multiple Gaussian seamounts.
    - Parabolic bowl topography (Thacker basin compatible).
    - Smooth underwater ridges and shelves.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_1d_seamount(
        self,
        n_cells: int,
        peak_height_range: Tuple[float, float] = (0.2, 0.5),
        width_range: Tuple[float, float] = (0.05, 0.15),
        num_mounts: int = 1,
    ) -> np.ndarray:
        """Sample 1D Gaussian hump topography."""
        x = np.linspace(0.0, 1.0, n_cells, endpoint=False) + 0.5 / n_cells
        b = np.zeros(n_cells, dtype=np.float64)

        for _ in range(num_mounts):
            h_peak = self.rng.uniform(*peak_height_range)
            sigma = self.rng.uniform(*width_range)
            cx = self.rng.uniform(0.3, 0.7)
            b += h_peak * np.exp(-((x - cx) ** 2) / (2.0 * sigma**2))

        return b

    def sample_2d_seamount(
        self,
        nx: int,
        ny: int,
        peak_height_range: Tuple[float, float] = (0.2, 0.5),
        width_range: Tuple[float, float] = (0.08, 0.20),
        num_mounts: int = 2,
    ) -> np.ndarray:
        """Sample 2D Gaussian topography hump array."""
        x = np.linspace(0.0, 1.0, nx, endpoint=False) + 0.5 / nx
        y = np.linspace(0.0, 1.0, ny, endpoint=False) + 0.5 / ny
        xx, yy = np.meshgrid(x, y, indexing="ij")
        b = np.zeros((nx, ny), dtype=np.float64)

        for _ in range(num_mounts):
            h_peak = self.rng.uniform(*peak_height_range)
            sigma_x = self.rng.uniform(*width_range)
            sigma_y = self.rng.uniform(*width_range)
            cx = self.rng.uniform(0.25, 0.75)
            cy = self.rng.uniform(0.25, 0.75)
            b += h_peak * np.exp(
                -((xx - cx) ** 2 / (2.0 * sigma_x**2) + (yy - cy) ** 2 / (2.0 * sigma_y**2))
            )

        return b

    def sample_thacker_parabolic_bowl(
        self,
        nx: int,
        ny: int,
        h0: float = 0.1,
        a: float = 1.0,
    ) -> np.ndarray:
        """Thacker planar oscillating bowl bathymetry: b(x, y) = -h0 * (1 - r^2 / a^2)."""
        x = np.linspace(-1.0, 1.0, nx)
        y = np.linspace(-1.0, 1.0, ny)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        r_sq = xx**2 + yy**2
        b = -h0 * (1.0 - r_sq / (a**2))
        return b


class GrayScottSeedSampler:
    """Generates initial perturbation fields for 2D Gray-Scott reaction-diffusion.

    Background state is homogeneous unreacted medium:
        u(x, y) = 1.0,  v(x, y) = 0.0
    Perturbations introduce non-zero concentration of species v to seed
    autocatalytic spot, stripe, or spiral morphogenesis.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_central_square(
        self,
        nx: int,
        ny: int,
        patch_size: float = 0.15,
        noise_level: float = 0.02,
        num_samples: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate standard central square seed state.

        Returns:
            Tuple of (u, v) arrays with shape (num_samples, nx, ny).
        """
        x = np.linspace(0.0, 1.0, nx, endpoint=False)
        y = np.linspace(0.0, 1.0, ny, endpoint=False)
        xx, yy = np.meshgrid(x, y, indexing="ij")

        u_batch = np.ones((num_samples, nx, ny), dtype=np.float64)
        v_batch = np.zeros((num_samples, nx, ny), dtype=np.float64)

        for i in range(num_samples):
            half = patch_size / 2.0
            cx = self.rng.uniform(0.45, 0.55)
            cy = self.rng.uniform(0.45, 0.55)
            mask = (
                (xx >= cx - half)
                & (xx <= cx + half)
                & (yy >= cy - half)
                & (yy <= cy + half)
            )

            u = np.ones((nx, ny), dtype=np.float64)
            v = np.zeros((nx, ny), dtype=np.float64)

            u[mask] = 0.50
            v[mask] = 0.25

            if noise_level > 0:
                noise_u = self.rng.uniform(-noise_level, noise_level, (nx, ny))
                noise_v = self.rng.uniform(-noise_level, noise_level, (nx, ny))
                u += noise_u
                v += noise_v

            # Enforce physical positivity and species upper bound [0, 1]
            u_batch[i] = np.clip(u, 0.0, 1.0)
            v_batch[i] = np.clip(v, 0.0, 1.0)

        return u_batch, v_batch

    def sample_multi_spot(
        self,
        nx: int,
        ny: int,
        num_spots: int = 5,
        spot_radius: float = 0.04,
        num_samples: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate multi-spot seed state with randomized centers."""
        x = np.linspace(0.0, 1.0, nx, endpoint=False)
        y = np.linspace(0.0, 1.0, ny, endpoint=False)
        xx, yy = np.meshgrid(x, y, indexing="ij")

        u_batch = np.ones((num_samples, nx, ny), dtype=np.float64)
        v_batch = np.zeros((num_samples, nx, ny), dtype=np.float64)

        for i in range(num_samples):
            u = np.ones((nx, ny), dtype=np.float64)
            v = np.zeros((nx, ny), dtype=np.float64)

            for _ in range(num_spots):
                cx = self.rng.uniform(0.2, 0.8)
                cy = self.rng.uniform(0.2, 0.8)
                dist_sq = (xx - cx)**2 + (yy - cy)**2
                mask = dist_sq < (spot_radius**2)
                u[mask] = 0.50
                v[mask] = 0.25

            u_batch[i] = np.clip(u, 0.0, 1.0)
            v_batch[i] = np.clip(v, 0.0, 1.0)

        return u_batch, v_batch


def sample_initial_conditions_torch(
    sampler_type: str,
    resolution: Union[int, Tuple[int, ...]],
    num_samples: int = 1,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    **sampler_kwargs,
) -> torch.Tensor:
    """Convenience bridge returning sampled initial conditions as PyTorch tensors.

    Args:
        sampler_type: One of 'grf_1d', 'grf_2d', 'kl_1d', 'dambreak_1d',
                      'dambreak_2d', 'gray_scott'.
        resolution: Spatial resolution integer (1D) or tuple (2D).
        num_samples: Number of batch items.
        seed: Random seed.
        device: Target PyTorch device.
        dtype: Target PyTorch floating-point dtype.
        **sampler_kwargs: Additional parameters forwarded to the underlying sampler.

    Returns:
        Tensor of shape (B, C, N) for 1D or (B, C, H, W) for 2D.
    """
    sampler_type = sampler_type.lower()

    if sampler_type == "grf_1d":
        n_cells = int(resolution) if isinstance(resolution, (int, float)) else int(resolution[0])
        gen = GaussianRandomField1D(seed=seed, **sampler_kwargs)
        arr = gen.sample_grid(n_cells, num_samples=num_samples)
        # shape: (B, 1, N)
        tensor = torch.from_numpy(arr[:, None, :]).to(dtype=dtype, device=device)
        return tensor

    elif sampler_type == "grf_2d":
        if isinstance(resolution, int):
            nx = ny = resolution
        else:
            nx, ny = int(resolution[0]), int(resolution[1])
        gen = GaussianRandomField2D(seed=seed, **sampler_kwargs)
        arr = gen.sample_grid(nx, ny, num_samples=num_samples)
        # shape: (B, 1, H, W)
        tensor = torch.from_numpy(arr[:, None, :, :]).to(dtype=dtype, device=device)
        return tensor

    elif sampler_type == "kl_1d":
        n_cells = int(resolution) if isinstance(resolution, (int, float)) else int(resolution[0])
        x = np.linspace(0.0, 1.0, n_cells, endpoint=False)
        gen = KarhunenLoeveExpander1D(seed=seed, **sampler_kwargs)
        arr = gen.sample(x, num_samples=num_samples)
        # shape: (B, 1, N)
        tensor = torch.from_numpy(arr[:, None, :]).to(dtype=dtype, device=device)
        return tensor

    elif sampler_type == "dambreak_1d":
        n_cells = int(resolution) if isinstance(resolution, (int, float)) else int(resolution[0])
        gen = RiemannDamBreakSampler(seed=seed)
        h, hu = gen.sample_1d(n_cells, num_samples=num_samples, **sampler_kwargs)
        stacked = np.stack([h, hu], axis=1)  # shape: (B, 2, N)
        return torch.from_numpy(stacked).to(dtype=dtype, device=device)

    elif sampler_type == "dambreak_2d":
        if isinstance(resolution, int):
            nx = ny = resolution
        else:
            nx, ny = int(resolution[0]), int(resolution[1])
        gen = RiemannDamBreakSampler(seed=seed)
        h, hu, hv = gen.sample_2d_radial(nx, ny, num_samples=num_samples, **sampler_kwargs)
        stacked = np.stack([h, hu, hv], axis=1)  # shape: (B, 3, H, W)
        return torch.from_numpy(stacked).to(dtype=dtype, device=device)

    elif sampler_type == "gray_scott":
        if isinstance(resolution, int):
            nx = ny = resolution
        else:
            nx, ny = int(resolution[0]), int(resolution[1])
        gen = GrayScottSeedSampler(seed=seed)
        u, v = gen.sample_central_square(nx, ny, num_samples=num_samples, **sampler_kwargs)
        stacked = np.stack([u, v], axis=1)  # shape: (B, 2, H, W)
        return torch.from_numpy(stacked).to(dtype=dtype, device=device)

    else:
        raise ValueError(f"Unsupported sampler_type: {sampler_type}")
