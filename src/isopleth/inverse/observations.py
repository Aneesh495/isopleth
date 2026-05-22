"""Sparse observation operators and noise contracts (I21).

Applies spatial point-sensor subsampling, temporal decimation, and calibrated Gaussian noise.
Prevents unobserved state leakage into inverse problems via strict boolean masks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch

@dataclass
class ObservationRecord:
    observed_values: torch.Tensor    # [T_obs, n_sensors, ...]
    mask: torch.Tensor               # Boolean mask [T_total, ...]
    sensor_indices: torch.Tensor     # Sensor coordinate or linear indices
    times_observed: torch.Tensor     # Actual observation timestamps
    noise_sigma: float
    temporal_subsample_stride: int
    spatial_coverage_fraction: float

class SparseObservationOperator:
    """Simulates sparse point sensors and noisy temporal measurements."""

    def __init__(
        self,
        spatial_fraction: float = 0.10,
        temporal_stride: int = 5,
        noise_std: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.spatial_fraction = spatial_fraction
        self.temporal_stride = temporal_stride
        self.noise_std = noise_std
        self.seed = seed

    def observe_1d(
        self,
        trajectory: torch.Tensor,
        times: torch.Tensor,
    ) -> ObservationRecord:
        """Observe 1D trajectory [T+1, nx]."""
        T_total, nx = trajectory.shape[-2], trajectory.shape[-1]
        rng = np.random.default_rng(self.seed)

        # Select sparse sensor locations
        n_sensors = max(2, int(round(nx * self.spatial_fraction)))
        sensor_idx = np.sort(rng.choice(nx, size=n_sensors, replace=False))
        sensor_tensor = torch.tensor(sensor_idx, dtype=torch.long, device=trajectory.device)

        # Subsample time
        time_indices = np.arange(0, T_total, self.temporal_stride)
        time_tensor = torch.tensor(time_indices, dtype=torch.long, device=trajectory.device)

        # Spatial-temporal mask
        mask = torch.zeros_like(trajectory, dtype=torch.bool)
        for t_idx in time_indices:
            mask[t_idx, sensor_tensor] = True

        # Extract values and add Gaussian noise
        raw_obs = trajectory[time_tensor][:, sensor_tensor]
        noise = torch.randn_like(raw_obs) * self.noise_std
        observed_noisy = raw_obs + noise

        return ObservationRecord(
            observed_values=observed_noisy,
            mask=mask,
            sensor_indices=sensor_tensor,
            times_observed=times[time_tensor],
            noise_sigma=self.noise_std,
            temporal_subsample_stride=self.temporal_stride,
            spatial_coverage_fraction=self.spatial_fraction,
        )

    def observe_2d(
        self,
        trajectory: torch.Tensor,
        times: torch.Tensor,
    ) -> ObservationRecord:
        """Observe 2D trajectory [T+1, C, ny, nx]."""
        T_total = trajectory.shape[0]
        C, ny, nx = trajectory.shape[1], trajectory.shape[2], trajectory.shape[3]
        total_cells = ny * nx
        rng = np.random.default_rng(self.seed)

        n_sensors = max(4, int(round(total_cells * self.spatial_fraction)))
        linear_sensor_idx = np.sort(rng.choice(total_cells, size=n_sensors, replace=False))
        sensor_tensor = torch.tensor(linear_sensor_idx, dtype=torch.long, device=trajectory.device)

        time_indices = np.arange(0, T_total, self.temporal_stride)
        time_tensor = torch.tensor(time_indices, dtype=torch.long, device=trajectory.device)

        mask = torch.zeros_like(trajectory, dtype=torch.bool)
        for t_idx in time_indices:
            for s_idx in linear_sensor_idx:
                r = s_idx // nx
                c = s_idx % nx
                mask[t_idx, :, r, c] = True

        # Extract observations
        flat_spatial = trajectory.view(T_total, C, total_cells)
        raw_obs = flat_spatial[time_tensor][:, :, sensor_tensor]  # [T_obs, C, n_sensors]
        noise = torch.randn_like(raw_obs) * self.noise_std
        observed_noisy = raw_obs + noise

        return ObservationRecord(
            observed_values=observed_noisy,
            mask=mask,
            sensor_indices=sensor_tensor,
            times_observed=times[time_tensor],
            noise_sigma=self.noise_std,
            temporal_subsample_stride=self.temporal_stride,
            spatial_coverage_fraction=self.spatial_fraction,
        )
