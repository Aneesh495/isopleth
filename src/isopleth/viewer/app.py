"""Scientific visualization workbench and data presentation models (I28).

Generates structured visualization payloads for spatiotemporal rollout fields,
spectral energy cascades, conservation invariant balances, sensor masks,
inverse optimization traces, and conformal prediction bands.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
import io
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from isopleth.data.contracts import PhysicalFamily
from isopleth.inverse.observations import ObservationRecord


@dataclass
class WaveformSnapshot1D:
    """1D field snapshot at a given point in time."""

    time: float
    step_index: int
    coordinates: List[float]
    values_predicted: List[float]
    values_ground_truth: Optional[List[float]] = None
    lower_conformal_bound: Optional[List[float]] = None
    upper_conformal_bound: Optional[List[float]] = None
    sensor_indices: Optional[List[int]] = None


@dataclass
class FieldSnapshot2D:
    """2D field snapshot at a given point in time."""

    time: float
    step_index: int
    nx: int
    ny: int
    channel_names: List[str]
    predicted_channels: Dict[str, List[List[float]]]
    ground_truth_channels: Optional[Dict[str, List[List[float]]]] = None
    sensor_coords: Optional[List[Tuple[int, int]]] = None


@dataclass
class SpectralCascadePoint:
    """Spectral energy value for a given Fourier wavenumber."""

    wavenumber: int
    energy_predicted: float
    energy_ground_truth: Optional[float] = None
    kolmogorov_slope_ref: Optional[float] = None


@dataclass
class ConservationBalancePoint:
    """Mass and momentum integral accounting across time."""

    time: float
    step_index: int
    mass_total: float
    mass_drift_fraction: float
    momentum_x: Optional[float] = None
    momentum_y: Optional[float] = None


@dataclass
class InverseOptimizationStep:
    """Progress snapshot of inverse reconstruction iteration."""

    iteration: int
    loss: float
    observation_loss: float
    regularization_loss: float
    gradient_norm: float
    estimated_parameters: Dict[str, float]


@dataclass
class ViewerPayload:
    """Self-contained interactive laboratory state payload."""

    system_family: str
    spatial_dimension: int
    grid_shape: List[int]
    time_step: float
    total_steps: int
    snapshots_1d: Optional[List[WaveformSnapshot1D]] = None
    snapshots_2d: Optional[List[FieldSnapshot2D]] = None
    spectral_cascade: List[SpectralCascadePoint] = field(default_factory=list)
    conservation_balances: List[ConservationBalancePoint] = field(default_factory=list)
    inverse_optimization_trace: Optional[List[InverseOptimizationStep]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serializes viewer payload to canonical JSON."""
        return json.dumps(asdict(self), indent=2)


class ScientificVisualizationBuilder:
    """Constructs visualization payloads from PyTorch rollout tensors and simulation records."""

    @staticmethod
    def build_1d_payload(
        predicted_trajectory: torch.Tensor,
        ground_truth_trajectory: Optional[torch.Tensor] = None,
        dt: float = 0.01,
        conformal_lower: Optional[torch.Tensor] = None,
        conformal_upper: Optional[torch.Tensor] = None,
        observation: Optional[ObservationRecord] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ViewerPayload:
        """Constructs 1D spatiotemporal visualization payload."""
        # Expected shape: [T+1, nx] or [B=1, T+1, C=1, nx]
        pred_sq = predicted_trajectory.detach().cpu().squeeze().numpy()
        gt_sq = ground_truth_trajectory.detach().cpu().squeeze().numpy() if ground_truth_trajectory is not None else None
        low_sq = conformal_lower.detach().cpu().squeeze().numpy() if conformal_lower is not None else None
        high_sq = conformal_upper.detach().cpu().squeeze().numpy() if conformal_upper is not None else None

        T_total, nx = pred_sq.shape[0], pred_sq.shape[1]
        coords = np.linspace(0.0, 1.0, nx, endpoint=False).tolist()

        sensor_list: Optional[List[int]] = None
        if observation is not None:
            sensor_list = observation.sensor_indices.cpu().tolist()

        snapshots: List[WaveformSnapshot1D] = []
        balances: List[ConservationBalancePoint] = []

        m0 = float(np.sum(pred_sq[0]))

        for t_idx in range(T_total):
            curr_pred = pred_sq[t_idx].tolist()
            curr_gt = gt_sq[t_idx].tolist() if gt_sq is not None else None
            curr_low = low_sq[t_idx].tolist() if low_sq is not None else None
            curr_high = high_sq[t_idx].tolist() if high_sq is not None else None

            time_val = float(t_idx * dt)
            snapshots.append(
                WaveformSnapshot1D(
                    time=time_val,
                    step_index=t_idx,
                    coordinates=coords,
                    values_predicted=curr_pred,
                    values_ground_truth=curr_gt,
                    lower_conformal_bound=curr_low,
                    upper_conformal_bound=curr_high,
                    sensor_indices=sensor_list,
                )
            )

            # Mass accounting
            m_t = float(np.sum(pred_sq[t_idx]))
            drift = abs(m_t - m0) / (abs(m0) + 1e-12)
            balances.append(
                ConservationBalancePoint(
                    time=time_val,
                    step_index=t_idx,
                    mass_total=m_t,
                    mass_drift_fraction=drift,
                )
            )

        # Spectral energy cascade of final state
        last_pred = pred_sq[-1]
        fft_pred = np.abs(np.fft.rfft(last_pred)) ** 2
        fft_gt = np.abs(np.fft.rfft(gt_sq[-1])) ** 2 if gt_sq is not None else None

        cascade: List[SpectralCascadePoint] = []
        for k in range(len(fft_pred)):
            k_val = k
            e_pred = float(fft_pred[k])
            e_gt = float(fft_gt[k]) if fft_gt is not None else None
            slope_ref = (1.0 / (k_val + 1.0) ** (5.0 / 3.0)) if k_val > 0 else 1.0
            cascade.append(
                SpectralCascadePoint(
                    wavenumber=k_val,
                    energy_predicted=e_pred,
                    energy_ground_truth=e_gt,
                    kolmogorov_slope_ref=slope_ref,
                )
            )

        return ViewerPayload(
            system_family=PhysicalFamily.BURGERS.value,
            spatial_dimension=1,
            grid_shape=[nx],
            time_step=dt,
            total_steps=T_total - 1,
            snapshots_1d=snapshots,
            spectral_cascade=cascade,
            conservation_balances=balances,
            metadata=metadata or {},
        )

    @staticmethod
    def build_2d_payload(
        predicted_trajectory: torch.Tensor,
        ground_truth_trajectory: Optional[torch.Tensor] = None,
        channel_names: Optional[List[str]] = None,
        dt: float = 0.005,
        observation: Optional[ObservationRecord] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ViewerPayload:
        """Constructs 2D spatiotemporal visualization payload."""
        # Expected shape: [T+1, C, ny, nx]
        pred_np = predicted_trajectory.detach().cpu().squeeze(0).numpy() if predicted_trajectory.ndim == 5 else predicted_trajectory.detach().cpu().numpy()
        gt_np = ground_truth_trajectory.detach().cpu().squeeze(0).numpy() if (ground_truth_trajectory is not None and ground_truth_trajectory.ndim == 5) else (ground_truth_trajectory.detach().cpu().numpy() if ground_truth_trajectory is not None else None)

        T_total, C, ny, nx = pred_np.shape[0], pred_np.shape[1], pred_np.shape[2], pred_np.shape[3]
        ch_names = channel_names or [f"channel_{c}" for c in range(C)]

        sensor_coords: Optional[List[Tuple[int, int]]] = None
        if observation is not None:
            lin_idx = observation.sensor_indices.cpu().numpy()
            sensor_coords = [(int(idx // nx), int(idx % nx)) for idx in lin_idx]

        snapshots: List[FieldSnapshot2D] = []
        balances: List[ConservationBalancePoint] = []

        m0 = float(np.sum(pred_np[0, 0]))

        # Subsample snapshots if T_total is large to keep payload lightweight
        stride = max(1, T_total // 25)

        for t_idx in range(0, T_total, stride):
            time_val = float(t_idx * dt)
            pred_ch_dict: Dict[str, List[List[float]]] = {}
            gt_ch_dict: Optional[Dict[str, List[List[float]]]] = {} if gt_np is not None else None

            for c_idx, name in enumerate(ch_names):
                pred_ch_dict[name] = pred_np[t_idx, c_idx].tolist()
                if gt_np is not None and gt_ch_dict is not None:
                    gt_ch_dict[name] = gt_np[t_idx, c_idx].tolist()

            snapshots.append(
                FieldSnapshot2D(
                    time=time_val,
                    step_index=t_idx,
                    nx=nx,
                    ny=ny,
                    channel_names=ch_names,
                    predicted_channels=pred_ch_dict,
                    ground_truth_channels=gt_ch_dict,
                    sensor_coords=sensor_coords,
                )
            )

            m_t = float(np.sum(pred_np[t_idx, 0]))
            drift = abs(m_t - m0) / (abs(m0) + 1e-12)
            mom_x = float(np.sum(pred_np[t_idx, 1])) if C >= 2 else 0.0
            mom_y = float(np.sum(pred_np[t_idx, 2])) if C >= 3 else 0.0

            balances.append(
                ConservationBalancePoint(
                    time=time_val,
                    step_index=t_idx,
                    mass_total=m_t,
                    mass_drift_fraction=drift,
                    momentum_x=mom_x,
                    momentum_y=mom_y,
                )
            )

        return ViewerPayload(
            system_family=PhysicalFamily.SHALLOW_WATER.value,
            spatial_dimension=2,
            grid_shape=[ny, nx],
            time_step=dt,
            total_steps=T_total - 1,
            snapshots_2d=snapshots,
            conservation_balances=balances,
            metadata=metadata or {},
        )
