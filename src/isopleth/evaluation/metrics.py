"""Comprehensive spatial, spectral, conservation, and stability evaluation metrics (I20).

Evaluates long-horizon rollouts using rigorous physical metrics, spectral power discrepancies,
Fourier phase errors, mass balance residuals, and stability diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

@dataclass
class EvaluationMetrics:
    # Spatial errors
    relative_l2_error: float
    l1_error: float
    linf_error: float
    # Spectral errors
    spectral_l1_mismatch: float
    phase_error_mean: float
    # Physical conservation residuals
    mass_conservation_residual: float
    max_mass_deviation: float
    # Stability diagnostics
    completed_horizon_steps: int
    is_stable: bool
    negative_depth_count: int
    blowup_event: bool

    def to_dict(self) -> Dict[str, float]:
        return {
            "relative_l2_error": self.relative_l2_error,
            "l1_error": self.l1_error,
            "linf_error": self.linf_error,
            "spectral_l1_mismatch": self.spectral_l1_mismatch,
            "phase_error_mean": self.phase_error_mean,
            "mass_conservation_residual": self.mass_conservation_residual,
            "max_mass_deviation": self.max_mass_deviation,
            "completed_horizon_steps": float(self.completed_horizon_steps),
            "is_stable": float(self.is_stable),
            "negative_depth_count": float(self.negative_depth_count),
            "blowup_event": float(self.blowup_event),
        }

class TrajectoryEvaluator:
    """Evaluates predicted physical trajectories against ground truth references."""

    @staticmethod
    def compute_spatial_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float]:
        """Compute relative L2, MAE L1, and Linf."""
        diff = pred - target
        norm_diff = torch.norm(diff)
        norm_target = torch.norm(target)
        rel_l2 = float((norm_diff / (norm_target + 1e-12)).item())
        l1 = float(torch.mean(torch.abs(diff)).item())
        linf = float(torch.max(torch.abs(diff)).item())
        return rel_l2, l1, linf

    @staticmethod
    def compute_spectral_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
        """Compute power spectral L1 difference and mean phase angle error."""
        # 1D or 2D FFT
        if pred.ndim == 1 or pred.ndim == 2:
            fft_p = torch.fft.rfft(pred, dim=-1)
            fft_t = torch.fft.rfft(target, dim=-1)
        else:
            fft_p = torch.fft.rfft2(pred, dim=(-2, -1))
            fft_t = torch.fft.rfft2(target, dim=(-2, -1))

        # Magnitude power spectrum difference
        mag_p = torch.abs(fft_p)
        mag_t = torch.abs(fft_t)
        spec_l1 = float(torch.mean(torch.abs(mag_p - mag_t)).item())

        # Phase angle error: angle(fft_p) - angle(fft_t)
        angle_p = torch.angle(fft_p)
        angle_t = torch.angle(fft_t)
        phase_diff = torch.abs(torch.atan2(torch.sin(angle_p - angle_t), torch.cos(angle_p - angle_t)))
        phase_err = float(torch.mean(phase_diff).item())

        return spec_l1, phase_err

    @staticmethod
    def compute_conservation_metrics(
        pred_trajectory: torch.Tensor,
        init_state: torch.Tensor,
        cell_measures: Tuple[float, ...],
    ) -> Tuple[float, float]:
        """Compute relative mass balance residual and maximum trajectory mass deviation."""
        # Trajectory shape: [T+1, ...]
        if len(cell_measures) == 1:
            dx = cell_measures[0]
            m0 = float(torch.sum(init_state).item()) * dx
            mass_t = [float(torch.sum(step).item()) * dx for step in pred_trajectory]
        else:
            dx, dy = cell_measures[0], cell_measures[1]
            dA = dx * dy
            # If multi-channel, use channel 0 (depth or primary mass)
            chan0_init = init_state[0] if init_state.ndim >= 3 else init_state
            m0 = float(torch.sum(chan0_init).item()) * dA
            mass_t = []
            for step in pred_trajectory:
                s0 = step[0] if step.ndim >= 3 else step
                mass_t.append(float(torch.sum(s0).item()) * dA)

        deviations = [abs(m - m0) for m in mass_t]
        max_dev = max(deviations)
        denom = max(1.0, abs(m0))
        rel_residual = max_dev / denom
        return rel_residual, max_dev

    @classmethod
    def evaluate_rollout(
        cls,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
        cell_measures: Tuple[float, ...],
        is_stable: bool = True,
        completed_steps: int = 50,
    ) -> EvaluationMetrics:
        """Run full evaluation suite over trajectory."""
        T = min(pred_trajectory.shape[0], target_trajectory.shape[0])
        pred_valid = pred_trajectory[:T]
        target_valid = target_trajectory[:T]

        rel_l2, l1, linf = cls.compute_spatial_metrics(pred_valid, target_valid)
        spec_l1, phase_err = cls.compute_spectral_metrics(pred_valid[-1], target_valid[-1])
        rel_cons, max_dev = cls.compute_conservation_metrics(pred_valid, pred_valid[0], cell_measures)

        # Check negative depth in channel 0 if 2D shallow water
        neg_depth = 0
        if pred_valid.ndim >= 4:
            h_slice = pred_valid[:, 0]
            neg_depth = int(torch.sum(h_slice < 0.0).item())

        is_finite = bool(torch.all(torch.isfinite(pred_valid)).item())
        blowup = not (is_stable and is_finite)

        return EvaluationMetrics(
            relative_l2_error=rel_l2,
            l1_error=l1,
            linf_error=linf,
            spectral_l1_mismatch=spec_l1,
            phase_error_mean=phase_err,
            mass_conservation_residual=rel_cons,
            max_mass_deviation=max_dev,
            completed_horizon_steps=completed_steps,
            is_stable=not blowup,
            negative_depth_count=neg_depth,
            blowup_event=blowup,
        )
