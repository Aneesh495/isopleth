"""Conformal trajectory calibration and coverage evaluation (I19).

Computes distribution-free conformal uncertainty intervals over long-horizon rollouts
using held-out calibration partitions and trajectory-level non-conformity scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from isopleth.uncertainty.ensembles import EnsemblePrediction, TrajectoryEnsemble

@dataclass
class CalibrationReport:
    target_coverage: float
    observed_coverage: float
    conformal_quantile_q: float
    average_band_width: float
    n_calibration_samples: int
    n_test_samples: int
    is_valid_exchangeable: bool

class ConformalTrajectoryCalibrator:
    """Computes conformal uncertainty intervals on held-out calibration trajectories."""

    def __init__(self, target_coverage: float = 0.90, eps: float = 1e-4) -> None:
        self.target_coverage = target_coverage
        self.eps = eps
        self.conformal_quantile: Optional[float] = None

    def fit_calibration_scores(
        self,
        ensemble_predictions: List[EnsemblePrediction],
        ground_truth_trajectories: List[torch.Tensor],
    ) -> float:
        """Compute non-conformity scores and conformal quantile over calibration partition.
        
        Score: s_i = max_{t, x} |y_true - y_mean| / (sqrt(var) + eps)
        """
        if len(ensemble_predictions) != len(ground_truth_trajectories):
            raise ValueError("Mismatched predictions and ground truth count")

        scores: List[float] = []
        for pred, truth in zip(ensemble_predictions, ground_truth_trajectories):
            mean = pred.mean_trajectory
            std = torch.sqrt(torch.clamp(pred.variance_trajectory, min=1e-8)) + self.eps

            # Standardized residual: [T+1, ...]
            residual = torch.abs(truth - mean) / std
            s_i = float(torch.max(residual).item())
            scores.append(s_i)

        scores_arr = np.sort(np.array(scores))
        N = len(scores_arr)
        # 1 - alpha conformal index: ceil((N + 1) * (1 - alpha)) / N
        alpha = 1.0 - self.target_coverage
        index = int(math.ceil((N + 1) * (1.0 - alpha))) - 1
        index = min(max(0, index), N - 1)
        self.conformal_quantile = float(scores_arr[index])
        return self.conformal_quantile

    def predict_intervals(
        self,
        pred: EnsemblePrediction,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate calibrated lower and upper trajectory prediction intervals."""
        if self.conformal_quantile is None:
            raise RuntimeError("Calibrator has not been fitted on calibration partition yet")

        mean = pred.mean_trajectory
        std = torch.sqrt(torch.clamp(pred.variance_trajectory, min=1e-8)) + self.eps
        half_width = self.conformal_quantile * std

        lower = mean - half_width
        upper = mean + half_width
        return lower, upper

    def evaluate_test_coverage(
        self,
        test_predictions: List[EnsemblePrediction],
        test_ground_truth: List[torch.Tensor],
    ) -> CalibrationReport:
        """Evaluate empirical coverage and average interval width on test partition."""
        if self.conformal_quantile is None:
            raise RuntimeError("Calibrator has not been fitted")

        covered_count = 0
        total_width = 0.0
        n_eval = len(test_predictions)

        for pred, truth in zip(test_predictions, test_ground_truth):
            lower, upper = self.predict_intervals(pred)
            # Trajectory is covered if lower <= truth <= upper everywhere
            is_covered = bool(torch.all((truth >= lower - 1e-6) & (truth <= upper + 1e-6)).item())
            if is_covered:
                covered_count += 1
            width = float(torch.mean(upper - lower).item())
            total_width += width

        emp_cov = covered_count / max(1, n_eval)
        avg_w = total_width / max(1, n_eval)

        return CalibrationReport(
            target_coverage=self.target_coverage,
            observed_coverage=emp_cov,
            conformal_quantile_q=self.conformal_quantile,
            average_band_width=avg_w,
            n_calibration_samples=len(test_predictions),
            n_test_samples=n_eval,
            is_valid_exchangeable=True,
        )
