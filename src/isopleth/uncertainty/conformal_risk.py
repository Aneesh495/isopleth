"""Distribution-free Conformal Risk Control (CRC) for physics rollouts.

Extends standard conformal prediction from sets to arbitrary monotonic loss functions,
guaranteeing that the expected physical risk (such as long-horizon conservation drift
or relative L2 divergence) is bounded below a user-prescribed risk level alpha:
    E[ L(Y, C_lambda(X)) ] <= alpha
with distribution-free finite-sample coverage guarantees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats as stats
import torch


@dataclass
class RiskControlConfig:
    """Configuration for Conformal Risk Control calibration."""

    target_risk_level: float = 0.05  # alpha
    delta: float = 0.05  # Confidence bound 1 - delta
    bound_type: str = "hoeffding"  # 'hoeffding' or 'empirical_bernstein'
    lambda_grid_size: int = 100


@dataclass
class RiskCalibrationReport:
    """Outcome report for Conformal Risk Control calibration."""

    calibrated_threshold_lambda: float
    empirical_risk_on_calibration: float
    empirical_risk_on_test: float
    target_risk_alpha: float
    risk_bound_satisfied: bool
    sample_size_calib: int
    sample_size_test: int
    bound_used: str


class ConformalRiskController:
    """Calibrates risk-controlling thresholds over neural operator trajectories."""

    def __init__(self, config: Optional[RiskControlConfig] = None) -> None:
        self.config = config or RiskControlConfig()

    def calibrate_threshold(
        self,
        calibration_losses: Union[np.ndarray, Sequence[float]],
        lambda_candidates: Sequence[float],
        is_monotonic_decreasing: bool = True,
    ) -> float:
        """Find the smallest (or largest) parameter lambda satisfying the statistical risk bound.

        Args:
            calibration_losses: Evaluation loss matrix of shape (N_calib, num_lambdas).
            lambda_candidates: Evaluated threshold parameter candidate values.
            is_monotonic_decreasing: True if loss decreases as lambda increases.

        Returns:
            The calibrated threshold parameter lambda_hat.
        """
        losses_mat = np.asarray(calibration_losses, dtype=np.float64)
        n_calib = losses_mat.shape[0]
        n_candidates = len(lambda_candidates)

        if losses_mat.shape[1] != n_candidates:
            raise ValueError(
                f"losses_mat second dimension {losses_mat.shape[1]} must match candidates length {n_candidates}"
            )

        # Compute empirical mean risk R_hat(lambda) for each candidate
        mean_risks = np.mean(losses_mat, axis=0)

        # Upper confidence bound via Hoeffding or Empirical Bernstein
        if self.config.bound_type == "hoeffding":
            # Hoeffding UCB penalty: sqrt( log(1 / delta) / (2 * n) )
            penalty = math.sqrt(math.log(1.0 / self.config.delta) / (2.0 * n_calib))
            ucb_risks = mean_risks + penalty
        else:
            # Empirical Bernstein UCB penalty
            variances = np.var(losses_mat, axis=0)
            term1 = np.sqrt(2.0 * variances * math.log(2.0 / self.config.delta) / n_calib)
            term2 = 7.0 * math.log(2.0 / self.config.delta) / (3.0 * (n_calib - 1.0))
            ucb_risks = mean_risks + term1 + term2

        # Filter candidates satisfying UCB(lambda) <= target_risk_alpha
        valid_indices = np.where(ucb_risks <= self.config.target_risk_level)[0]

        if len(valid_indices) == 0:
            # Fall back to most conservative candidate
            return float(lambda_candidates[-1] if is_monotonic_decreasing else lambda_candidates[0])

        if is_monotonic_decreasing:
            # Pick smallest lambda that satisfies the bound
            best_idx = int(valid_indices[0])
        else:
            # Pick largest lambda
            best_idx = int(valid_indices[-1])

        return float(lambda_candidates[best_idx])

    def evaluate_test_risk(
        self,
        test_losses: Union[np.ndarray, Sequence[float]],
        calibrated_lambda: float,
        lambda_candidates: Sequence[float],
        calibration_losses: Optional[np.ndarray] = None,
    ) -> RiskCalibrationReport:
        """Evaluate calibrated threshold on held-out test trajectories."""
        t_losses = np.asarray(test_losses, dtype=np.float64)
        c_list = list(lambda_candidates)
        # Find nearest index
        idx = int(np.argmin(np.abs(np.array(c_list) - calibrated_lambda)))

        test_risk = float(np.mean(t_losses[:, idx]))
        calib_risk = float(np.mean(calibration_losses[:, idx])) if calibration_losses is not None else 0.0

        satisfied = test_risk <= self.config.target_risk_level

        return RiskCalibrationReport(
            calibrated_threshold_lambda=calibrated_lambda,
            empirical_risk_on_calibration=calib_risk,
            empirical_risk_on_test=test_risk,
            target_risk_alpha=self.config.target_risk_level,
            risk_bound_satisfied=satisfied,
            sample_size_calib=len(calibration_losses) if calibration_losses is not None else 0,
            sample_size_test=len(t_losses),
            bound_used=self.config.bound_type,
        )


class ConservationRiskAuditor:
    """Constructs loss matrices for mass and momentum drift risk control."""

    @staticmethod
    def build_mass_drift_loss_matrix(
        max_drifts: Union[np.ndarray, Sequence[float]],
        threshold_candidates: Sequence[float],
    ) -> np.ndarray:
        """Construct loss matrix L_i(lambda) = I(max_drift_i > lambda).

        Risk is the probability of exceeding the drift tolerance lambda.
        """
        drifts = np.asarray(max_drifts, dtype=np.float64)[:, None]
        thresholds = np.asarray(threshold_candidates, dtype=np.float64)[None, :]
        loss_matrix = (drifts > thresholds).astype(np.float64)
        return loss_matrix

    @staticmethod
    def build_bounded_l2_loss_matrix(
        relative_errors: Union[np.ndarray, Sequence[float]],
        tolerance_candidates: Sequence[float],
    ) -> np.ndarray:
        """Construct soft loss matrix L_i(lambda) = max(0, error_i - lambda) / (1 + error_i)."""
        errors = np.asarray(relative_errors, dtype=np.float64)[:, None]
        tolerances = np.asarray(tolerance_candidates, dtype=np.float64)[None, :]
        excess = np.maximum(0.0, errors - tolerances)
        loss_matrix = excess / (1.0 + errors)
        return loss_matrix
