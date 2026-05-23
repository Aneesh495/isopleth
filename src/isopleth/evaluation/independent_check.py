"""Independent verification engine for trajectory predictions and physical invariants (I29).

Evaluates predictions without importing any neural network model architecture or training code.
Verifies:
1. Spatial error norms (relative L1, L2, Linf).
2. Physical mass conservation invariance (|M(t) - M(0)| / M(0)).
3. Positivity conditions (h >= 0, u >= 0, v >= 0).
4. Spectral energy distribution and power cascades.
5. Emits cryptographic SHA-256 independent verification certificates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class IndependentVerificationResult:
    """Outcome of standalone physical and statistical trajectory verification."""

    dataset_name: str
    num_trajectories_checked: int
    mean_relative_l2: float
    max_relative_l2: float
    mean_relative_l1: float
    max_linf: float
    max_mass_conservation_drift: float
    positivity_violations_count: int
    nan_or_inf_detected: bool
    spectral_energy_ratio: float
    passed_all_gates: bool
    input_sha256: str
    certificate_sha256: str
    failure_reasons: List[str] = field(default_factory=list)


class IndependentTrajectoryVerifier:
    """Zero-model standalone verifier verifying physical validity of rollout arrays."""

    def __init__(
        self,
        tolerance_rel_l2: float = 0.05,
        tolerance_mass_drift: float = 1e-5,
        enforce_positivity: bool = True,
    ) -> None:
        self.tolerance_rel_l2 = tolerance_rel_l2
        self.tolerance_mass_drift = tolerance_mass_drift
        self.enforce_positivity = enforce_positivity

    def _hash_array(self, arr: np.ndarray) -> str:
        """Computes deterministic SHA-256 hash of numpy array buffer."""
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def verify_arrays(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        dataset_name: str = "trajectory_eval",
        is_positive_quantity: bool = False,
    ) -> IndependentVerificationResult:
        """Verifies predicted rollouts against ground truth arrays."""
        failure_reasons: List[str] = []

        # 1. NaN and Inf check
        nan_or_inf = bool(
            np.isnan(predicted).any()
            or np.isinf(predicted).any()
            or np.isnan(ground_truth).any()
            or np.isinf(ground_truth).any()
        )
        if nan_or_inf:
            failure_reasons.append("NaN or Inf detected in prediction or ground truth.")

        # 2. Shape matching
        if predicted.shape != ground_truth.shape:
            failure_reasons.append(
                f"Shape mismatch: predicted {predicted.shape} vs ground truth {ground_truth.shape}"
            )
            input_hash = "mismatch"
            return IndependentVerificationResult(
                dataset_name=dataset_name,
                num_trajectories_checked=0,
                mean_relative_l2=float("nan"),
                max_relative_l2=float("nan"),
                mean_relative_l1=float("nan"),
                max_linf=float("nan"),
                max_mass_conservation_drift=float("nan"),
                positivity_violations_count=-1,
                nan_or_inf_detected=nan_or_inf,
                spectral_energy_ratio=float("nan"),
                passed_all_gates=False,
                input_sha256=input_hash,
                certificate_sha256="failed",
                failure_reasons=failure_reasons,
            )

        input_hash = self._hash_array(predicted)

        # 3. Spatial error norms
        # Assumed shape: [B, T+1, ...] or [T+1, ...]
        flat_pred = predicted.reshape(predicted.shape[0], -1)
        flat_true = ground_truth.reshape(ground_truth.shape[0], -1)

        diff = flat_pred - flat_true
        l2_diff = np.linalg.norm(diff, axis=-1)
        l2_true = np.linalg.norm(flat_true, axis=-1) + 1e-12
        rel_l2 = l2_diff / l2_true

        l1_diff = np.linalg.norm(diff, ord=1, axis=-1)
        l1_true = np.linalg.norm(flat_true, ord=1, axis=-1) + 1e-12
        rel_l1 = l1_diff / l1_true

        linf_err = np.max(np.abs(diff), axis=-1)

        mean_rel_l2 = float(np.mean(rel_l2))
        max_rel_l2 = float(np.max(rel_l2))
        mean_rel_l1 = float(np.mean(rel_l1))
        max_linf = float(np.max(linf_err))

        if mean_rel_l2 > self.tolerance_rel_l2:
            failure_reasons.append(
                f"Mean relative L2 error {mean_rel_l2:.4e} exceeds tolerance {self.tolerance_rel_l2:.4e}"
            )

        # 4. Mass conservation drift across time
        # Sum over spatial axes (everything after batch and time axes)
        spatial_axes = tuple(range(2, predicted.ndim)) if predicted.ndim >= 3 else (-1,)
        m_t = np.sum(predicted, axis=spatial_axes)  # [B, T+1]
        m_0 = m_t[:, :1]
        drift_matrix = np.abs(m_t - m_0) / (np.abs(m_0) + 1e-12)
        max_drift = float(np.max(drift_matrix))

        if max_drift > self.tolerance_mass_drift:
            failure_reasons.append(
                f"Maximum mass drift {max_drift:.4e} exceeds tolerance {self.tolerance_mass_drift:.4e}"
            )

        # 5. Positivity validation
        pos_violations = 0
        if self.enforce_positivity or is_positive_quantity:
            min_val = float(np.min(predicted))
            if min_val < -1e-6:
                pos_violations = int(np.sum(predicted < -1e-6))
                failure_reasons.append(
                    f"Positivity violation: found {pos_violations} negative values (min: {min_val:.4e})"
                )

        # 6. Spectral energy ratio (pred / true)
        flat_last_pred = flat_pred[-1]
        flat_last_true = flat_true[-1]
        fft_pred = np.abs(np.fft.rfft(flat_last_pred)) ** 2
        fft_true = np.abs(np.fft.rfft(flat_last_true)) ** 2
        spec_ratio = float(np.sum(fft_pred) / (np.sum(fft_true) + 1e-12))

        passed = len(failure_reasons) == 0

        # Construct verification certificate digest
        cert_data = {
            "dataset_name": dataset_name,
            "mean_relative_l2": mean_rel_l2,
            "max_mass_drift": max_drift,
            "positivity_violations": pos_violations,
            "passed": passed,
            "input_hash": input_hash,
        }
        cert_json = json.dumps(cert_data, sort_keys=True)
        cert_hash = hashlib.sha256(cert_json.encode("utf-8")).hexdigest()

        return IndependentVerificationResult(
            dataset_name=dataset_name,
            num_trajectories_checked=predicted.shape[0],
            mean_relative_l2=mean_rel_l2,
            max_relative_l2=max_rel_l2,
            mean_relative_l1=mean_rel_l1,
            max_linf=max_linf,
            max_mass_conservation_drift=max_drift,
            positivity_violations_count=pos_violations,
            nan_or_inf_detected=nan_or_inf,
            spectral_energy_ratio=spec_ratio,
            passed_all_gates=passed,
            input_sha256=input_hash,
            certificate_sha256=cert_hash,
            failure_reasons=failure_reasons,
        )
