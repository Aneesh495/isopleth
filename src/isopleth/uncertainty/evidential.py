"""Evidential Deep Learning for physical operator uncertainty quantification.

Places Normal-Inverse-Gamma (NIG) prior distributions over state predictions,
providing simultaneous analytical decomposition into aleatoric and epistemic
uncertainties without costly Monte Carlo sampling:
- Aleatoric uncertainty: variance of observation noise sigma^2 = beta / (alpha - 1).
- Epistemic uncertainty: model parameter uncertainty Var[mu] = beta / (v * (alpha - 1)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EvidentialPrediction:
    """Outcomes of evidential forward pass with decomposed uncertainties."""

    mean: torch.Tensor
    aleatoric_variance: torch.Tensor
    epistemic_variance: torch.Tensor
    total_variance: torch.Tensor
    v: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor


class EvidentialHead1D(nn.Module):
    """Predicts Normal-Inverse-Gamma parameters for 1D fields."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Output 4 channels per field: mu, v, alpha, beta
        self.conv = nn.Conv1d(in_channels, 4 * out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Output NIG parameters.

        Args:
            x: Latent features of shape (B, C_in, N).

        Returns:
            Tuple of (mu, v, alpha, beta) each of shape (B, C_out, N).
        """
        raw = self.conv(x)
        mu, log_v, log_alpha, log_beta = torch.chunk(raw, chunks=4, dim=1)

        # Enforce positive parameter support
        v = F.softplus(log_v) + 1e-4
        alpha = F.softplus(log_alpha) + 1.0 + 1e-4  # alpha > 1 for finite mean
        beta = F.softplus(log_beta) + 1e-4

        return mu, v, alpha, beta


class EvidentialHead2D(nn.Module):
    """Predicts Normal-Inverse-Gamma parameters for 2D fields."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels, 4 * out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Output NIG parameters for 2D field."""
        raw = self.conv(x)
        mu, log_v, log_alpha, log_beta = torch.chunk(raw, chunks=4, dim=1)

        v = F.softplus(log_v) + 1e-4
        alpha = F.softplus(log_alpha) + 1.0 + 1e-4
        beta = F.softplus(log_beta) + 1e-4

        return mu, v, alpha, beta


class EvidentialLoss(nn.Module):
    """Negative log-likelihood of Student-t marginal plus evidence regularizer.

    Penalizes both prediction error and confident predictions with high error.
    """

    def __init__(self, reg_coeff: float = 1e-2) -> None:
        super().__init__()
        self.reg_coeff = float(reg_coeff)

    def forward(
        self,
        mu: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute evidential loss.

        Args:
            mu, v, alpha, beta: Predicted NIG parameters.
            y_true: Ground truth target field.

        Returns:
            Scalar training loss.
        """
        error = y_true - mu
        omega = 2.0 * beta * (1.0 + v)

        # Student-t log likelihood term
        nll = (
            0.5 * torch.log(math.pi / v)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log(v * (error**2) + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        # Evidence regularizer: penalize high evidence on large residuals
        total_evidence = 2.0 * v + alpha
        reg = torch.abs(error) * total_evidence

        loss = torch.mean(nll + self.reg_coeff * reg)
        return loss


class EvidentialUncertaintyEstimator:
    """Computes aleatoric, epistemic, and total prediction variances."""

    @staticmethod
    def estimate_uncertainty(
        mu: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> EvidentialPrediction:
        """Decompose predictive variance into aleatoric and epistemic components."""
        # Aleatoric variance: observation noise
        aleatoric = beta / (alpha - 1.0)

        # Epistemic variance: model parameter variance
        epistemic = beta / (v * (alpha - 1.0))

        total = aleatoric + epistemic

        return EvidentialPrediction(
            mean=mu,
            aleatoric_variance=aleatoric,
            epistemic_variance=epistemic,
            total_variance=total,
            v=v,
            alpha=alpha,
            beta=beta,
        )
