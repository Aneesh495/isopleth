"""Multi-horizon physical-unit loss functions for neural operators.

Combines normalized MSE, Sobolev H^1 gradient mismatch, Fourier spectral loss,
and physical mass conservation penalties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class LossComponents:
    """Record of individual loss term contributions."""
    total_loss: torch.Tensor
    mse_loss: torch.Tensor
    h1_loss: torch.Tensor
    spectral_loss: torch.Tensor
    conservation_penalty: torch.Tensor

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": float(self.total_loss.item()),
            "mse": float(self.mse_loss.item()),
            "h1": float(self.h1_loss.item()),
            "spectral": float(self.spectral_loss.item()),
            "conservation": float(self.conservation_penalty.item()),
        }

class PhysicalLoss(nn.Module):
    """Composite physical loss function."""

    def __init__(
        self,
        weight_mse: float = 1.0,
        weight_h1: float = 0.1,
        weight_spectral: float = 0.05,
        weight_conservation: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.w_mse = weight_mse
        self.w_h1 = weight_h1
        self.w_spectral = weight_spectral
        self.w_cons = weight_conservation
        self.eps = eps

    def compute_h1_loss_1d(self, pred: torch.Tensor, target: torch.Tensor, dx: float) -> torch.Tensor:
        """Compute discrete Sobolev H^1 gradient semi-norm: ||grad(pred - target)||^2."""
        d_pred = (torch.roll(pred, shifts=-1, dims=-1) - pred) / dx
        d_target = (torch.roll(target, shifts=-1, dims=-1) - target) / dx
        return torch.mean((d_pred - d_target)**2)

    def compute_h1_loss_2d(self, pred: torch.Tensor, target: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
        """Compute 2D discrete Sobolev H^1 gradient semi-norm."""
        gx_pred = (torch.roll(pred, shifts=-1, dims=-1) - pred) / dx
        gx_target = (torch.roll(target, shifts=-1, dims=-1) - target) / dx
        gy_pred = (torch.roll(pred, shifts=-1, dims=-2) - pred) / dy
        gy_target = (torch.roll(target, shifts=-1, dims=-2) - target) / dy
        return torch.mean((gx_pred - gx_target)**2) + torch.mean((gy_pred - gy_target)**2)

    def compute_spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 mismatch between Fourier power spectra."""
        if pred.ndim == 3 or (pred.ndim == 4 and pred.shape[1] == 1):  # 1D
            fft_pred = torch.abs(torch.fft.rfft(pred, dim=-1))
            fft_target = torch.abs(torch.fft.rfft(target, dim=-1))
            return torch.mean(torch.abs(fft_pred - fft_target))
        else:  # 2D
            fft_pred = torch.abs(torch.fft.rfft2(pred, dim=(-2, -1)))
            fft_target = torch.abs(torch.fft.rfft2(target, dim=(-2, -1)))
            return torch.mean(torch.abs(fft_pred - fft_target))

    def compute_conservation_penalty(
        self,
        pred: torch.Tensor,
        init: torch.Tensor,
        cell_measures: Tuple[float, ...],
        source_integrated: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute penalty on violation of integral mass balance: |Mass(pred) - Mass(init) - Source|^2."""
        if len(cell_measures) == 1:
            dx = cell_measures[0]
            mass_pred = torch.sum(pred, dim=-1) * dx
            mass_init = torch.sum(init, dim=-1) * dx
        else:
            dx, dy = cell_measures[0], cell_measures[1]
            mass_pred = torch.sum(pred, dim=(-2, -1)) * (dx * dy)
            mass_init = torch.sum(init, dim=(-2, -1)) * (dx * dy)

        expected_mass = mass_init
        if source_integrated is not None:
            expected_mass = expected_mass + source_integrated

        rel_violation = (mass_pred - expected_mass) / (torch.abs(mass_init) + 1.0)
        return torch.mean(rel_violation**2)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        init: torch.Tensor,
        cell_measures: Tuple[float, ...],
        source_integrated: Optional[torch.Tensor] = None,
    ) -> LossComponents:
        """Compute weighted composite loss."""
        # Relative Normalized MSE
        mse_denom = torch.mean(target**2) + self.eps
        mse = torch.mean((pred - target)**2) / mse_denom

        # Sobolev H^1
        if len(cell_measures) == 1:
            h1 = self.compute_h1_loss_1d(pred, target, dx=cell_measures[0])
        else:
            h1 = self.compute_h1_loss_2d(pred, target, dx=cell_measures[0], dy=cell_measures[1])

        # Spectral loss
        spec = self.compute_spectral_loss(pred, target)

        # Conservation penalty
        cons = self.compute_conservation_penalty(pred, init, cell_measures, source_integrated)

        total = self.w_mse * mse + self.w_h1 * h1 + self.w_spectral * spec + self.w_cons * cons

        return LossComponents(
            total_loss=total,
            mse_loss=mse,
            h1_loss=h1,
            spectral_loss=spec,
            conservation_penalty=cons,
        )
