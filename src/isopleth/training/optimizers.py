"""Physics-informed optimization algorithms and projection operators.

Provides specialized optimizers and projection operators for conservation laws:
1. ProjectedGradientDescent: enforces physical positivity constraints (depth, species).
2. ConservationConstrainedOptimizer: orthogonal projection onto zero-divergence hyperplanes.
3. StochasticWeightAveragingPDE: parameter checkpoint averaging for flat loss basins.
4. SecondOrderLBFGSWrapper: robust L-BFGS optimizer tailored for inverse closures.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class ProjectedGradientDescent:
    """Projects state or parameter tensors onto convex constraint sets."""

    @staticmethod
    def project_positive_depth(
        state: torch.Tensor, h_min: float = 1e-6, depth_channel: int = 0
    ) -> torch.Tensor:
        """Enforces strict water depth positivity: h >= h_min."""
        projected = state.clone()
        projected[:, depth_channel : depth_channel + 1] = torch.clamp(
            projected[:, depth_channel : depth_channel + 1], min=h_min
        )
        return projected

    @staticmethod
    def project_species_bounds(
        state: torch.Tensor,
        u_min: float = 0.0,
        u_max: float = 1.0,
        v_min: float = 0.0,
        v_max: float = 1.0,
    ) -> torch.Tensor:
        """Enforces physical chemical concentration bounds [min, max] for Gray-Scott."""
        projected = state.clone()
        if projected.shape[1] >= 2:
            projected[:, 0:1] = torch.clamp(projected[:, 0:1], min=u_min, max=u_max)
            projected[:, 1:2] = torch.clamp(projected[:, 1:2], min=v_min, max=v_max)
        return projected

    @staticmethod
    def project_zero_divergence_1d(flux_updates: torch.Tensor) -> torch.Tensor:
        """Projects numerical fluxes so that domain integral of divergence vanishes exactly."""
        # Integral of divergence over periodic domain is F(1) - F(0) = 0
        projected = flux_updates.clone()
        mean_flux = torch.mean(projected, dim=-1, keepdim=True)
        return projected - mean_flux


class ConservationConstrainedOptimizer:
    """Wraps PyTorch optimizer with orthogonal zero-mass-drift projection step.

    For state optimization in inverse problems, ensures that gradient update delta_u
    satisfies sum(delta_u) = 0, so that prior mass is strictly preserved during
    gradient descent iterations:
        delta_u_proj = delta_u - (1 / N) * sum(delta_u)
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        target_tensor: torch.Tensor,
        preserve_mass: bool = True,
        enforce_positive: bool = False,
        min_value: float = 0.0,
    ) -> None:
        self.optimizer = base_optimizer
        self.target = target_tensor
        self.preserve_mass = preserve_mass
        self.enforce_positive = enforce_positive
        self.min_value = min_value

        # Record initial invariant mass
        with torch.no_grad():
            self.initial_mass = float(torch.sum(self.target).item())

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        """Perform optimization step followed by invariant projection."""
        loss = self.optimizer.step(closure)

        with torch.no_grad():
            if self.target.grad is not None and self.preserve_mass:
                # Orthogonal projection onto zero-sum hyperplane
                grad_mean = torch.mean(self.target.grad)
                self.target.grad.sub_(grad_mean)

            if self.enforce_positive:
                self.target.clamp_(min=self.min_value)

            if self.preserve_mass:
                current_mass = float(torch.sum(self.target).item())
                discrepancy = self.initial_mass - current_mass
                numel = self.target.numel()
                self.target.add_(discrepancy / float(numel))

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Reset parameter gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)


class StochasticWeightAveragingPDE:
    """Averages model weight checkpoints over training epochs for flatter PDE loss basins.

    Stochastic Weight Averaging (Izmailov et al., 2018) leads to broader local optima
    that generalize significantly better across out-of-distribution initial frequencies.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.requires_grad_(False)
        self.num_checkpoints = 0

    def update(self) -> None:
        """Incorporate current model weights into running uniform average."""
        self.num_checkpoints += 1
        alpha = 1.0 / float(self.num_checkpoints)

        with torch.no_grad():
            for p_avg, p_curr in zip(
                self.averaged_model.parameters(), self.model.parameters()
            ):
                p_avg.lerp_(p_curr, weight=alpha)

            for b_avg, b_curr in zip(
                self.averaged_model.buffers(), self.model.buffers()
            ):
                b_avg.copy_(b_curr)

    def apply_averaged_weights(self) -> None:
        """Copy averaged weights into active working model."""
        with torch.no_grad():
            for p_curr, p_avg in zip(
                self.model.parameters(), self.averaged_model.parameters()
            ):
                p_curr.copy_(p_avg)


class SecondOrderLBFGSWrapper:
    """Quasi-Newton L-BFGS optimizer tailored for inverse PDE state reconstruction."""

    def __init__(
        self,
        parameters: Sequence[torch.Tensor],
        learning_rate: float = 1.0,
        max_iter: int = 20,
        history_size: int = 10,
        line_search_fn: str = "strong_wolfe",
    ) -> None:
        self.optimizer = torch.optim.LBFGS(
            parameters,
            lr=learning_rate,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Execute L-BFGS second-order step using evaluated closure."""
        return self.optimizer.step(closure)
