"""Multi-seed trajectory ensembles and epistemic variance estimation (I19)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from isopleth.data.contracts import BoundaryCondition
from isopleth.rollout.runner import AutoregressiveRolloutRunner, RolloutResult

@dataclass
class EnsemblePrediction:
    mean_trajectory: torch.Tensor       # [B, T+1, ...]
    variance_trajectory: torch.Tensor   # [B, T+1, ...]
    individual_trajectories: List[torch.Tensor]
    times: torch.Tensor
    is_stable: bool

class TrajectoryEnsemble:
    """Manages an ensemble of independently trained neural operator models."""

    def __init__(self, models: List[nn.Module], device: str | torch.device = "cpu") -> None:
        if len(models) < 2:
            raise ValueError(f"Ensemble requires at least 2 models, got {len(models)}")
        self.models = models
        self.device = torch.device(device)
        self.runners = [AutoregressiveRolloutRunner(m, device=self.device) for m in models]

    def predict_1d(
        self,
        x_init: torch.Tensor,
        dx: float,
        dt: float,
        steps: int,
        cond: Optional[torch.Tensor] = None,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> EnsemblePrediction:
        """Compute ensemble mean and epistemic variance across 1D rollout."""
        trajs = []
        is_all_stable = True
        times = None

        for runner in self.runners:
            res = runner.rollout_1d(x_init, dx=dx, dt=dt, steps=steps, cond=cond, boundary=boundary)
            trajs.append(res.trajectory)
            times = res.times
            if not res.is_stable:
                is_all_stable = False

        stacked = torch.stack(trajs, dim=0)  # [M, T+1, nx] or [M, B, T+1, nx]
        mean_traj = torch.mean(stacked, dim=0)
        var_traj = torch.var(stacked, dim=0, unbiased=True)

        return EnsemblePrediction(
            mean_trajectory=mean_traj,
            variance_trajectory=var_traj,
            individual_trajectories=trajs,
            times=times,
            is_stable=is_all_stable,
        )

    def predict_2d(
        self,
        x_init: torch.Tensor,
        dx: float,
        dy: float,
        dt: float,
        steps: int,
        cond: Optional[torch.Tensor] = None,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> EnsemblePrediction:
        """Compute ensemble mean and epistemic variance across 2D rollout."""
        trajs = []
        is_all_stable = True
        times = None

        for runner in self.runners:
            res = runner.rollout_2d(x_init, dx=dx, dy=dy, dt=dt, steps=steps, cond=cond, boundary=boundary)
            trajs.append(res.trajectory)
            times = res.times
            if not res.is_stable:
                is_all_stable = False

        stacked = torch.stack(trajs, dim=0)
        mean_traj = torch.mean(stacked, dim=0)
        var_traj = torch.var(stacked, dim=0, unbiased=True)

        return EnsemblePrediction(
            mean_trajectory=mean_traj,
            variance_trajectory=var_traj,
            individual_trajectories=trajs,
            times=times,
            is_stable=is_all_stable,
        )
