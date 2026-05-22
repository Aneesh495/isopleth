"""Differentiable inverse reconstruction from sparse observations (I22).

Optimizes unknown initial states and physical parameters by backpropagating
through learned forward neural operators using PyTorch autograd.
Enforces physical box constraints and smoothness regularization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from isopleth.inverse.observations import ObservationRecord


@dataclass
class InverseConfig:
    """Hyperparameters for differentiable inverse reconstruction."""

    max_iterations: int = 150
    optimizer_type: str = "adam"  # "adam", "adamw", or "lbfgs"
    learning_rate: float = 0.05
    lbfgs_max_iter: int = 20
    lbfgs_history_size: int = 25
    weight_observation: float = 1.0
    weight_smoothness_h1: float = 0.01
    weight_total_variation: float = 0.001
    weight_mass_prior: float = 0.05
    tolerance_grad: float = 1e-6
    tolerance_change: float = 1e-7
    target_mass: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    num_restarts: int = 1
    seed: int = 42


@dataclass
class InverseReconstructionResult:
    """Outcomes and convergence diagnostics for inverse problem solving."""

    reconstructed_initial_state: torch.Tensor
    reconstructed_parameters: Optional[torch.Tensor]
    loss_history: List[float]
    obs_loss_history: List[float]
    reg_loss_history: List[float]
    grad_norm_history: List[float]
    converged: bool
    iterations_run: int
    wall_clock_seconds: float
    final_loss: float
    initial_loss: float
    best_loss: float


class DifferentiableInverseReconstructor:
    """Solves sparse inverse problems via autograd through forward surrogate operators."""

    def __init__(
        self,
        forward_model: nn.Module,
        config: Optional[InverseConfig] = None,
    ) -> None:
        self.model = forward_model
        self.config = config or InverseConfig()
        # Ensure model weights remain frozen during state optimization
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def _compute_smoothness_h1(self, field: torch.Tensor) -> torch.Tensor:
        """Computes Sobolev H1 semi-norm (squared spatial gradients)."""
        if field.ndim == 3:
            # 1D field: [B, C, N]
            diff = field[:, :, 1:] - field[:, :, :-1]
            return torch.mean(diff ** 2)
        elif field.ndim == 4:
            # 2D field: [B, C, Ny, Nx]
            diff_y = field[:, :, 1:, :] - field[:, :, :-1, :]
            diff_x = field[:, :, :, 1:] - field[:, :, :, :-1]
            return torch.mean(diff_y ** 2) + torch.mean(diff_x ** 2)
        return torch.tensor(0.0, device=field.device)

    def _compute_total_variation(self, field: torch.Tensor) -> torch.Tensor:
        """Computes anisotropic Total Variation norm."""
        if field.ndim == 3:
            return torch.mean(torch.abs(field[:, :, 1:] - field[:, :, :-1]))
        elif field.ndim == 4:
            tv_y = torch.mean(torch.abs(field[:, :, 1:, :] - field[:, :, :-1, :]))
            tv_x = torch.mean(torch.abs(field[:, :, :, 1:] - field[:, :, :, :-1]))
            return tv_y + tv_x
        return torch.tensor(0.0, device=field.device)

    def _compute_mass(self, field: torch.Tensor) -> torch.Tensor:
        """Computes spatial integral or total mass."""
        if field.ndim == 3:
            return torch.sum(field, dim=-1)
        elif field.ndim == 4:
            return torch.sum(field, dim=(-2, -1))
        return torch.sum(field)

    def _rollout_surrogate(
        self,
        initial_state: torch.Tensor,
        num_steps: int,
        params: Optional[torch.Tensor] = None,
        dt: float = 0.05,
    ) -> torch.Tensor:
        """Rolls out the forward surrogate over num_steps time increments."""
        # Initial state: [B, C, ... spatial dims ...]
        current = initial_state
        trajectory = [current]
        dt_tensor = torch.tensor([dt], device=initial_state.device)

        # Spatial grid spacing
        is_1d = (initial_state.ndim == 3)
        if is_1d:
            dx = 1.0 / float(initial_state.shape[-1])
        else:
            dx = (1.0 / float(initial_state.shape[-2]), 1.0 / float(initial_state.shape[-1]))

        for _ in range(num_steps):
            if params is not None:
                try:
                    step_pred = self.model(current, params, dt=dt_tensor)
                except TypeError:
                    step_pred = self.model(current, params, dt_tensor)
            else:
                try:
                    step_pred = self.model(current, dx=dx, dt=dt)
                except TypeError:
                    try:
                        step_pred = self.model(current, dt=dt_tensor)
                    except TypeError:
                        step_pred = self.model(current)
            if isinstance(step_pred, tuple):
                step_pred = step_pred[0]
            current = step_pred
            trajectory.append(current)

        return torch.stack(trajectory, dim=1)  # [B, T+1, C, ...]

    def reconstruct(
        self,
        observation: ObservationRecord,
        num_rollout_steps: int,
        initial_guess: Optional[torch.Tensor] = None,
        param_guess: Optional[torch.Tensor] = None,
        optimize_params: bool = False,
        dt: float = 0.05,
        target_shape: Optional[Tuple[int, ...]] = None,
    ) -> InverseReconstructionResult:
        """Executes differentiable optimization to reconstruct initial state and parameters."""
        best_result: Optional[InverseReconstructionResult] = None

        for restart_idx in range(self.config.num_restarts):
            run_result = self._execute_single_run(
                observation=observation,
                num_rollout_steps=num_rollout_steps,
                initial_guess=initial_guess,
                param_guess=param_guess,
                optimize_params=optimize_params,
                dt=dt,
                target_shape=target_shape,
                seed_offset=restart_idx * 1000,
            )
            if best_result is None or run_result.final_loss < best_result.final_loss:
                best_result = run_result

        assert best_result is not None
        return best_result

    def _execute_single_run(
        self,
        observation: ObservationRecord,
        num_rollout_steps: int,
        initial_guess: Optional[torch.Tensor],
        param_guess: Optional[torch.Tensor],
        optimize_params: bool,
        dt: float,
        target_shape: Optional[Tuple[int, ...]],
        seed_offset: int,
    ) -> InverseReconstructionResult:
        """Single optimization trajectory run."""
        start_time = time.perf_counter()
        torch.manual_seed(self.config.seed + seed_offset)

        # Initialize candidate state tensor
        if initial_guess is not None:
            state_opt = initial_guess.clone().detach()
        elif target_shape is not None:
            state_opt = torch.randn(target_shape, dtype=torch.float32) * 0.1
        else:
            raise ValueError("Must provide initial_guess or target_shape.")

        state_opt.requires_grad_(True)
        opt_params_list = [state_opt]

        params_tensor: Optional[torch.Tensor] = None
        if optimize_params:
            if param_guess is not None:
                params_tensor = param_guess.clone().detach()
            else:
                params_tensor = torch.zeros((1, 1), dtype=torch.float32)
            params_tensor.requires_grad_(True)
            opt_params_list.append(params_tensor)

        # Setup optimizer
        if self.config.optimizer_type.lower() == "adam":
            optimizer: torch.optim.Optimizer = torch.optim.Adam(
                opt_params_list, lr=self.config.learning_rate
            )
        elif self.config.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                opt_params_list, lr=self.config.learning_rate, weight_decay=1e-4
            )
        elif self.config.optimizer_type.lower() == "lbfgs":
            optimizer = torch.optim.LBFGS(
                opt_params_list,
                lr=self.config.learning_rate,
                max_iter=self.config.lbfgs_max_iter,
                history_size=self.config.lbfgs_history_size,
                line_search_fn="strong_wolfe",
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

        loss_history: List[float] = []
        obs_loss_history: List[float] = []
        reg_loss_history: List[float] = []
        grad_norm_history: List[float] = []

        is_1d = state_opt.ndim == 3
        prev_loss = float("inf")
        converged = False

        def objective_closure() -> torch.Tensor:
            nonlocal state_opt, params_tensor
            optimizer.zero_grad()

            # Rollout forward surrogate
            pred_traj = self._rollout_surrogate(
                initial_state=state_opt,
                num_steps=num_rollout_steps,
                params=params_tensor,
                dt=dt,
            )

            # Extract predicted measurements at sensor locations
            # pred_traj shape: [B, T+1, C, ...]
            # Observation mask shape matches trajectory or indexed
            if is_1d:
                # 1D sensors: sensor_indices is [n_sensors]
                # observed_values is [T_obs, n_sensors]
                sensor_idx = observation.sensor_indices
                stride = observation.temporal_subsample_stride
                time_indices = torch.arange(
                    0, num_rollout_steps + 1, stride, device=state_opt.device
                )
                pred_obs = pred_traj[0, time_indices, 0, :][:, sensor_idx]
                target_obs = observation.observed_values
                # Match lengths if trajectory lengths differ
                min_t = min(pred_obs.shape[0], target_obs.shape[0])
                obs_err = F.mse_loss(pred_obs[:min_t], target_obs[:min_t])
            else:
                # 2D sensors: sensor_indices is linear [n_sensors]
                ny, nx = state_opt.shape[-2], state_opt.shape[-1]
                stride = observation.temporal_subsample_stride
                time_indices = torch.arange(
                    0, num_rollout_steps + 1, stride, device=state_opt.device
                )
                pred_sub = pred_traj[0, time_indices]  # [T_sub, C, Ny, Nx]
                flat_spatial = pred_sub.view(pred_sub.shape[0], pred_sub.shape[1], -1)
                sensor_idx = observation.sensor_indices
                pred_obs = flat_spatial[:, :, sensor_idx]
                target_obs = observation.observed_values
                min_t = min(pred_obs.shape[0], target_obs.shape[0])
                obs_err = F.mse_loss(pred_obs[:min_t], target_obs[:min_t])

            # Regularization terms
            h1_norm = self._compute_smoothness_h1(state_opt)
            tv_norm = self._compute_total_variation(state_opt)

            reg_err = (
                self.config.weight_smoothness_h1 * h1_norm
                + self.config.weight_total_variation * tv_norm
            )

            if self.config.target_mass is not None:
                current_mass = self._compute_mass(state_opt)
                target_m = torch.tensor(
                    self.config.target_mass,
                    device=state_opt.device,
                    dtype=current_mass.dtype,
                )
                mass_err = torch.mean((current_mass - target_m) ** 2)
                reg_err = reg_err + self.config.weight_mass_prior * mass_err

            total_loss = self.config.weight_observation * obs_err + reg_err
            total_loss.backward()

            return total_loss

        initial_loss = 0.0
        best_loss = float("inf")

        # Optimization loop
        for iter_idx in range(self.config.max_iterations):
            if isinstance(optimizer, torch.optim.LBFGS):
                loss = optimizer.step(objective_closure)
            else:
                loss = objective_closure()
                optimizer.step()

            # Apply projection box constraints if specified
            with torch.no_grad():
                if self.config.min_value is not None:
                    state_opt.clamp_min_(self.config.min_value)
                if self.config.max_value is not None:
                    state_opt.clamp_max_(self.config.max_value)

            loss_val = float(loss.item())
            if iter_idx == 0:
                initial_loss = loss_val

            best_loss = min(best_loss, loss_val)
            loss_history.append(loss_val)

            # Grad norm diagnostic
            grad_norm = 0.0
            if state_opt.grad is not None:
                grad_norm = float(torch.norm(state_opt.grad).item())
            grad_norm_history.append(grad_norm)

            # Convergence checks
            if grad_norm < self.config.tolerance_grad:
                converged = True
                break
            if abs(prev_loss - loss_val) < self.config.tolerance_change:
                converged = True
                break
            prev_loss = loss_val

        wall_time = time.perf_counter() - start_time

        return InverseReconstructionResult(
            reconstructed_initial_state=state_opt.detach(),
            reconstructed_parameters=params_tensor.detach() if params_tensor is not None else None,
            loss_history=loss_history,
            obs_loss_history=obs_loss_history,
            reg_loss_history=reg_loss_history,
            grad_norm_history=grad_norm_history,
            converged=converged,
            iterations_run=len(loss_history),
            wall_clock_seconds=wall_time,
            final_loss=loss_history[-1] if loss_history else 0.0,
            initial_loss=initial_loss,
            best_loss=best_loss,
        )
