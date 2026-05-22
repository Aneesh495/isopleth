"""Rigorous gradient checks and surrogate distortion diagnostics (I24).

Verifies PyTorch autograd gradients using float64 central finite differences (FD),
and evaluates how learned forward neural surrogate errors distort adjoint gradients
relative to true numerical solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from isopleth.inverse.observations import ObservationRecord
from isopleth.numerics.burgers import Burgers1DSolver, BurgersSolverConfig


@dataclass
class FiniteDifferenceCheckResult:
    """Outcome of central finite-difference gradient verification."""

    autograd_grad: torch.Tensor
    fd_grad: torch.Tensor
    relative_l2_error: float
    max_absolute_error: float
    passed: bool
    epsilon: float
    precision_used: str


@dataclass
class SurrogateDistortionResult:
    """Quantitative evaluation of gradient distortion introduced by neural surrogates."""

    cosine_similarity: float
    magnitude_ratio: float
    relative_gradient_error: float
    spurious_high_freq_ratio: float
    true_gradient_norm: float
    surrogate_gradient_norm: float
    spectral_energy_true: np.ndarray
    spectral_energy_surrogate: np.ndarray


def _rollout_burgers(
    solver: Burgers1DSolver,
    u0_np: np.ndarray,
    num_steps: int,
    dt: float,
    dx: float,
) -> np.ndarray:
    """Helper to step Burgers numerical solver and return trajectory array [T+1, nx]."""
    u_curr = torch.tensor(u0_np, dtype=torch.float32)
    traj = [u_curr.clone().numpy()]
    for _ in range(num_steps):
        u_curr = solver.step(u_curr, dx=dx, dt=dt)
        traj.append(u_curr.clone().numpy())
    return np.array(traj)


class GradientVerificationSuite:
    """Checks mathematical correctness of backpropagation gradients."""

    @staticmethod
    def check_central_finite_difference(
        objective_fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        epsilon: float = 1e-6,
        tolerance: float = 1e-5,
    ) -> FiniteDifferenceCheckResult:
        """Compares autograd gradient with central finite difference in float64."""
        # Convert to float64 for high-precision finite differencing
        x_d = x.clone().detach().to(dtype=torch.float64).requires_grad_(True)

        loss = objective_fn(x_d)
        loss.backward()
        assert x_d.grad is not None, "Autograd produced no gradient."
        autograd_grad = x_d.grad.clone().detach()

        flat_x = x_d.detach().clone().reshape(-1)
        num_elements = flat_x.numel()
        fd_grad_flat = torch.zeros(num_elements, dtype=torch.float64, device=x.device)

        with torch.no_grad():
            for i in range(num_elements):
                flat_p = flat_x.clone()
                flat_p[i] += epsilon
                xp = flat_p.view_as(x_d)
                lp = objective_fn(xp)

                flat_m = flat_x.clone()
                flat_m[i] -= epsilon
                xm = flat_m.view_as(x_d)
                lm = objective_fn(xm)

                fd_grad_flat[i] = (lp - lm) / (2.0 * epsilon)

        fd_grad = fd_grad_flat.view_as(x_d)

        diff = autograd_grad - fd_grad
        norm_diff = torch.norm(diff)
        norm_sum = torch.norm(autograd_grad) + torch.norm(fd_grad) + 1e-12
        rel_err = float((norm_diff / norm_sum).item())
        max_abs = float(torch.max(torch.abs(diff)).item())

        passed = rel_err < tolerance

        return FiniteDifferenceCheckResult(
            autograd_grad=autograd_grad.to(dtype=x.dtype),
            fd_grad=fd_grad.to(dtype=x.dtype),
            relative_l2_error=rel_err,
            max_absolute_error=max_abs,
            passed=passed,
            epsilon=epsilon,
            precision_used="float64",
        )


class SurrogateGradientDistortionAnalyzer:
    """Quantifies distortion of inverse loss gradients when replacing numerical solvers with neural operators."""

    def __init__(
        self,
        surrogate_model: nn.Module,
        numerical_solver: Burgers1DSolver,
    ) -> None:
        self.surrogate = surrogate_model
        self.solver = numerical_solver
        self.surrogate.eval()

    def _compute_numerical_true_gradient(
        self,
        u0: np.ndarray,
        observation: ObservationRecord,
        num_steps: int,
        dt: float = 0.05,
        eps_fd: float = 1e-5,
    ) -> np.ndarray:
        """Evaluates true gradient dJ/du0 via finite-difference through numerical solver."""
        nx = len(u0)
        dx = 1.0 / float(nx)
        grad = np.zeros(nx, dtype=np.float64)

        sensor_idx = observation.sensor_indices.cpu().numpy()
        obs_stride = observation.temporal_subsample_stride
        targets = observation.observed_values.cpu().numpy()

        for i in range(nx):
            u_p = u0.copy()
            u_p[i] += eps_fd
            traj_p = _rollout_burgers(self.solver, u_p, num_steps=num_steps, dt=dt, dx=dx)
            pred_p = traj_p[::obs_stride, sensor_idx]
            min_tp = min(pred_p.shape[0], targets.shape[0])
            loss_p = np.mean((pred_p[:min_tp] - targets[:min_tp]) ** 2)

            u_m = u0.copy()
            u_m[i] -= eps_fd
            traj_m = _rollout_burgers(self.solver, u_m, num_steps=num_steps, dt=dt, dx=dx)
            pred_m = traj_m[::obs_stride, sensor_idx]
            min_tm = min(pred_m.shape[0], targets.shape[0])
            loss_m = np.mean((pred_m[:min_tm] - targets[:min_tm]) ** 2)

            grad[i] = (loss_p - loss_m) / (2.0 * eps_fd)

        return grad

    def _compute_surrogate_autograd_gradient(
        self,
        u0: torch.Tensor,
        observation: ObservationRecord,
        num_steps: int,
        dt: float = 0.05,
    ) -> torch.Tensor:
        """Evaluates surrogate gradient dJ/du0 via PyTorch autograd."""
        u0_opt = u0.clone().detach().requires_grad_(True)
        current = u0_opt
        trajectory = [current]
        dt_tensor = torch.tensor([dt], device=u0.device)

        dx = 1.0 / float(u0.shape[-1])
        for _ in range(num_steps):
            try:
                current = self.surrogate(current, dx=dx, dt=dt)
            except TypeError:
                try:
                    current = self.surrogate(current, dt=dt_tensor)
                except TypeError:
                    current = self.surrogate(current)
            if isinstance(current, tuple):
                current = current[0]
            trajectory.append(current)

        traj_stacked = torch.stack(trajectory, dim=1)  # [1, T+1, 1, nx]

        sensor_idx = observation.sensor_indices
        stride = observation.temporal_subsample_stride
        time_idx = torch.arange(0, num_steps + 1, stride, device=u0.device)

        pred_obs = traj_stacked[0, time_idx, 0, :][:, sensor_idx]
        targets = observation.observed_values
        min_t = min(pred_obs.shape[0], targets.shape[0])

        loss = F.mse_loss(pred_obs[:min_t], targets[:min_t])
        loss.backward()

        assert u0_opt.grad is not None
        return u0_opt.grad.clone().detach()

    def analyze_distortion(
        self,
        u0_initial: torch.Tensor,
        observation: ObservationRecord,
        num_steps: int = 5,
        dt: float = 0.05,
    ) -> SurrogateDistortionResult:
        """Analyzes alignment, magnitude, and spectral spurious modes between true and surrogate gradients."""
        nx = u0_initial.shape[-1]
        u0_np = u0_initial.detach().cpu().squeeze().numpy()

        # True numerical solver gradient
        grad_true_np = self._compute_numerical_true_gradient(
            u0=u0_np,
            observation=observation,
            num_steps=num_steps,
            dt=dt,
        )
        grad_true = torch.tensor(grad_true_np, dtype=torch.float32)

        # Surrogate autograd gradient
        grad_surr_tensor = self._compute_surrogate_autograd_gradient(
            u0=u0_initial,
            observation=observation,
            num_steps=num_steps,
            dt=dt,
        )
        grad_surr = grad_surr_tensor.detach().cpu().squeeze().to(dtype=torch.float32)

        # Cosine similarity
        norm_true = float(torch.norm(grad_true).item())
        norm_surr = float(torch.norm(grad_surr).item())
        dot_product = float(torch.dot(grad_true, grad_surr).item())

        denom = (norm_true * norm_surr) + 1e-12
        cosine_sim = dot_product / denom
        mag_ratio = norm_surr / (norm_true + 1e-12)

        rel_grad_err = float((torch.norm(grad_surr - grad_true) / (norm_true + 1e-12)).item())

        # Spectral energy distribution of gradients
        fft_true = np.fft.rfft(grad_true_np)
        fft_surr = np.fft.rfft(grad_surr.numpy())

        energy_true = np.abs(fft_true) ** 2
        energy_surr = np.abs(fft_surr) ** 2

        # Ratio of energy in top 25% highest frequencies
        n_freq = len(energy_surr)
        cutoff = int(0.75 * n_freq)
        high_freq_surr = float(np.sum(energy_surr[cutoff:]))
        total_freq_surr = float(np.sum(energy_surr) + 1e-12)
        spurious_high_freq = high_freq_surr / total_freq_surr

        return SurrogateDistortionResult(
            cosine_similarity=cosine_sim,
            magnitude_ratio=mag_ratio,
            relative_gradient_error=rel_grad_err,
            spurious_high_freq_ratio=spurious_high_freq,
            true_gradient_norm=norm_true,
            surrogate_gradient_norm=norm_surr,
            spectral_energy_true=energy_true,
            spectral_energy_surrogate=energy_surr,
        )
