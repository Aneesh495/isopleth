"""Standard and numerical solver baselines for inverse reconstruction (I23).

Includes:
1. Prior mean baseline (constant/uninformative prior).
2. Spatial interpolation baseline (linear and cubic spline interpolation of sparse initial sensors).
3. Numerical solver-based adjoint/finite-difference sensitivity reconstruction.
4. Comparative benchmark suite computing spatial recovery error and forecast error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.interpolate
import torch

from isopleth.inverse.observations import ObservationRecord
from isopleth.numerics.burgers import Burgers1DSolver, BurgersSolverConfig


@dataclass
class BaselineComparisonMetrics:
    """Quantitative performance metrics across reconstruction methods."""

    method_name: str
    relative_l2_initial_error: float
    relative_l1_initial_error: float
    linf_initial_error: float
    forecast_relative_l2_error: float
    wall_clock_seconds: float
    iterations: int
    converged: bool


class PriorMeanBaseline:
    """Estimates initial state using the empirical spatial mean of sparse observations."""

    def __init__(self, default_value: float = 0.0) -> None:
        self.default_value = default_value

    def reconstruct_1d(
        self,
        observation: ObservationRecord,
        nx: int,
    ) -> torch.Tensor:
        """Reconstructs 1D initial state [1, 1, nx] as constant mean."""
        # First observation slice (t=0)
        t0_obs = observation.observed_values[0]
        mean_val = float(torch.mean(t0_obs).item())
        reconstructed = torch.full((1, 1, nx), fill_value=mean_val, dtype=torch.float32)
        return reconstructed

    def reconstruct_2d(
        self,
        observation: ObservationRecord,
        shape: Tuple[int, int],
        channels: int = 3,
    ) -> torch.Tensor:
        """Reconstructs 2D initial state [1, C, ny, nx] as constant mean."""
        ny, nx = shape
        t0_obs = observation.observed_values[0]  # [C, n_sensors]
        mean_vals = torch.mean(t0_obs, dim=-1, keepdim=True)  # [C, 1]
        reconstructed = mean_vals.view(1, channels, 1, 1).expand(1, channels, ny, nx).clone()
        return reconstructed


class SpatialInterpolationBaseline:
    """Interpolates sparse point sensor observations to full spatial domain."""

    def __init__(self, method: str = "linear") -> None:
        self.method = method  # "linear", "cubic", or "nearest"

    def reconstruct_1d(
        self,
        observation: ObservationRecord,
        nx: int,
    ) -> torch.Tensor:
        """Interpolates 1D spatial measurements using piecewise linear or cubic splines."""
        sensor_idx = observation.sensor_indices.cpu().numpy()
        t0_obs = observation.observed_values[0].cpu().numpy()

        # Coordinates on [0, 1]
        x_sensors = sensor_idx / float(nx)
        x_full = np.linspace(0.0, 1.0, nx, endpoint=False)

        # For periodic domain, wrap boundary sensor points
        x_aug = np.concatenate([x_sensors - 1.0, x_sensors, x_sensors + 1.0])
        y_aug = np.concatenate([t0_obs, t0_obs, t0_obs])

        if self.method == "cubic" and len(x_sensors) >= 4:
            spline = scipy.interpolate.CubicSpline(x_aug, y_aug, bc_type="periodic")
            y_interp = spline(x_full)
        elif self.method == "nearest":
            interp_func = scipy.interpolate.interp1d(
                x_aug, y_aug, kind="nearest", bounds_error=False, fill_value="extrapolate"
            )
            y_interp = interp_func(x_full)
        else:
            interp_func = scipy.interpolate.interp1d(
                x_aug, y_aug, kind="linear", bounds_error=False, fill_value="extrapolate"
            )
            y_interp = interp_func(x_full)

        out_tensor = torch.tensor(y_interp, dtype=torch.float32).view(1, 1, nx)
        return out_tensor

    def reconstruct_2d(
        self,
        observation: ObservationRecord,
        shape: Tuple[int, int],
        channels: int = 3,
    ) -> torch.Tensor:
        """Interpolates 2D sparse point sensors to regular grid."""
        ny, nx = shape
        linear_indices = observation.sensor_indices.cpu().numpy()
        t0_obs = observation.observed_values[0].cpu().numpy()  # [C, n_sensors]

        sensor_y = (linear_indices // nx) / float(ny)
        sensor_x = (linear_indices % nx) / float(nx)
        points = np.stack([sensor_y, sensor_x], axis=-1)

        grid_y, grid_x = np.mgrid[0:1:complex(0, ny), 0:1:complex(0, nx)]
        reconstructed_channels = []

        interp_method = "nearest" if self.method == "nearest" else "linear"

        for c in range(channels):
            values = t0_obs[c]
            grid_vals = scipy.interpolate.griddata(
                points, values, (grid_y, grid_x), method=interp_method
            )
            # Fill NaNs with nearest neighbor
            if np.isnan(grid_vals).any():
                grid_nearest = scipy.interpolate.griddata(
                    points, values, (grid_y, grid_x), method="nearest"
                )
                grid_vals = np.where(np.isnan(grid_vals), grid_nearest, grid_vals)
            reconstructed_channels.append(grid_vals)

        arr = np.stack(reconstructed_channels, axis=0)  # [C, ny, nx]
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)


def _rollout_burgers_solver(
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


class NumericalSolverInverseBaseline:
    """Inverse reconstruction directly utilizing the high-fidelity numerical PDE solver.

    Uses central finite-difference sensitivities or direct derivative-free optimization
    to reconstruct the initial state against the actual physical equations.
    """

    def __init__(
        self,
        solver_config: BurgersSolverConfig,
        nx: int = 64,
        max_iterations: int = 50,
        learning_rate: float = 0.05,
    ) -> None:
        self.solver_config = solver_config
        self.nx = nx
        self.solver = Burgers1DSolver(solver_config)
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def reconstruct_burgers(
        self,
        observation: ObservationRecord,
        num_steps: int,
        initial_guess: Optional[torch.Tensor] = None,
        dt: float = 0.02,
    ) -> Tuple[torch.Tensor, BaselineComparisonMetrics]:
        """Reconstructs 1D Burgers initial state via numerical solver finite-difference gradients."""
        start_time = time.perf_counter()
        nx = self.nx
        dx = 1.0 / float(nx)

        # Initial guess from linear interpolation if not provided
        if initial_guess is not None:
            u0 = initial_guess.clone().detach().squeeze().numpy()
        else:
            interpolator = SpatialInterpolationBaseline(method="linear")
            u0_torch = interpolator.reconstruct_1d(observation, nx=nx)
            u0 = u0_torch.squeeze().numpy()

        sensor_idx = observation.sensor_indices.cpu().numpy()
        obs_stride = observation.temporal_subsample_stride
        obs_targets = observation.observed_values.cpu().numpy()  # [T_obs, n_sensors]

        best_u0 = u0.copy()
        best_loss = float("inf")

        eps_fd = 1e-4

        for it in range(self.max_iterations):
            # Compute current trajectory
            u_traj = _rollout_burgers_solver(self.solver, u0, num_steps=num_steps, dt=dt, dx=dx)
            pred_obs = u_traj[::obs_stride, sensor_idx]
            min_t = min(pred_obs.shape[0], obs_targets.shape[0])
            loss = float(np.mean((pred_obs[:min_t] - obs_targets[:min_t]) ** 2))

            if loss < best_loss:
                best_loss = loss
                best_u0 = u0.copy()

            if loss < 1e-6:
                break

            # Compute gradient with respect to u0 via finite difference approximation on sampled modes
            grad = np.zeros_like(u0)
            step_stride = max(1, nx // 32)
            for i in range(0, nx, step_stride):
                u_pert_p = u0.copy()
                u_pert_p[i] += eps_fd
                traj_p = _rollout_burgers_solver(self.solver, u_pert_p, num_steps=num_steps, dt=dt, dx=dx)
                loss_p = float(np.mean((traj_p[::obs_stride, sensor_idx][:min_t] - obs_targets[:min_t]) ** 2))

                u_pert_m = u0.copy()
                u_pert_m[i] -= eps_fd
                traj_m = _rollout_burgers_solver(self.solver, u_pert_m, num_steps=num_steps, dt=dt, dx=dx)
                loss_m = float(np.mean((traj_m[::obs_stride, sensor_idx][:min_t] - obs_targets[:min_t]) ** 2))

                grad[i] = (loss_p - loss_m) / (2.0 * eps_fd)

            # Interpolate gradient to remaining cells if sub-sampled
            if step_stride > 1:
                indices = np.arange(0, nx, step_stride)
                interp = scipy.interpolate.interp1d(
                    indices, grad[indices], kind="linear", fill_value="extrapolate"
                )
                grad = interp(np.arange(nx))

            # Gradient descent step
            u0 = u0 - self.learning_rate * grad

        wall_time = time.perf_counter() - start_time
        res_tensor = torch.tensor(best_u0, dtype=torch.float32).view(1, 1, nx)

        metrics = BaselineComparisonMetrics(
            method_name="NumericalSolverAdjointFD",
            relative_l2_initial_error=0.0,  # Computed when true initial state is known
            relative_l1_initial_error=0.0,
            linf_initial_error=0.0,
            forecast_relative_l2_error=0.0,
            wall_clock_seconds=wall_time,
            iterations=self.max_iterations,
            converged=best_loss < 1e-4,
        )

        return res_tensor, metrics


def evaluate_reconstruction_accuracy(
    reconstructed_initial: torch.Tensor,
    true_initial: torch.Tensor,
    reconstructed_rollout: Optional[torch.Tensor] = None,
    true_rollout: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Computes rigorous spatial and rollout relative errors between reconstruction and ground truth."""
    rec_flat = reconstructed_initial.reshape(-1)
    true_flat = true_initial.reshape(-1)

    diff = rec_flat - true_flat
    true_norm2 = torch.norm(true_flat, p=2) + 1e-12
    true_norm1 = torch.norm(true_flat, p=1) + 1e-12

    rel_l2 = float((torch.norm(diff, p=2) / true_norm2).item())
    rel_l1 = float((torch.norm(diff, p=1) / true_norm1).item())
    linf = float(torch.max(torch.abs(diff)).item())

    out: Dict[str, float] = {
        "rel_l2_initial": rel_l2,
        "rel_l1_initial": rel_l1,
        "linf_initial": linf,
    }

    if reconstructed_rollout is not None and true_rollout is not None:
        rec_traj_flat = reconstructed_rollout.reshape(-1)
        true_traj_flat = true_rollout.reshape(-1)
        traj_diff = rec_traj_flat - true_traj_flat
        traj_l2 = float((torch.norm(traj_diff, p=2) / (torch.norm(true_traj_flat, p=2) + 1e-12)).item())
        out["rel_l2_forecast"] = traj_l2

    return out
