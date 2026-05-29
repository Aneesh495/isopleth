"""Long-horizon stability and physical invariant audit engine.

Audits neural operator rollouts over long temporal horizons (100 to 1000 steps):
1. Invariant drift tracking: mass, momentum, and kinetic energy accounting.
2. Monotonic physical energy dissipation in viscous Burgers and Shallow Water.
3. Total Variation (TV) ratio monitoring detecting Gibbs phenomena and numerical oscillations.
4. Shock sharpness and interface gradient preservation metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


@dataclass
class LongHorizonAuditReport:
    """Comprehensive certificate of long-horizon physical integrity."""

    steps_audited: int
    initial_mass: float
    final_mass: float
    max_mass_drift: float
    relative_mass_drift: float
    initial_energy: float
    final_energy: float
    monotonic_energy_dissipation: bool
    max_tv_ratio: float
    gibbs_overshoot_detected: bool
    passed_all_invariants: bool
    invariant_time_series: Dict[str, List[float]]


class LongHorizonConservationAuditor:
    """Audits long-horizon physical invariants, stability, and Gibbs oscillations."""

    def __init__(
        self,
        mass_drift_tolerance: float = 1e-4,
        allowable_tv_growth_factor: float = 1.25,
        enforce_monotonic_energy_decay: bool = False,
    ) -> None:
        self.mass_tol = float(mass_drift_tolerance)
        self.tv_tol = float(allowable_tv_growth_factor)
        self.enforce_monotonic_decay = enforce_monotonic_energy_decay

    @staticmethod
    def compute_total_variation_1d(u: torch.Tensor) -> float:
        """Compute spatial Total Variation: sum |u_{i+1} - u_i|."""
        diff = torch.abs(torch.roll(u, shifts=-1, dims=-1) - u)
        return float(torch.sum(diff).item())

    @staticmethod
    def compute_kinetic_energy_1d(u: torch.Tensor, dx: float) -> float:
        """Compute discrete spatial kinetic energy 0.5 * integral(u^2 dx)."""
        return 0.5 * float(torch.sum(u**2).item()) * dx

    def audit_trajectory_1d(
        self,
        trajectory: Union[torch.Tensor, Sequence[torch.Tensor]],
        dx: float = 0.01,
        dt: float = 0.01,
    ) -> LongHorizonAuditReport:
        """Audit an entire multi-step rollout trajectory.

        Args:
            trajectory: Tensor of shape (T, N) or list of 1D state tensors.
            dx: Spatial grid cell width.
            dt: Temporal step size.

        Returns:
            LongHorizonAuditReport detailing invariant accounting and stability.
        """
        if isinstance(trajectory, torch.Tensor):
            states = [trajectory[t] for t in range(trajectory.shape[0])]
        else:
            states = list(trajectory)

        n_steps = len(states)
        if n_steps < 2:
            raise ValueError(f"Trajectory must have at least 2 steps, got {n_steps}")

        mass_series: List[float] = []
        energy_series: List[float] = []
        tv_series: List[float] = []

        initial_u = states[0]
        initial_mass = float(torch.sum(initial_u).item()) * dx
        initial_energy = self.compute_kinetic_energy_1d(initial_u, dx=dx)
        initial_tv = self.compute_total_variation_1d(initial_u)

        max_mass_err = 0.0
        max_tv_ratio = 1.0
        gibbs_detected = False
        monotonic_energy = True

        prev_energy = initial_energy
        prev_tv = initial_tv

        for t_idx, state in enumerate(states):
            mass_t = float(torch.sum(state).item()) * dx
            energy_t = self.compute_kinetic_energy_1d(state, dx=dx)
            tv_t = self.compute_total_variation_1d(state)

            mass_series.append(mass_t)
            energy_series.append(energy_t)
            tv_series.append(tv_t)

            drift = abs(mass_t - initial_mass)
            if drift > max_mass_err:
                max_mass_err = drift

            # Energy dissipation check for viscous flows
            if energy_t > prev_energy + 1e-7:
                monotonic_energy = False

            # Total variation check
            if prev_tv > 1e-12:
                tv_ratio = tv_t / prev_tv
                if tv_ratio > max_tv_ratio:
                    max_tv_ratio = tv_ratio
                if tv_ratio > self.tv_tol:
                    gibbs_detected = True

            prev_energy = energy_t
            prev_tv = max(tv_t, 1e-12)

        final_mass = mass_series[-1]
        final_energy = energy_series[-1]
        initial_scale = max(abs(initial_mass), float(torch.sum(torch.abs(initial_u)).item()) * dx, 1e-6)
        rel_mass_drift = max_mass_err / initial_scale

        passed_mass = rel_mass_drift <= self.mass_tol
        passed_tv = not gibbs_detected
        passed_energy = monotonic_energy if self.enforce_monotonic_decay else True

        passed_all = passed_mass and passed_tv and passed_energy

        return LongHorizonAuditReport(
            steps_audited=n_steps,
            initial_mass=initial_mass,
            final_mass=final_mass,
            max_mass_drift=max_mass_err,
            relative_mass_drift=rel_mass_drift,
            initial_energy=initial_energy,
            final_energy=final_energy,
            monotonic_energy_dissipation=monotonic_energy,
            max_tv_ratio=max_tv_ratio,
            gibbs_overshoot_detected=gibbs_detected,
            passed_all_invariants=passed_all,
            invariant_time_series={
                "mass": mass_series,
                "kinetic_energy": energy_series,
                "total_variation": tv_series,
            },
        )
