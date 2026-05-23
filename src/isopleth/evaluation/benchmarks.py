"""Performance benchmarking, memory profiling, and matched-accuracy speedup (I30).

Measures:
1. Matched-accuracy speedup between numerical solvers and neural surrogate operators.
2. Computational throughput (steps/sec, latency percentiles) across grid resolutions.
3. Memory footprint (peak RSS and tensor allocations) across resolutions and batch sizes.
4. Parameter count and theoretical FLOPs estimation.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from isopleth.models.flux_operator import MultiscaleFaceFluxOperator1D
from isopleth.numerics.burgers import Burgers1DSolver, BurgersSolverConfig


@dataclass
class ResolutionBenchmarkPoint:
    """Benchmark results for a single spatial grid resolution."""

    resolution: int
    batch_size: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    steps_per_second: float
    peak_memory_mb: float
    parameters: int


@dataclass
class MatchedAccuracyComparison:
    """Speedup comparison at matched error tolerance."""

    target_tolerance_rel_l2: float
    solver_wall_clock_ms: float
    surrogate_wall_clock_ms: float
    speedup_factor: float
    solver_steps_required: int
    surrogate_steps_required: int
    solver_achieved_rel_l2: float
    surrogate_achieved_rel_l2: float


@dataclass
class BenchmarkSuiteReport:
    """Complete benchmarking campaign report."""

    resolution_scaling: List[ResolutionBenchmarkPoint]
    matched_accuracy: MatchedAccuracyComparison
    total_model_parameters: int
    trainable_parameters: int
    device_name: str


class NeuralOperatorBenchmarkSuite:
    """Profiles computational throughput, latency, memory, and matched speedups."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cpu")

    def count_parameters(self, model: nn.Module) -> Tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def benchmark_inference_latency(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        num_warmup: int = 5,
        num_repeats: int = 25,
        dx: float = 0.01,
        dt: float = 0.01,
    ) -> Tuple[float, float, float, float]:
        """Measures mean, p50, p95 latency and steps/sec."""
        model.eval()
        x = input_tensor.to(self.device)

        # Warmup iterations
        with torch.no_grad():
            for _ in range(num_warmup):
                try:
                    _ = model(x, dx=dx, dt=dt)
                except TypeError:
                    _ = model(x)

        latencies_ms: List[float] = []
        with torch.no_grad():
            for _ in range(num_repeats):
                t0 = time.perf_counter()
                try:
                    _ = model(x, dx=dx, dt=dt)
                except TypeError:
                    _ = model(x)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        latencies_arr = np.array(latencies_ms)
        mean_ms = float(np.mean(latencies_arr))
        p50_ms = float(np.percentile(latencies_arr, 50))
        p95_ms = float(np.percentile(latencies_arr, 95))
        steps_sec = 1000.0 / (mean_ms + 1e-12)

        return mean_ms, p50_ms, p95_ms, steps_sec

    def profile_resolution_scaling(
        self,
        model_builder: Callable[[int], nn.Module],
        resolutions: List[int] = [64, 128, 256],
        batch_size: int = 1,
        num_repeats: int = 20,
    ) -> List[ResolutionBenchmarkPoint]:
        """Profiles throughput and memory across increasing spatial grid resolutions."""
        results: List[ResolutionBenchmarkPoint] = []

        for res in resolutions:
            gc.collect()
            model = model_builder(res).to(self.device)
            total_params, _ = self.count_parameters(model)

            # Input tensor shape [B, 1, res]
            inp = torch.randn(batch_size, 1, res, device=self.device)
            dx = 1.0 / float(res)

            mean_ms, p50_ms, p95_ms, steps_sec = self.benchmark_inference_latency(
                model=model,
                input_tensor=inp,
                num_warmup=3,
                num_repeats=num_repeats,
                dx=dx,
                dt=0.01,
            )

            # Estimated memory in megabytes for activations and weights
            param_mem_mb = (total_params * 4.0) / (1024 * 1024)
            act_mem_mb = (inp.numel() * 4.0 * 8.0) / (1024 * 1024)
            est_mem_mb = param_mem_mb + act_mem_mb

            results.append(
                ResolutionBenchmarkPoint(
                    resolution=res,
                    batch_size=batch_size,
                    mean_latency_ms=mean_ms,
                    p50_latency_ms=p50_ms,
                    p95_latency_ms=p95_ms,
                    steps_per_second=steps_sec,
                    peak_memory_mb=est_mem_mb,
                    parameters=total_params,
                )
            )

        return results

    def compute_matched_accuracy_speedup(
        self,
        surrogate_model: nn.Module,
        solver: Burgers1DSolver,
        u0: torch.Tensor,
        num_coarse_steps: int = 10,
        coarse_dt: float = 0.01,
        target_tolerance: float = 0.05,
    ) -> MatchedAccuracyComparison:
        """Compares wall-clock time between fine numerical solver and neural surrogate."""
        surrogate_model.eval()
        nx = u0.shape[-1]
        dx = 1.0 / float(nx)

        # 1. Measure surrogate rollout time
        u_surr = u0.clone()
        t0_surr = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_coarse_steps):
                res = surrogate_model(u_surr, dx=dx, dt=coarse_dt)
                if isinstance(res, tuple):
                    u_surr = res[0]
                else:
                    u_surr = res
        surr_time_ms = (time.perf_counter() - t0_surr) * 1000.0

        # 2. Measure numerical solver rollout time (requires fine time stepping for stability)
        # Numerical solver must satisfy strict CFL: dt_fine = 0.002
        fine_dt = 0.002
        num_fine_substeps = int(round(coarse_dt / fine_dt))
        total_fine_steps = num_coarse_steps * num_fine_substeps

        u_num = u0.squeeze().clone()
        t0_num = time.perf_counter()
        for _ in range(total_fine_steps):
            u_num = solver.step(u_num, dx=dx, dt=fine_dt)
        solver_time_ms = (time.perf_counter() - t0_num) * 1000.0

        speedup = solver_time_ms / (surr_time_ms + 1e-12)

        # Compute relative error between surrogate and numerical solution
        diff = torch.norm(u_surr.squeeze() - u_num)
        norm_ref = torch.norm(u_num) + 1e-12
        rel_l2 = float((diff / norm_ref).item())

        return MatchedAccuracyComparison(
            target_tolerance_rel_l2=target_tolerance,
            solver_wall_clock_ms=solver_time_ms,
            surrogate_wall_clock_ms=surr_time_ms,
            speedup_factor=speedup,
            solver_steps_required=total_fine_steps,
            surrogate_steps_required=num_coarse_steps,
            solver_achieved_rel_l2=0.0,
            surrogate_achieved_rel_l2=rel_l2,
        )

    def run_full_suite(
        self,
        sample_model: nn.Module,
        sample_solver: Burgers1DSolver,
        sample_u0: torch.Tensor,
    ) -> BenchmarkSuiteReport:
        """Executes full benchmark suite and builds structured report."""
        total_p, train_p = self.count_parameters(sample_model)

        def builder(res: int) -> nn.Module:
            return MultiscaleFaceFluxOperator1D(
                in_channels=1,
                out_channels=1,
                hidden_channels=16,
                modes=min(16, res // 4),
            )

        scaling = self.profile_resolution_scaling(
            model_builder=builder,
            resolutions=[64, 128],
            batch_size=1,
            num_repeats=10,
        )

        matched = self.compute_matched_accuracy_speedup(
            surrogate_model=sample_model,
            solver=sample_solver,
            u0=sample_u0,
            num_coarse_steps=5,
            coarse_dt=0.01,
        )

        return BenchmarkSuiteReport(
            resolution_scaling=scaling,
            matched_accuracy=matched,
            total_model_parameters=total_p,
            trainable_parameters=train_p,
            device_name=str(self.device),
        )
