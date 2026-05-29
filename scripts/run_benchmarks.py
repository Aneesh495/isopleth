#!/usr/bin/env python3
"""Run performance benchmarks and model comparison profiling.

Measures:
1. Model parameter counts and forward step latencies across operator architectures.
2. Resolution scaling throughput (steps/sec) across 32, 64, and 128 cells.
3. Matched-accuracy speedup comparing MFFNO against numerical solvers.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import torch

from isopleth.evaluation.benchmarks import NeuralOperatorBenchmarkSuite
from isopleth.models.deeponet import DeepONet1D
from isopleth.models.flux_operator import MultiscaleFaceFluxOperator1D
from isopleth.models.fno import FourierNeuralOperator1D
from isopleth.models.unet import UNet1D
from isopleth.models.wavelet_operator import WaveletNeuralOperator1D


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Isopleth computational throughput and model benchmarks."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to export benchmark results as JSON.",
    )
    args = parser.parse_args()

    print("=================================================================")
    print("Isopleth Model Architecture and Performance Benchmarks")
    print("=================================================================")

    # 1. Parameter and latency profile across model architectures
    models = {
        "FNO-1D": FourierNeuralOperator1D(in_channels=1, out_channels=1, hidden_channels=32, modes=8),
        "UNet-1D": UNet1D(in_channels=1, out_channels=1, base_channels=16),
        "DeepONet-1D": DeepONet1D(in_channels=1, out_channels=1, num_sensors=64, latent_dim=64),
        "WNO-1D": WaveletNeuralOperator1D(in_channels=1, out_channels=1, hidden_channels=32),
        "MFFNO-1D (Flux)": MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=32, modes=8),
    }

    print("\n1. Architecture Parameter Footprint:")
    print(f"{'Model':<20} | {'Total Parameters':<18} | {'Trainable Parameters':<20}")
    print("-" * 65)

    suite = NeuralOperatorBenchmarkSuite()
    model_stats = {}
    for name, model in models.items():
        total_p, train_p = suite.count_parameters(model)
        model_stats[name] = {"total_parameters": total_p, "trainable_parameters": train_p}
        print(f"{name:<20} | {total_p:<18,d} | {train_p:<20,d}")

    # 2. Resolution scaling on MFFNO
    print("\n2. Resolution Scaling Profile (MFFNO-1D):")
    res_points = suite.profile_resolution_scaling(
        model_builder=lambda res: MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=32, modes=min(8, res // 4)),
        resolutions=[32, 64, 128],
        batch_size=8,
        num_repeats=15,
    )

    print(f"{'Resolution':<12} | {'Mean Latency':<16} | {'P95 Latency':<16} | {'Throughput':<18}")
    print("-" * 68)
    for p in res_points:
        print(
            f"{p.resolution:<12} | {p.mean_latency_ms:<13.3f} ms | "
            f"{p.p95_latency_ms:<13.3f} ms | {p.steps_per_second:<12.1f} steps/s"
        )

    # 3. Matched-accuracy speedup analysis
    print("\n3. Matched-Accuracy Solver vs Surrogate Speedup:")
    from isopleth.numerics.burgers import Burgers1DSolver, BurgersSolverConfig
    solver = Burgers1DSolver(BurgersSolverConfig(viscosity=0.01))
    u0_test = torch.sin(2 * 3.14159 * torch.linspace(0, 1, 64)).view(1, 1, 64)
    matched = suite.compute_matched_accuracy_speedup(
        surrogate_model=models["MFFNO-1D (Flux)"],
        solver=solver,
        u0=u0_test,
        num_coarse_steps=10,
        coarse_dt=0.01,
        target_tolerance=0.05,
    )
    print(f"  Target Relative L2 Tolerance: {matched.target_tolerance_rel_l2}")
    print(f"  Numerical Solver Time:        {matched.solver_wall_clock_ms:.2f} ms")
    print(f"  Neural Surrogate Time:        {matched.surrogate_wall_clock_ms:.2f} ms")
    print(f"  Speedup Factor:               {matched.speedup_factor:.2f}x")
    print("=================================================================")

    if args.output:
        out_dict = {
            "model_architectures": model_stats,
            "resolution_scaling": [asdict(p) for p in res_points],
            "matched_accuracy": asdict(matched),
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2)
        print(f"Exported benchmark metrics to: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
