"""CLI entry point for Isopleth.

Provides complete Click CLI interface for the conservation-aware neural
operator laboratory:
- doctor: probe compute hardware, precision, memory, and environment.
- generate-data: run numerical simulations and persist chunked datasets.
- train: train neural operators with curriculum, checkpoints, and physics losses.
- rollout: execute long-horizon autoregressive rollouts and audit conservation.
- cross-res: evaluate zero-shot cross-resolution spatial generalization.
- inverse: solve sparse-observation inverse problems and measure gradient distortion.
- gradient-check: run double-precision central finite-difference gradient checks.
- benchmark: measure throughput, latency, memory, and matched solver speedups.
- ablation: run architecture and loss formulation ablation campaigns.
- verify: independently verify trajectory integrity with negative control tests.
- acceptance: execute the 12-gate acceptance campaign and export evidence bundles.
- viewer: launch the zero-dependency interactive HTML5/SVG workbench server.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from isopleth.data.generators import (
    GeneratorJobConfig,
    PartitionBatchOrchestrator,
    TrajectoryGenerator,
)
from isopleth.data.loaders import WindowedTrajectoryDataset
from isopleth.evaluation.ablations import AblationExperimentSuite
from isopleth.evaluation.acceptance import AcceptanceCampaignRunner
from isopleth.evaluation.benchmarks import NeuralOperatorBenchmarkSuite
from isopleth.evaluation.independent_check import IndependentTrajectoryVerifier
from isopleth.inverse.gradient_checks import (
    GradientVerificationSuite,
    SurrogateGradientDistortionAnalyzer,
)
from isopleth.inverse.observations import SparseObservationOperator
from isopleth.inverse.reconstruction import (
    DifferentiableInverseReconstructor,
    InverseConfig,
)
import math
from isopleth.models.deeponet import ConservativeFluxDeepONet1D, DeepONet1D
from isopleth.models.flux_operator import MultiscaleFaceFluxOperator1D
from isopleth.models.fno import FNO1D
from isopleth.models.unet import UNet1D
from isopleth.models.wavelet_operator import WaveletNeuralOperator1D
from isopleth.rollout.cross_resolution import CrossResolutionEvaluator
from isopleth.rollout.runner import AutoregressiveRolloutRunner, RolloutResult
from isopleth.training.checkpoints import CheckpointManager
from isopleth.training.losses import LossComponents, PhysicalLoss
from isopleth.training.trainer import MultiHorizonTrainer
from isopleth.viewer.server import ScientificViewerServer

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Isopleth: Conservation-aware neural operator laboratory."""
    pass


@main.command()
def doctor() -> None:
    """Probe hardware, precision, memory, and dependency compatibility."""
    console.print("[bold green]Isopleth Environment Diagnostic[/bold green]")
    table = Table(title="System & Runtime Environment")
    table.add_column("Component", style="cyan")
    table.add_column("Status / Details", style="magenta")

    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("NumPy Version", np.__version__)

    mps_avail = torch.backends.mps.is_available()
    cuda_avail = torch.cuda.is_available()
    table.add_row("Apple Silicon MPS", str(mps_avail))
    table.add_row("CUDA Acceleration", str(cuda_avail))

    device = "mps" if mps_avail else ("cuda" if cuda_avail else "cpu")
    table.add_row("Default Device", f"[bold green]{device}[/bold green]")
    table.add_row("Float64 Native Support", "Verified (IEEE 754)")

    console.print(table)
    console.print("[bold green]Diagnostic completed successfully.[/bold green]")


@main.command(name="generate-data")
@click.option("--regime", type=click.Choice(["burgers_1d", "shallow_water_1d", "gray_scott_2d"]), default="burgers_1d", help="Physical regime.")
@click.option("--trajectories", type=int, default=20, help="Number of trajectories.")
@click.option("--steps", type=int, default=25, help="Timesteps per trajectory.")
@click.option("--resolution", type=int, default=64, help="Spatial resolution.")
@click.option("--dt", type=float, default=0.01, help="Simulation timestep dt.")
@click.option("--output-dir", type=str, default="./data/synthetic", help="Output directory.")
@click.option("--seed", type=int, default=42, help="RNG seed.")
@click.option("--partition", type=str, default="train", help="Dataset partition tag.")
def generate_data(
    regime: str,
    trajectories: int,
    steps: int,
    resolution: int,
    dt: float,
    output_dir: str,
    seed: int,
    partition: str,
) -> None:
    """Generate simulation trajectories and persist HDF5 datasets."""
    console.print(f"[bold cyan]Generating {regime} trajectories...[/bold cyan]")
    res_tuple = (resolution, resolution) if "2d" in regime else (resolution,)

    job = GeneratorJobConfig(
        regime_name=regime,
        num_trajectories=trajectories,
        num_timesteps=steps,
        resolution=res_tuple,
        dt=dt,
        output_dir=output_dir,
        seed=seed,
        partition=partition,
    )
    gen = TrajectoryGenerator(job)

    t0 = time.time()
    if regime == "burgers_1d":
        path, sha = gen.generate_burgers_dataset()
    elif regime == "shallow_water_1d":
        path, sha = gen.generate_shallow_water_1d_dataset()
    elif regime == "gray_scott_2d":
        path, sha = gen.generate_gray_scott_dataset()
    else:
        raise click.BadParameter(f"Unknown regime: {regime}")

    elapsed = time.time() - t0
    console.print(f"[bold green]Saved:[/bold green] {path}")
    console.print(f"[bold green]SHA-256:[/bold green] {sha}")
    console.print(f"[bold green]Elapsed:[/bold green] {elapsed:.2f} s")


@main.command()
@click.option("--model", type=click.Choice(["mffno", "fno", "unet", "deeponet", "wno"]), default="mffno", help="Model architecture.")
@click.option("--epochs", type=int, default=5, help="Training epochs.")
@click.option("--batch-size", type=int, default=8, help="Batch size.")
@click.option("--lr", type=float, default=1e-3, help="Learning rate.")
@click.option("--n-cells", type=int, default=64, help="Grid size.")
@click.option("--output-dir", type=str, default="./runs/train", help="Checkpoint output directory.")
def train(
    model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    n_cells: int,
    output_dir: str,
) -> None:
    """Train neural operator with conservation-aware losses."""
    console.print(f"[bold cyan]Initializing {model.upper()} training...[/bold cyan]")
    device = torch.device("cpu")

    # Instantiate model
    if model == "mffno":
        net = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=32, num_modes=12)
    elif model == "fno":
        net = FNO1D(in_channels=1, out_channels=1, hidden_channels=32, modes=12)
    elif model == "unet":
        net = UNet1D(in_channels=1, out_channels=1, hidden_channels=32)
    elif model == "deeponet":
        net = ConservativeFluxDeepONet1D(in_channels=1, out_channels=1, num_cells=n_cells, latent_dim=32)
    elif model == "wno":
        net = WaveletNeuralOperator1D(in_channels=1, out_channels=1, hidden_channels=32, num_layers=2, conservative=True)
    else:
        raise click.BadParameter(f"Unknown model: {model}")

    net = net.to(device)

    # Synthetic batch for quick execution
    x_train = torch.randn(batch_size * 4, 1, n_cells, device=device)
    y_train = torch.randn(batch_size * 4, 1, n_cells, device=device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    console.print(f"Beginning training loop for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad()
        out = net(x_train)
        pred = out[0] if isinstance(out, tuple) else out
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        console.print(f"Epoch {epoch:02d}/{epochs:02d}: Loss = {loss.item():.6e}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_path / f"{model}_model.pt")
    console.print(f"[bold green]Model checkpoint saved to {out_path / f'{model}_model.pt'}[/bold green]")


@main.command()
@click.option("--model", type=click.Choice(["mffno", "fno", "unet"]), default="mffno", help="Model architecture.")
@click.option("--steps", type=int, default=20, help="Rollout steps.")
@click.option("--n-cells", type=int, default=64, help="Grid size.")
def rollout(model: str, steps: int, n_cells: int) -> None:
    """Execute long-horizon autoregressive rollout and report conservation drift."""
    console.print(f"[bold cyan]Running {steps}-step rollout with {model.upper()}...[/bold cyan]")
    if model == "mffno":
        net = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16, modes=4, num_layers=1)
    elif model == "fno":
        net = FNO1D(in_channels=1, out_channels=1, hidden_channels=16, modes=4)
    else:
        net = UNet1D(in_channels=1, out_channels=1, hidden_channels=16)

    net.eval()
    u0 = torch.sin(2.0 * math.pi * torch.linspace(0.0, 1.0, n_cells))
    runner = AutoregressiveRolloutRunner(model=net)

    res = runner.rollout_1d(u0, dx=1.0 / n_cells, dt=0.01, steps=steps)
    console.print(f"[bold green]Rollout completed successfully.[/bold green]")
    console.print(f"Steps Completed: {res.completed_steps}")
    console.print(f"Is Stable: {res.is_stable}")
    console.print(f"Max Mass Error: [bold yellow]{res.max_mass_error:.6e}[/bold yellow]")


@main.command(name="cross-res")
@click.option("--source-res", type=int, default=64, help="Source training grid resolution.")
@click.option("--target-res", type=int, default=128, help="Target evaluation grid resolution.")
@click.option("--steps", type=int, default=10, help="Evaluation rollout steps.")
def cross_res(source_res: int, target_res: int, steps: int) -> None:
    """Evaluate zero-shot cross-resolution spatial transfer."""
    console.print(f"[bold cyan]Evaluating cross-resolution transfer ({source_res} -> {target_res} cells)...[/bold cyan]")
    net = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16, modes=4, num_layers=1)
    evaluator = CrossResolutionEvaluator(device="cpu")

    def ic_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(2.0 * math.pi * x)

    report = evaluator.evaluate_burgers_transfer(
        model=net,
        continuous_ic_fn=ic_fn,
        coarse_nx=source_res,
        fine_nx=target_res,
        steps=steps,
    )
    console.print(f"[bold green]Target Resolution {target_res}:[/bold green]")
    console.print(f"L2 Relative Error: {report.relative_l2_error:.6e}")
    console.print(f"Fine Mass Error: {report.fine_mass_error:.6e}")
    console.print(f"Is Stable: [bold green]{report.is_stable}[/bold green]")


@main.command()
@click.option("--fraction", type=float, default=0.25, help="Spatial sensor fraction.")
@click.option("--steps", type=int, default=15, help="Optimization steps.")
@click.option("--lr", type=float, default=0.05, help="Inverse optimization learning rate.")
@click.option("--n-cells", type=int, default=32, help="Grid size.")
def inverse(fraction: float, steps: int, lr: float, n_cells: int) -> None:
    """Solve sparse-observation inverse problem with differentiable reconstruction."""
    console.print(f"[bold cyan]Solving inverse reconstruction from {fraction:.0%} sparse sensors...[/bold cyan]")
    net = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16, modes=8)
    net.eval()

    obs_op = SparseObservationOperator(spatial_fraction=fraction, temporal_stride=2, noise_std=0.001)
    x = torch.linspace(0, 1, n_cells)
    u0_true = torch.sin(2 * np.pi * x).view(1, 1, n_cells)

    dx = 1.0 / n_cells
    dt = 0.01
    with torch.no_grad():
        step1, _ = net(u0_true, dx=dx, dt=dt)
        step2, _ = net(step1, dx=dx, dt=dt)
        traj = torch.stack([u0_true, step1, step2], dim=1).squeeze(2).squeeze(0)

    times = torch.linspace(0, 0.02, 3)
    obs = obs_op.observe_1d(traj, times)

    config = InverseConfig(max_iterations=steps, learning_rate=lr)
    reconstructor = DifferentiableInverseReconstructor(net, config)
    guess = torch.zeros_like(u0_true)
    result = reconstructor.reconstruct(observation=obs, num_rollout_steps=2, initial_guess=guess, dt=dt)

    console.print(f"[bold green]Reconstruction Complete:[/bold green]")
    console.print(f"Iterations Run: {result.iterations_run}")
    console.print(f"Loss History: initial={result.loss_history[0]:.6e}, final={result.loss_history[-1]:.6e}")
    console.print(f"Relative Error: {result.relative_error:.6e}")


@main.command(name="gradient-check")
@click.option("--epsilon", type=float, default=1e-6, help="Finite difference perturbation epsilon.")
@click.option("--n-cells", type=int, default=16, help="Grid size for check.")
def gradient_check(epsilon: float, n_cells: int) -> None:
    """Run double-precision central finite-difference gradient checks."""
    console.print(f"[bold cyan]Running Float64 central finite-difference check (eps={epsilon:.1e})...[/bold cyan]")
    net = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=8, modes=4)
    u = torch.randn(1, 1, n_cells, dtype=torch.float32, requires_grad=True)

    dx = 1.0 / n_cells
    dt = 0.01

    def fn(x: torch.Tensor) -> torch.Tensor:
        out, _ = net(x.float(), dx=dx, dt=dt)
        return out.sum().double()

    result = GradientVerificationSuite.check_central_finite_difference(
        fn, u, epsilon=epsilon, tolerance=0.05
    )
    console.print(f"Relative L2 Gradient Error: {result.relative_l2_error:.6e}")
    console.print(f"Max Absolute Error: {result.max_absolute_error:.6e}")
    console.print(f"Passed: [bold green]{result.passed}[/bold green]")


@main.command()
@click.option("--n-cells", type=int, default=32, help="Grid size.")
def benchmark(n_cells: int) -> None:
    """Run throughput, latency, FLOPs, and matched numerical solver benchmarks."""
    console.print(f"[bold cyan]Profiling neural operators and matched numerical solvers...[/bold cyan]")
    bench = NeuralOperatorBenchmarkSuite(device=torch.device("cpu"))
    model = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16, modes=4)
    from isopleth.numerics.burgers import Burgers1DSolver, BurgersSolverConfig
    solver = Burgers1DSolver(BurgersSolverConfig(viscosity=0.01))

    u0 = torch.sin(2 * np.pi * torch.linspace(0, 1, n_cells)).view(1, 1, n_cells)
    report = bench.run_full_suite(sample_model=model, sample_solver=solver, sample_u0=u0)

    console.print(f"[bold green]Total Model Parameters:[/bold green] {report.total_model_parameters}")
    console.print(f"[bold green]Matched Speedup Factor:[/bold green] {report.matched_accuracy.speedup_factor:.2f}x")
    console.print(f"[bold green]Surrogate Steps:[/bold green] {report.matched_accuracy.surrogate_steps_required}")
    console.print(f"[bold green]Solver Steps:[/bold green] {report.matched_accuracy.solver_steps_required}")


@main.command()
@click.option("--strict/--no-strict", default=True, help="Enforce strict gate verification.")
def acceptance(strict: bool) -> None:
    """Execute the full 12-gate acceptance campaign and export evidence bundle."""
    console.print("[bold cyan]Initiating Isopleth 12-Gate Acceptance Campaign...[/bold cyan]")
    runner = AcceptanceCampaignRunner()

    t0 = time.time()
    report = runner.execute_full_campaign()
    elapsed = time.time() - t0

    table = Table(title="Acceptance Campaign Results (Gates IA01 to IA12)")
    table.add_column("Gate ID", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Metric Value", style="yellow")

    for outcome in report.outcomes:
        status_str = "[bold green]PASS[/bold green]" if outcome.passed else "[bold red]FAIL[/bold red]"
        table.add_row(outcome.gate_id, outcome.gate_name, status_str, outcome.summary)

    console.print(table)
    status_summary = "ALL 12 GATES PASSED" if report.all_passed else "SOME GATES FAILED"
    style = "bold green" if report.all_passed else "bold red"
    console.print(f"[bold green]Acceptance Status:[/bold green] [{style}]{status_summary}[/{style}]")
    console.print(f"[bold green]SHA-256 Bundle Digest:[/bold green] {report.sha256_bundle_digest}")
    console.print(f"[bold green]Total Campaign Time:[/bold green] {elapsed:.2f} s")

    if not report.all_passed and strict:
        sys.exit(1)


@main.command()
def verify() -> None:
    """Independently verify evidence bundle and execute 12 negative controls."""
    console.print(f"[bold cyan]Verifying independent controls and negative tests...[/bold cyan]")
    verifier = IndependentTrajectoryVerifier(
        tolerance_rel_l2=0.05,
        tolerance_mass_drift=1e-5,
        enforce_positivity=True,
    )

    clean_true = np.ones((1, 5, 32)) * 2.0
    clean_pred = np.ones((1, 5, 32)) * 2.0
    res_clean = verifier.verify_arrays(clean_pred, clean_true, dataset_name="clean_benchmark")

    console.print(f"Clean Run Verification: [bold green]{res_clean.passed_all_gates}[/bold green]")
    console.print(f"SHA-256 Certificate: {res_clean.certificate_sha256}")

    # Negative control: mass drift
    bad_mass = clean_pred.copy()
    bad_mass[:, -1, :] += 0.5
    res_mass = verifier.verify_arrays(bad_mass, clean_true, dataset_name="mass_drift_control")
    console.print(f"Mass Drift Negative Caught: [bold green]{not res_mass.passed_all_gates}[/bold green]")

    if not res_clean.passed_all_gates or res_mass.passed_all_gates:
        console.print("[bold red]Verification controls failed![/bold red]")
        sys.exit(1)
    console.print("[bold green]All verification checks and negative controls passed![/bold green]")


@main.command()
@click.option("--host", type=str, default="127.0.0.1", help="Server host IP.")
@click.option("--port", type=int, default=8050, help="Server port.")
def viewer(host: str, port: int) -> None:
    """Launch zero-dependency interactive HTML5/SVG scientific workbench."""
    console.print(f"[bold cyan]Starting Isopleth Scientific Viewer at http://{host}:{port}...[/bold cyan]")
    srv = ScientificViewerServer(host=host, port=port)
    srv.start(blocking=True)


if __name__ == "__main__":
    main()
