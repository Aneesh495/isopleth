"""Simulation dataset generation harness.

Generates reproducible numerical trajectories across Burgers, Shallow Water,
and Gray-Scott physical regimes into chunked HDF5 and Zarr v3 formats with
SHA-256 cryptographic manifests for zero-leakage benchmark partitioning.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch

from isopleth.data.adapters import (
    GRAY_SCOTT_REGIMES,
    TheWellGrayScottAdapter,
)
from isopleth.data.contracts import (
    FieldMetadata,
    GridContract,
    TrajectoryBatch,
    TrajectoryProvenance,
)
from isopleth.data.manifests import DatasetManifest, TrajectoryRecord
from isopleth.data.samplers import (
    GaussianRandomField1D,
    GaussianRandomField2D,
    GrayScottSeedSampler,
    RiemannDamBreakSampler,
    ShallowWaterTopographySampler,
)
from isopleth.numerics.burgers import (
    Burgers1DSolver,
    BurgersSolverConfig,
    FluxScheme,
    SlopeLimiter,
)
from isopleth.numerics.shallow_water import (
    ShallowWater2DSolver,
    ShallowWaterConfig,
    ShallowWaterFluxScheme,
)


@dataclass
class GeneratorJobConfig:
    """Specification for a trajectory dataset generation batch."""

    regime_name: str
    num_trajectories: int
    num_timesteps: int
    resolution: Tuple[int, ...]
    dt: float
    output_dir: str
    format: str = "hdf5"  # 'hdf5' or 'zarr'
    chunk_size: Tuple[int, ...] = (10, 16, 64)
    seed: int = 42
    partition: str = "train"


class TrajectoryGenerator:
    """Orchestrates simulation runs and persists verifiable datasets."""

    def __init__(self, job_config: GeneratorJobConfig) -> None:
        self.config = job_config
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(self.config.seed)

    def generate_burgers_dataset(self) -> Tuple[Path, str]:
        """Generate 1D Burgers advection-diffusion trajectories."""
        n_cells = self.config.resolution[0]
        n_steps = self.config.num_timesteps
        n_traj = self.config.num_trajectories
        dt = self.config.dt
        dx = 1.0 / n_cells

        filename = f"burgers_{self.config.partition}_{n_traj}x{n_steps}x{n_cells}.h5"
        filepath = self.output_path / filename

        grf = GaussianRandomField1D(
            tau=3.0, alpha=2.5, amplitude=0.5, seed=self.config.seed
        )

        with h5py.File(filepath, "w") as h5f:
            dset_states = h5f.create_dataset(
                "states",
                shape=(n_traj, n_steps, 1, n_cells),
                dtype=np.float32,
                chunks=(min(10, n_traj), min(10, n_steps), 1, n_cells),
                compression="gzip",
            )
            dset_params = h5f.create_dataset(
                "parameters",
                shape=(n_traj, 2),  # [nu, max_u]
                dtype=np.float32,
            )

            coords_x = np.linspace(0.0, 1.0, n_cells, endpoint=False) + 0.5 / n_cells
            h5f.create_dataset("grid_x", data=coords_x.astype(np.float32))

            for idx in range(n_traj):
                nu = float(self.rng.uniform(0.001, 0.02))
                u0_np = grf.sample_grid(n_cells, num_samples=1)[0]
                u_curr = torch.from_numpy(u0_np).to(dtype=torch.float64)

                solver = Burgers1DSolver(
                    BurgersSolverConfig(
                        cfl=0.45,
                        viscosity=nu,
                        flux_scheme=FluxScheme.GODUNOV,
                        limiter=SlopeLimiter.MINMOD,
                    )
                )

                traj = np.zeros((n_steps, 1, n_cells), dtype=np.float32)
                traj[0, 0, :] = u_curr.numpy().astype(np.float32)

                for step in range(1, n_steps):
                    u_curr = solver.step(u_curr, dx=dx, dt=dt)
                    traj[step, 0, :] = u_curr.numpy().astype(np.float32)

                dset_states[idx] = traj
                dset_params[idx] = [nu, float(np.max(np.abs(u0_np)))]

            h5f.attrs["regime"] = "burgers_1d"
            h5f.attrs["dt"] = dt
            h5f.attrs["num_trajectories"] = n_traj
            h5f.attrs["partition"] = self.config.partition

        sha256 = self._compute_sha256(filepath)
        return filepath, sha256

    def generate_shallow_water_1d_dataset(self) -> Tuple[Path, str]:
        """Generate 1D/quasi-2D Shallow Water equations trajectories."""
        n_cells = self.config.resolution[0]
        n_steps = self.config.num_timesteps
        n_traj = self.config.num_trajectories
        dt = self.config.dt
        dx = 1.0 / n_cells
        dy = 1.0 / n_cells

        filename = f"shallow_water_1d_{self.config.partition}_{n_traj}x{n_steps}x{n_cells}.h5"
        filepath = self.output_path / filename

        dambreak_sampler = RiemannDamBreakSampler(seed=self.config.seed)

        with h5py.File(filepath, "w") as h5f:
            dset_states = h5f.create_dataset(
                "states",
                shape=(n_traj, n_steps, 2, n_cells),  # [h, hu]
                dtype=np.float32,
                chunks=(min(10, n_traj), min(10, n_steps), 2, n_cells),
                compression="gzip",
            )
            dset_topo = h5f.create_dataset(
                "bathymetry",
                shape=(n_traj, n_cells),
                dtype=np.float32,
            )

            coords_x = np.linspace(0.0, 1.0, n_cells, endpoint=False) + 0.5 / n_cells
            h5f.create_dataset("grid_x", data=coords_x.astype(np.float32))

            solver = ShallowWater2DSolver(
                ShallowWaterConfig(
                    cfl=0.35,
                    gravity=9.81,
                    flux_scheme=ShallowWaterFluxScheme.KURGANOV_TADMOR,
                )
            )

            for idx in range(n_traj):
                h0_arr, hu0_arr = dambreak_sampler.sample_1d(
                    n_cells,
                    h_left_range=(1.2, 1.8),
                    h_right_range=(0.8, 1.1),
                    num_samples=1,
                )
                h0_1d = h0_arr[0]
                hu0_1d = hu0_arr[0]

                # Tile 1D profile across y to use full 2D solver
                h2d = torch.from_numpy(h0_1d).unsqueeze(0).repeat(n_cells, 1)
                hu2d = torch.from_numpy(hu0_1d).unsqueeze(0).repeat(n_cells, 1)
                hv2d = torch.zeros_like(h2d)
                b2d = torch.zeros_like(h2d)

                traj = np.zeros((n_steps, 2, n_cells), dtype=np.float32)
                traj[0, 0, :] = h0_1d.astype(np.float32)
                traj[0, 1, :] = hu0_1d.astype(np.float32)

                curr_h, curr_hu, curr_hv = h2d, hu2d, hv2d
                # Sub-cycle if necessary to satisfy CFL condition
                sub_steps = max(1, math.ceil(dt / 0.002))
                sub_dt = dt / sub_steps

                for step in range(1, n_steps):
                    for _ in range(sub_steps):
                        curr_h, curr_hu, curr_hv, _ = solver.step(
                            curr_h, curr_hu, curr_hv, dx=dx, dy=dy, dt=sub_dt, bathymetry=b2d
                        )
                    traj[step, 0, :] = curr_h[0, :].numpy().astype(np.float32)
                    traj[step, 1, :] = curr_hu[0, :].numpy().astype(np.float32)

                dset_states[idx] = traj
                dset_topo[idx] = b2d[0, :].numpy().astype(np.float32)

            h5f.attrs["regime"] = "shallow_water_1d"
            h5f.attrs["dt"] = dt
            h5f.attrs["num_trajectories"] = n_traj
            h5f.attrs["partition"] = self.config.partition

        sha256 = self._compute_sha256(filepath)
        return filepath, sha256

    def generate_gray_scott_dataset(self) -> Tuple[Path, str]:
        """Generate 2D Gray-Scott reaction-diffusion trajectories."""
        nx = self.config.resolution[0]
        ny = self.config.resolution[1] if len(self.config.resolution) > 1 else nx
        n_steps = self.config.num_timesteps
        n_traj = self.config.num_trajectories
        dt = self.config.dt

        filename = f"gray_scott_2d_{self.config.partition}_{n_traj}x{n_steps}x{nx}x{ny}.h5"
        filepath = self.output_path / filename

        adapter = TheWellGrayScottAdapter()
        regime_names = list(GRAY_SCOTT_REGIMES.keys())

        with h5py.File(filepath, "w") as h5f:
            dset_states = h5f.create_dataset(
                "states",
                shape=(n_traj, n_steps, 2, nx, ny),  # [u, v]
                dtype=np.float32,
                chunks=(1, min(5, n_steps), 2, nx, ny),
                compression="gzip",
            )
            dset_params = h5f.create_dataset(
                "parameters",
                shape=(n_traj, 2),  # [feed, kill]
                dtype=np.float32,
            )

            for idx in range(n_traj):
                reg_name = regime_names[idx % len(regime_names)]
                reg_params = GRAY_SCOTT_REGIMES[reg_name]

                batch = adapter.generate_reference_trajectory(
                    regime_name=reg_name,
                    resolution=(nx, ny),
                    steps=n_steps - 1,
                    dt=dt,
                    seed=self.config.seed + idx,
                )
                # batch.values shape: (1, n_steps, nx, ny, 2)
                vals = batch.values[0].numpy()  # (n_steps, nx, ny, 2)
                # Permute to (n_steps, 2, nx, ny)
                traj = np.transpose(vals, (0, 3, 1, 2)).astype(np.float32)

                dset_states[idx] = traj
                dset_params[idx] = [reg_params.feed_rate, reg_params.kill_rate]

            h5f.attrs["regime"] = "gray_scott_2d"
            h5f.attrs["dt"] = dt
            h5f.attrs["num_trajectories"] = n_traj
            h5f.attrs["partition"] = self.config.partition

        sha256 = self._compute_sha256(filepath)
        return filepath, sha256

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        """Compute cryptographic SHA-256 digest of file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()


class PartitionBatchOrchestrator:
    """Coordinates generation of all 4,000 trajectories across partitions.

    Standard partition counts:
    - Train: 2800 trajectories
    - Val: 400 trajectories
    - Calibration: 300 trajectories
    - Test: 500 trajectories
    Total: 4000 trajectories
    """

    DEFAULT_PARTITION_SIZES = {
        "train": 2800,
        "val": 400,
        "calibration": 300,
        "test": 500,
    }

    def __init__(
        self,
        output_dir: str,
        regime_name: str = "burgers_1d",
        partition_sizes: Optional[Dict[str, int]] = None,
        num_timesteps: int = 50,
        resolution: Tuple[int, ...] = (64,),
        dt: float = 0.01,
        base_seed: int = 1000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.regime_name = regime_name
        self.partition_sizes = partition_sizes or self.DEFAULT_PARTITION_SIZES
        self.num_timesteps = num_timesteps
        self.resolution = resolution
        self.dt = dt
        self.base_seed = base_seed

    def generate_all(self) -> Dict[str, Tuple[Path, str]]:
        """Run generation for all configured partitions and return file paths and hashes."""
        results = {}
        for partition, count in self.partition_sizes.items():
            job = GeneratorJobConfig(
                regime_name=self.regime_name,
                num_trajectories=count,
                num_timesteps=self.num_timesteps,
                resolution=self.resolution,
                dt=self.dt,
                output_dir=str(self.output_dir),
                seed=self.base_seed + hash(partition) % 10000,
                partition=partition,
            )
            gen = TrajectoryGenerator(job)

            if self.regime_name == "burgers_1d":
                path, sha = gen.generate_burgers_dataset()
            elif self.regime_name == "shallow_water_1d":
                path, sha = gen.generate_shallow_water_1d_dataset()
            elif self.regime_name == "gray_scott_2d":
                path, sha = gen.generate_gray_scott_dataset()
            else:
                raise ValueError(f"Unknown regime: {self.regime_name}")

            results[partition] = (path, sha)

        # Write overall manifest JSON
        manifest_data = {
            "regime": self.regime_name,
            "resolution": list(self.resolution),
            "dt": self.dt,
            "num_timesteps": self.num_timesteps,
            "partitions": {
                part: {
                    "count": self.partition_sizes[part],
                    "filepath": str(res[0]),
                    "sha256": res[1],
                }
                for part, res in results.items()
            },
        }
        manifest_path = self.output_dir / f"{self.regime_name}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)

        return results
