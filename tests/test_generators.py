"""Tests for simulation dataset generator harness and manifest integration."""

import json
from pathlib import Path

import h5py
import pytest

from isopleth.data.generators import (
    GeneratorJobConfig,
    PartitionBatchOrchestrator,
    TrajectoryGenerator,
)


def test_burgers_trajectory_generator_mini(tmp_path: Path):
    job = GeneratorJobConfig(
        regime_name="burgers_1d",
        num_trajectories=3,
        num_timesteps=10,
        resolution=(32,),
        dt=0.01,
        output_dir=str(tmp_path),
        partition="test_part",
        seed=123,
    )
    generator = TrajectoryGenerator(job)
    filepath, sha256 = generator.generate_burgers_dataset()

    assert filepath.exists()
    assert len(sha256) == 64  # Valid hex digest

    with h5py.File(filepath, "r") as h5f:
        assert h5f["states"].shape == (3, 10, 1, 32)
        assert h5f["parameters"].shape == (3, 2)
        assert h5f.attrs["regime"] == "burgers_1d"
        assert h5f.attrs["partition"] == "test_part"


def test_shallow_water_trajectory_generator_mini(tmp_path: Path):
    job = GeneratorJobConfig(
        regime_name="shallow_water_1d",
        num_trajectories=2,
        num_timesteps=8,
        resolution=(32,),
        dt=0.005,
        output_dir=str(tmp_path),
        partition="train",
        seed=456,
    )
    generator = TrajectoryGenerator(job)
    filepath, sha256 = generator.generate_shallow_water_1d_dataset()

    assert filepath.exists()
    assert len(sha256) == 64

    with h5py.File(filepath, "r") as h5f:
        assert h5f["states"].shape == (2, 8, 2, 32)
        assert h5f["bathymetry"].shape == (2, 32)


def test_partition_batch_orchestrator_mini(tmp_path: Path):
    orchestrator = PartitionBatchOrchestrator(
        output_dir=str(tmp_path),
        regime_name="burgers_1d",
        partition_sizes={"train": 2, "val": 1},
        num_timesteps=5,
        resolution=(32,),
        dt=0.01,
        base_seed=789,
    )
    results = orchestrator.generate_all()

    assert "train" in results
    assert "val" in results
    manifest_file = tmp_path / "burgers_1d_manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, "r") as f:
        data = json.load(f)
        assert data["regime"] == "burgers_1d"
        assert "train" in data["partitions"]
