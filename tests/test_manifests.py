"""Tests for trajectory manifests and partition split auditing (I02)."""

import pytest
import tempfile
from pathlib import Path

from isopleth.data.contracts import PhysicalFamily, Role
from isopleth.data.manifests import (
    DatasetManifest,
    TrajectoryRecord,
    create_synthetic_manifest_partition,
)

def test_manifest_partition_integrity():
    manifest = create_synthetic_manifest_partition(
        family=PhysicalFamily.BURGERS,
        num_train=28,
        num_val=4,
        num_cal=3,
        num_test=5,
    )
    audit = manifest.audit()
    assert audit.passed is True
    assert audit.total_trajectories == 40
    assert audit.counts_by_role["train"] == 28
    assert audit.counts_by_role["validation"] == 4
    assert audit.counts_by_role["calibration"] == 3
    assert audit.counts_by_role["test"] == 5
    assert audit.is_disjoint is True

def test_manifest_leakage_detection():
    manifest = DatasetManifest()
    rec1 = TrajectoryRecord(
        trajectory_id="traj_overlap_01",
        family="burgers_1d",
        role="train",
        seed=1,
        parameters={},
        resolution=(64,),
        time_step=0.01,
        time_horizon=1.0,
        total_steps=100,
        storage_path="path/1",
        sha256_checksum="abc",
    )
    manifest.add_record(rec1)

    # Attempting to add duplicate trajectory_id directly raises ValueError
    with pytest.raises(ValueError, match="Duplicate trajectory_id detected"):
        manifest.add_record(rec1)

def test_manifest_serialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "manifest.json"
        manifest = create_synthetic_manifest_partition(
            family=PhysicalFamily.SHALLOW_WATER,
            num_train=10,
            num_val=2,
            num_cal=2,
            num_test=2,
        )
        manifest.save_json(path)
        assert path.exists()

        loaded = DatasetManifest.load_json(path)
        audit = loaded.audit()
        assert audit.passed is True
        assert audit.total_trajectories == 16
