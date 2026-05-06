"""Tests for PDEBench and The Well Gray-Scott benchmark adapters (I03)."""

import pytest
import torch

from isopleth.data.contracts import PhysicalFamily
from isopleth.data.adapters import (
    GRAY_SCOTT_REGIMES,
    PDEBenchAdapter,
    TheWellGrayScottAdapter,
    get_gray_scott_sign_audit,
)

def test_gray_scott_sign_audit():
    audit = get_gray_scott_sign_audit()
    assert audit.has_discrepancy is True
    assert "+ u * v^2" in audit.generator_equation_species_v
    assert "- u * v^2" in audit.documented_equation_species_v
    assert "positive production" in audit.adopted_sign_species_v

def test_gray_scott_regimes_coverage():
    adapter = TheWellGrayScottAdapter()
    manifest = adapter.build_bounded_manifest(total_trajectories=64)
    audit = manifest.audit()
    assert audit.passed is True
    assert audit.total_trajectories == 64

    # Verify all 6 regimes are present
    regimes_in_manifest = set()
    for rec in manifest.list_records():
        assert rec.regime_tag is not None
        regimes_in_manifest.add(rec.regime_tag)

    for regime_name in GRAY_SCOTT_REGIMES.keys():
        assert regime_name in regimes_in_manifest

def test_gray_scott_solver_simulation():
    adapter = TheWellGrayScottAdapter()
    # Run short 5-step simulation on coarse 16x16 grid
    batch = adapter.generate_reference_trajectory(
        regime_name="solitons",
        resolution=(16, 16),
        steps=5,
        dt=1.0,
        seed=123,
    )
    assert batch.values.shape == (1, 6, 16, 16, 2)
    # Check concentrations stay in physical range [0, 1.2]
    u = batch.extract_field("species_u")
    v = batch.extract_field("species_v")
    assert torch.all(u >= 0.0)
    assert torch.all(u <= 1.2)
    assert torch.all(v >= 0.0)
    assert torch.all(v <= 1.2)

def test_pdebench_adapter_manifest():
    adapter = PDEBenchAdapter()
    manifest = adapter.build_bounded_manifest(
        family=PhysicalFamily.BURGERS,
        num_trajectories=128,
    )
    assert len(manifest.list_records()) == 128
    audit = manifest.audit()
    assert audit.passed is True
