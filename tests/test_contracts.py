"""Tests for physical field, grid, time, and unit contracts (I01)."""

import pytest
import torch
import numpy as np

from isopleth.data.contracts import (
    BoundaryCondition,
    GridContract,
    PhysicalFamily,
    TrajectoryBatch,
    TrajectoryProvenance,
    BURGERS_FIELDS,
    SHALLOW_WATER_FIELDS,
)

def test_grid_contract_validation():
    # Valid 1D grid
    grid_1d = GridContract(
        spatial_dim=1,
        resolution=(64,),
        domain_bounds=((0.0, 1.0),),
        boundary_conditions=(BoundaryCondition.PERIODIC,),
        time_step=0.01,
        time_horizon=1.0,
    )
    assert grid_1d.cell_measures == (1.0 / 64.0,)
    assert len(grid_1d.cell_centers()[0]) == 64
    assert len(grid_1d.cell_faces()[0]) == 65
    assert grid_1d.total_steps == 100

    # Invalid spatial dimension
    with pytest.raises(ValueError):
        GridContract(
            spatial_dim=3,
            resolution=(64, 64, 64),
            domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            boundary_conditions=(BoundaryCondition.PERIODIC,) * 3,
            time_step=0.01,
            time_horizon=1.0,
        )

    # Negative dt
    with pytest.raises(ValueError):
        GridContract(
            spatial_dim=1,
            resolution=(64,),
            domain_bounds=((0.0, 1.0),),
            boundary_conditions=(BoundaryCondition.PERIODIC,),
            time_step=-0.01,
            time_horizon=1.0,
        )

def test_trajectory_batch_validation():
    grid = GridContract(
        spatial_dim=1,
        resolution=(64,),
        domain_bounds=((0.0, 1.0),),
        boundary_conditions=(BoundaryCondition.PERIODIC,),
        time_step=0.05,
        time_horizon=0.5,
    )

    prov = TrajectoryProvenance(
        trajectory_id="test_001",
        family=PhysicalFamily.BURGERS,
        generator_name="test_gen",
        generator_version="1.0",
        solver_accuracy_order=2,
        seed=42,
        parameters={"viscosity": 0.01},
        creation_timestamp="2026-10-09T00:00:00Z",
    )

    values = torch.zeros(1, 11, 64)
    times = torch.linspace(0.0, 0.5, 11)

    batch = TrajectoryBatch(
        family=PhysicalFamily.BURGERS,
        grid=grid,
        field_names=("u",),
        values=values,
        times=times,
        provenance=[prov],
    )
    assert batch.extract_field("u").shape == (1, 11, 64)

    # Non-finite values rejected
    values_nan = values.clone()
    values_nan[0, 5, 10] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        TrajectoryBatch(
            family=PhysicalFamily.BURGERS,
            grid=grid,
            field_names=("u",),
            values=values_nan,
            times=times,
            provenance=[prov],
        )

    # Non-monotonic times rejected
    times_bad = times.clone()
    times_bad[5] = times_bad[4] - 0.1
    with pytest.raises(ValueError, match="strictly monotonically increasing"):
        TrajectoryBatch(
            family=PhysicalFamily.BURGERS,
            grid=grid,
            field_names=("u",),
            values=values,
            times=times_bad,
            provenance=[prov],
        )

def test_shallow_water_positivity_contract():
    grid = GridContract(
        spatial_dim=2,
        resolution=(32, 32),
        domain_bounds=((0.0, 1.0), (0.0, 1.0)),
        boundary_conditions=(BoundaryCondition.PERIODIC, BoundaryCondition.PERIODIC),
        time_step=0.05,
        time_horizon=0.1,
    )
    prov = TrajectoryProvenance(
        trajectory_id="sw_test_01",
        family=PhysicalFamily.SHALLOW_WATER,
        generator_name="sw_gen",
        generator_version="1.0",
        solver_accuracy_order=2,
        seed=1,
        parameters={"gravity": 9.81},
        creation_timestamp="2026-10-09T00:00:00Z",
    )
    times = torch.tensor([0.0, 0.05, 0.1])
    # h must be positive
    bad_values = torch.ones(1, 3, 32, 32, 3)
    bad_values[0, 1, 10, 10, 0] = -0.05  # h < 0 violates positivity!

    with pytest.raises(ValueError, match="violated positivity contract"):
        TrajectoryBatch(
            family=PhysicalFamily.SHALLOW_WATER,
            grid=grid,
            field_names=("h", "hu", "hv"),
            values=bad_values,
            times=times,
            provenance=[prov],
        )
