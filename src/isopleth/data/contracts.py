"""Physical field, grid, time, and unit contracts for Isopleth.

Every trajectory is represented as a strictly validated, versioned array structure
with explicit axis specifications, physical cell measures, units, boundary semantics,
and provenance tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

class BoundaryCondition(str, Enum):
    """Supported physical boundary conditions."""
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    CLOSED_SLIP = "closed_slip"
    CLOSED_NOSLIP = "closed_noslip"

class PhysicalFamily(str, Enum):
    """Supported physical equation families."""
    BURGERS = "burgers_1d"
    SHALLOW_WATER = "shallow_water_2d"
    GRAY_SCOTT = "gray_scott_2d"

class Role(str, Enum):
    """Dataset partition role."""
    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    TEST = "test"

@dataclass(frozen=True)
class FieldMetadata:
    """Metadata describing a single physical field."""
    name: str
    unit: str
    description: str
    min_physical_bound: Optional[float] = None
    max_physical_bound: Optional[float] = None
    is_positive_definite: bool = False

    def validate_bounds(self, array: np.ndarray | torch.Tensor) -> None:
        """Validate that all field values obey physical boundaries."""
        if isinstance(array, torch.Tensor):
            arr = array.detach().cpu().numpy()
        else:
            arr = array

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Field '{self.name}' contains non-finite values (NaN or Inf).")

        if self.is_positive_definite:
            min_val = float(np.min(arr))
            if min_val < 0.0:
                raise ValueError(
                    f"Positive-definite field '{self.name}' violated positivity contract with minimum {min_val:.6e}."
                )

        if self.min_physical_bound is not None:
            min_val = float(np.min(arr))
            if min_val < self.min_physical_bound - 1e-7:
                raise ValueError(
                    f"Field '{self.name}' violated minimum bound {self.min_physical_bound} with observed {min_val:.6e}."
                )

        if self.max_physical_bound is not None:
            max_val = float(np.max(arr))
            if max_val > self.max_physical_bound + 1e-7:
                raise ValueError(
                    f"Field '{self.name}' violated maximum bound {self.max_physical_bound} with observed {max_val:.6e}."
                )

# Canonical field catalogs for supported physical systems
BURGERS_FIELDS: Dict[str, FieldMetadata] = {
    "u": FieldMetadata(
        name="u",
        unit="m/s",
        description="Velocity field in 1D viscous Burgers flow",
        min_physical_bound=-20.0,
        max_physical_bound=20.0,
        is_positive_definite=False,
    )
}

SHALLOW_WATER_FIELDS: Dict[str, FieldMetadata] = {
    "h": FieldMetadata(
        name="h",
        unit="m",
        description="Fluid layer depth (positive definite)",
        min_physical_bound=0.0,
        max_physical_bound=100.0,
        is_positive_definite=True,
    ),
    "hu": FieldMetadata(
        name="hu",
        unit="m^2/s",
        description="Zonal depth-integrated discharge",
        min_physical_bound=-500.0,
        max_physical_bound=500.0,
        is_positive_definite=False,
    ),
    "hv": FieldMetadata(
        name="hv",
        unit="m^2/s",
        description="Meridional depth-integrated discharge",
        min_physical_bound=-500.0,
        max_physical_bound=500.0,
        is_positive_definite=False,
    ),
}

GRAY_SCOTT_FIELDS: Dict[str, FieldMetadata] = {
    "species_u": FieldMetadata(
        name="species_u",
        unit="dimensionless",
        description="Chemical species U concentration",
        min_physical_bound=0.0,
        max_physical_bound=1.2,
        is_positive_definite=True,
    ),
    "species_v": FieldMetadata(
        name="species_v",
        unit="dimensionless",
        description="Chemical species V concentration",
        min_physical_bound=0.0,
        max_physical_bound=1.2,
        is_positive_definite=True,
    ),
}

@dataclass
class GridContract:
    """Discrete spatial and temporal geometry contract."""
    spatial_dim: int
    resolution: Tuple[int, ...]
    domain_bounds: Tuple[Tuple[float, float], ...]
    boundary_conditions: Tuple[BoundaryCondition, ...]
    time_step: float
    time_horizon: float

    def __post_init__(self) -> None:
        if self.spatial_dim not in (1, 2):
            raise ValueError(f"Spatial dimension must be 1 or 2, got {self.spatial_dim}")
        if len(self.resolution) != self.spatial_dim:
            raise ValueError(
                f"Resolution length {len(self.resolution)} does not match spatial dim {self.spatial_dim}"
            )
        if len(self.domain_bounds) != self.spatial_dim:
            raise ValueError("Domain bounds tuple length must equal spatial dimension")
        if len(self.boundary_conditions) != self.spatial_dim:
            raise ValueError("Boundary conditions length must equal spatial dimension")
        if self.time_step <= 0.0:
            raise ValueError(f"Time step dt must be strictly positive, got {self.time_step}")
        if self.time_horizon <= self.time_step:
            raise ValueError(
                f"Time horizon {self.time_horizon} must be greater than time step {self.time_step}"
            )

    @property
    def cell_measures(self) -> Tuple[float, ...]:
        """Compute spatial grid spacing dx (and dy)."""
        measures = []
        for (low, high), n in zip(self.domain_bounds, self.resolution):
            if high <= low:
                raise ValueError(f"Invalid domain bounds: high {high} <= low {low}")
            if n <= 0:
                raise ValueError(f"Resolution must be positive, got {n}")
            measures.append((high - low) / float(n))
        return tuple(measures)

    def cell_centers(self) -> List[np.ndarray]:
        """Compute 1D coordinate arrays for cell centers."""
        coords = []
        for (low, high), n in zip(self.domain_bounds, self.resolution):
            dx = (high - low) / float(n)
            coords.append(np.linspace(low + 0.5 * dx, high - 0.5 * dx, n))
        return coords

    def cell_faces(self) -> List[np.ndarray]:
        """Compute 1D coordinate arrays for cell faces."""
        coords = []
        for (low, high), n in zip(self.domain_bounds, self.resolution):
            coords.append(np.linspace(low, high, n + 1))
        return coords

    @property
    def total_steps(self) -> int:
        """Total temporal discrete steps in horizon."""
        return int(round(self.time_horizon / self.time_step))

@dataclass
class TrajectoryProvenance:
    """Provenance tracking for a generated or acquired physical trajectory."""
    trajectory_id: str
    family: PhysicalFamily
    generator_name: str
    generator_version: str
    solver_accuracy_order: int
    seed: int
    parameters: Dict[str, float]
    creation_timestamp: str
    float_precision: str = "float64"

@dataclass
class TrajectoryBatch:
    """Strictly typed container for trajectories."""
    family: PhysicalFamily
    grid: GridContract
    field_names: Tuple[str, ...]
    # Tensor shape: [batch, time, x] for 1D or [batch, time, y, x, channels] for 2D
    values: torch.Tensor
    times: torch.Tensor
    provenance: List[TrajectoryProvenance]
    source_terms: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Execute strict semantic and numerical invariants."""
        if not torch.is_tensor(self.values):
            raise TypeError("Trajectory values must be a torch.Tensor")
        if not torch.is_tensor(self.times):
            raise TypeError("Trajectory times must be a torch.Tensor")

        if not torch.all(torch.isfinite(self.values)):
            raise ValueError("Trajectory contains NaN or Inf values")
        if not torch.all(torch.isfinite(self.times)):
            raise ValueError("Trajectory times contain NaN or Inf values")

        if self.times.ndim != 1:
            raise ValueError(f"Times tensor must be 1D, got ndim {self.times.ndim}")
        
        batch_size = self.values.shape[0]
        n_times = self.values.shape[1]

        if len(self.provenance) != batch_size:
            raise ValueError(
                f"Provenance length {len(self.provenance)} does not match batch size {batch_size}"
            )
        if len(self.times) != n_times:
            raise ValueError(
                f"Times length {len(self.times)} does not match temporal dimension {n_times}"
            )

        # Monotonic time check
        time_diffs = torch.diff(self.times)
        if torch.any(time_diffs <= 0.0):
            raise ValueError("Time coordinates must be strictly monotonically increasing")

        # Dimensionality check
        if self.grid.spatial_dim == 1:
            # Expected shape: [batch, time, nx, n_channels] or [batch, time, nx]
            if self.values.ndim == 3:
                nx = self.values.shape[2]
                if nx != self.grid.resolution[0]:
                    raise ValueError(f"Expected spatial resolution {self.grid.resolution[0]}, got {nx}")
                if len(self.field_names) != 1:
                    raise ValueError(f"Expected 1 field name for 3D tensor, got {len(self.field_names)}")
            elif self.values.ndim == 4:
                nx = self.values.shape[2]
                nc = self.values.shape[3]
                if nx != self.grid.resolution[0]:
                    raise ValueError(f"Expected spatial resolution {self.grid.resolution[0]}, got {nx}")
                if nc != len(self.field_names):
                    raise ValueError(f"Channel dimension {nc} != field names {len(self.field_names)}")
            else:
                raise ValueError(f"Invalid tensor rank for 1D spatial system: {self.values.ndim}")
        elif self.grid.spatial_dim == 2:
            # Expected shape: [batch, time, ny, nx, n_channels]
            if self.values.ndim != 5:
                raise ValueError(f"Expected 5D tensor for 2D spatial system, got rank {self.values.ndim}")
            ny, nx, nc = self.values.shape[2], self.values.shape[3], self.values.shape[4]
            if (ny, nx) != self.grid.resolution:
                raise ValueError(f"Expected spatial resolution {self.grid.resolution}, got {(ny, nx)}")
            if nc != len(self.field_names):
                raise ValueError(f"Channel count {nc} does not match fields {len(self.field_names)}")

        # Validate field physical bounds
        catalog = self._get_field_catalog()
        for idx, fname in enumerate(self.field_names):
            if fname in catalog:
                meta = catalog[fname]
                if self.grid.spatial_dim == 1:
                    field_slice = self.values[..., idx] if self.values.ndim == 4 else self.values
                else:
                    field_slice = self.values[..., idx]
                meta.validate_bounds(field_slice)

    def _get_field_catalog(self) -> Dict[str, FieldMetadata]:
        if self.family == PhysicalFamily.BURGERS:
            return BURGERS_FIELDS
        elif self.family == PhysicalFamily.SHALLOW_WATER:
            return SHALLOW_WATER_FIELDS
        elif self.family == PhysicalFamily.GRAY_SCOTT:
            return GRAY_SCOTT_FIELDS
        return {}

    def extract_field(self, field_name: str) -> torch.Tensor:
        """Extract a single field tensor by name."""
        if field_name not in self.field_names:
            raise KeyError(f"Field '{field_name}' not in batch fields {self.field_names}")
        idx = self.field_names.index(field_name)
        if self.grid.spatial_dim == 1 and self.values.ndim == 3:
            return self.values
        return self.values[..., idx]

    def to(self, device: Union[str, torch.device]) -> TrajectoryBatch:
        """Move batch tensors to target device."""
        return TrajectoryBatch(
            family=self.family,
            grid=self.grid,
            field_names=self.field_names,
            values=self.values.to(device),
            times=self.times.to(device),
            provenance=self.provenance,
            source_terms=self.source_terms.to(device) if self.source_terms is not None else None,
        )
