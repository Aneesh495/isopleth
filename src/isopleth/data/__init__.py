"""Data models, contracts, manifests, and loaders for Isopleth."""

from isopleth.data.contracts import (
    BoundaryCondition,
    BURGERS_FIELDS,
    FieldMetadata,
    GRAY_SCOTT_FIELDS,
    GridContract,
    PhysicalFamily,
    Role,
    SHALLOW_WATER_FIELDS,
    TrajectoryBatch,
    TrajectoryProvenance,
)
from isopleth.data.manifests import (
    DatasetManifest,
    ManifestAuditReport,
    TrajectoryRecord,
    create_synthetic_manifest_partition,
)
from isopleth.data.adapters import (
    GRAY_SCOTT_REGIMES,
    GrayScottRegime,
    GrayScottSignAudit,
    PDEBenchAdapter,
    TheWellGrayScottAdapter,
    get_gray_scott_sign_audit,
)
from isopleth.data.loaders import (
    FieldNormalizer,
    NormalizerStats,
    WindowedTrajectoryDataset,
)

__all__ = [
    "BoundaryCondition",
    "BURGERS_FIELDS",
    "DatasetManifest",
    "FieldMetadata",
    "FieldNormalizer",
    "GRAY_SCOTT_FIELDS",
    "GRAY_SCOTT_REGIMES",
    "GrayScottRegime",
    "GrayScottSignAudit",
    "GridContract",
    "ManifestAuditReport",
    "NormalizerStats",
    "PDEBenchAdapter",
    "PhysicalFamily",
    "Role",
    "SHALLOW_WATER_FIELDS",
    "TheWellGrayScottAdapter",
    "TrajectoryBatch",
    "TrajectoryProvenance",
    "TrajectoryRecord",
    "WindowedTrajectoryDataset",
    "create_synthetic_manifest_partition",
    "get_gray_scott_sign_audit",
]
