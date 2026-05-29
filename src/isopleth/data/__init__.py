"""Data models, contracts, manifests, loaders, and generators for Isopleth."""

from isopleth.data.adapters import (
    GRAY_SCOTT_REGIMES,
    GrayScottRegime,
    GrayScottSignAudit,
    PDEBenchAdapter,
    TheWellGrayScottAdapter,
    get_gray_scott_sign_audit,
)
from isopleth.data.contracts import (
    BURGERS_FIELDS,
    GRAY_SCOTT_FIELDS,
    SHALLOW_WATER_FIELDS,
    BoundaryCondition,
    FieldMetadata,
    GridContract,
    PhysicalFamily,
    Role,
    TrajectoryBatch,
    TrajectoryProvenance,
)
from isopleth.data.generators import (
    GeneratorJobConfig,
    PartitionBatchOrchestrator,
    TrajectoryGenerator,
)
from isopleth.data.loaders import (
    FieldNormalizer,
    NormalizerStats,
    WindowedTrajectoryDataset,
)
from isopleth.data.manifests import (
    DatasetManifest,
    ManifestAuditReport,
    TrajectoryRecord,
    create_synthetic_manifest_partition,
)
from isopleth.data.samplers import (
    GaussianRandomField1D,
    GaussianRandomField2D,
    GrayScottSeedSampler,
    KarhunenLoeveExpander1D,
    RiemannDamBreakSampler,
    SamplerMetadata,
    ShallowWaterTopographySampler,
    sample_initial_conditions_torch,
)

__all__ = [
    "BoundaryCondition",
    "BURGERS_FIELDS",
    "DatasetManifest",
    "FieldMetadata",
    "FieldNormalizer",
    "GRAY_SCOTT_FIELDS",
    "GRAY_SCOTT_REGIMES",
    "GaussianRandomField1D",
    "GaussianRandomField2D",
    "GeneratorJobConfig",
    "GrayScottRegime",
    "GrayScottSeedSampler",
    "GrayScottSignAudit",
    "GridContract",
    "KarhunenLoeveExpander1D",
    "ManifestAuditReport",
    "NormalizerStats",
    "PDEBenchAdapter",
    "PartitionBatchOrchestrator",
    "PhysicalFamily",
    "RiemannDamBreakSampler",
    "Role",
    "SHALLOW_WATER_FIELDS",
    "SamplerMetadata",
    "ShallowWaterTopographySampler",
    "TheWellGrayScottAdapter",
    "TrajectoryBatch",
    "TrajectoryGenerator",
    "TrajectoryProvenance",
    "TrajectoryRecord",
    "WindowedTrajectoryDataset",
    "create_synthetic_manifest_partition",
    "get_gray_scott_sign_audit",
    "sample_initial_conditions_torch",
]
