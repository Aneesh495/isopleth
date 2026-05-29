# ADR 0006: Immutable Zero-Leakage Dataset Manifests

## Status
Accepted

## Context
In scientific machine learning, subtle evaluation data leakage is pervasive: identical or correlated initial condition random seeds appear across training and test splits, multi-step autoregressive rollouts leak future test frames into training sequences, and unversioned HDF5/Zarr files lead to irreproducible benchmark claims.

## Decision
We enforce an immutable dataset manifest architecture:
1. Four-Way Canonical Partition: Total 4,000 trajectories partitioned into:
   - Train: 2,800 trajectories (70%)
   - Validation: 400 trajectories (10%)
   - Calibration: 300 trajectories (7.5%)
   - Held-Out Test: 500 trajectories (12.5%)
2. Cryptographic Binary Hash Auditing: Every trajectory record contains a deterministic UUIDv5 identifier and a SHA-256 hash of its complete binary floating-point payload.
3. Verification Gate IA01: The build process audits the set intersection of hashes across all partitions. If the intersection is non-empty, execution halts immediately with a data contamination error.
4. Immutable Manifest Artifacts: Manifest files are saved as frozen JSON artifacts with a master SHA-256 bundle digest.

## Consequences
### Positive:
- Guarantees zero data leakage across train, validation, calibration, and test partitions.
- Any accidental file modification or re-seeding produces a hash mismatch and fails builds immediately.
- Enables reproducible benchmarks across independent computing environments.

### Negative:
- Manifest generation must occur prior to training runs.
