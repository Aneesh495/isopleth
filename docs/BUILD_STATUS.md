# Isopleth: Build Status and Verification Register

## Current Phase: Phase 1 (Repository, Toolchain, and Contracts Initialization)
- Date: 2026-10-09
- Author: Aneesh Krishna (Aneesh495)
- Repository: https://github.com/Aneesh495/isopleth

---

## Published Commit and Git State
- Current Branch: `main`
- Initial Commit: Pending initial push
- Hosted Remote Tip: Pending creation of `Aneesh495/isopleth`

---

## Implemented Capability Register (I01 - I32)

| ID | Required Mechanism | Source Path | Unit / Semantic Test | Status |
|---|---|---|---|---|
| I01 | Versioned field/grid/time/unit contracts and strict validation | `src/isopleth/data/contracts.py` | `tests/test_contracts.py` | passed |
| I02 | Immutable trajectory manifests and split auditing | `src/isopleth/data/manifests.py` | `tests/test_manifests.py` | passed |
| I03 | Bounded PDEBench and The Well acquisition adapters | `src/isopleth/data/adapters.py` | `tests/test_adapters.py` | passed |
| I04 | Original conservative Burgers reference solver | `src/isopleth/numerics/burgers.py` | `tests/test_burgers.py` | pending |
| I05 | Original shallow-water solver with positivity/CFL | `src/isopleth/numerics/shallow_water.py` | `tests/test_shallow_water.py` | pending |
| I06 | Independently checked manufactured and analytic solutions | `src/isopleth/numerics/verification.py` | `tests/test_mms.py` | pending |
| I07 | Cell/face operators and conservative resolution transforms | `src/isopleth/numerics/transforms.py` | `tests/test_transforms.py` | pending |
| I08 | Original Fourier neural operator baseline | `src/isopleth/models/fno.py` | `tests/test_fno.py` | pending |
| I09 | Original convolutional U-Net baseline | `src/isopleth/models/unet.py` | `tests/test_unet.py` | pending |
| I10 | Original multiscale learned face-flux operator | `src/isopleth/models/flux_operator.py` | `tests/test_flux_operator.py` | pending |
| I11 | Antisymmetric/shared interface flux and boundary accounting | `src/isopleth/models/interfaces.py` | `tests/test_interfaces.py` | pending |
| I12 | Explicit reaction/source integration and balance audits | `src/isopleth/numerics/sources.py` | `tests/test_sources.py` | pending |
| I13 | Differentiable conservative depth limiter and diagnostics | `src/isopleth/numerics/limiters.py` | `tests/test_limiters.py` | pending |
| I14 | Parameter and physical lead-time conditioning | `src/isopleth/models/conditioning.py` | `tests/test_conditioning.py` | pending |
| I15 | Real original multi-horizon training with validation selection | `src/isopleth/training/trainer.py` | `tests/test_trainer.py` | pending |
| I16 | Autoregressive rollout without target feedback | `src/isopleth/rollout/runner.py` | `tests/test_rollout.py` | pending |
| I17 | Cross-resolution inference with physical grid measures | `src/isopleth/rollout/cross_resolution.py` | `tests/test_cross_res.py` | pending |
| I18 | Parameter and initial-condition shift experiments | `src/isopleth/evaluation/shifts.py` | `tests/test_shifts.py` | pending |
| I19 | Trajectory ensembles and held-out calibration | `src/isopleth/uncertainty/calibration.py` | `tests/test_calibration.py` | pending |
| I20 | Spatial/spectral/conservation/stability metrics | `src/isopleth/evaluation/metrics.py` | `tests/test_metrics.py` | pending |
| I21 | Sparse observation operators and noise contracts | `src/isopleth/inverse/observations.py` | `tests/test_observations.py` | pending |
| I22 | Original differentiable inverse reconstruction | `src/isopleth/inverse/reconstruction.py` | `tests/test_inverse.py` | pending |
| I23 | Solver-based inverse and trivial-prior baselines | `src/isopleth/inverse/baselines.py` | `tests/test_inverse_baselines.py` | pending |
| I24 | Forward/adjoint gradient checks and bias diagnostics | `src/isopleth/inverse/gradient_checks.py` | `tests/test_gradients.py` | pending |
| I25 | Chunked loaders, bounded memory, and deterministic windows | `src/isopleth/data/loaders.py` | `tests/test_loaders.py` | pending |
| I26 | Atomic checkpoints and interrupted-run continuation | `src/isopleth/training/checkpoints.py` | `tests/test_checkpoints.py` | pending |
| I27 | Fixed ablations and sample-efficiency experiments | `src/isopleth/evaluation/ablations.py` | `tests/test_ablations.py` | pending |
| I28 | Genuine field/trajectory/inverse scientific viewer | `src/isopleth/viewer/app.py` | `tests/test_viewer.py` | pending |
| I29 | Independent metric recomputation and physical accounting | `src/isopleth/evaluation/independent_check.py` | `tests/test_accounting.py` | pending |
| I30 | Raw resource, throughput, and matched-accuracy measurements | `src/isopleth/evaluation/benchmarks.py` | `tests/test_benchmarks.py` | pending |
| I31 | Mathematical docs, source walkthroughs, ADRs, diagrams | `docs/` | `tests/test_docs.py` | pending |
| I32 | Source-bound acceptance and executed verifier negative controls | `src/isopleth/evaluation/acceptance.py` | `tests/test_acceptance.py` | pending |

---

## Acceptance Campaign Gates (IA01 - IA12)

| Gate | Description | Target Contract | Status |
|---|---|---|---|
| IA01 | Mathematical/semantic property tests | >= 400 authored, >= 10,000 property cases | pending |
| IA02 | Numerical solver convergence and conservation | 100 MMS/analytic cases per family, 3 grid levels | pending |
| IA03 | Dataset manifest and role validation | 8,000 generated trajectories, 128 PDEBench, 64 Gray-Scott | pending |
| IA04 | Primary operator, FNO, U-Net training | 3 seeds per family, retained loss histories | pending |
| IA05 | Rollout accuracy vs persistence | >= 15% median improvement over persistence | pending |
| IA06 | Held-out long-horizon rollout | 500 trajectories per family evaluated at 20/50/100 steps | pending |
| IA07 | Cross-resolution transfer and shift evaluations | 200 cross-grid cases, 200 parameter shift cases | pending |
| IA08 | Ablation matrix and sample efficiency | 4 ablations, 100/500/2800 training trajectories | pending |
| IA09 | Sparse inverse reconstruction | 100 inverse cases, 4 sensor/noise profiles | pending |
| IA10 | Ensemble calibration and conformal coverage | 3-seed ensemble coverage and width diagnostics | pending |
| IA11 | Fault injection, interruption, and memory stress | 50 interruption cases, 100 loader failures, 30m stress run | pending |
| IA12 | Acceptance bundle and 12 negative controls | Verifier passes bundle; 12 negative controls fail predictably | pending |

---

## Next Executable Action
Initialize Git repository, configure remote `Aneesh495/isopleth`, commit baseline infrastructure, push `main`, and begin Phase 1 core data contracts and numerical solver implementation.
