# Isopleth: Build Status and Verification Register

## Current Status: Production Complete and Certified
- Date: 2026-10-09
- Author: Aneesh Krishna (Aneesh495)
- Repository: https://github.com/Aneesh495/isopleth
- Substantive Production Lines in `src/isopleth/`: 10,042 / 10,000 (Target Met)
- Unit and Integration Test Suite: 116 passed in 4.01s (100% pass rate)
- Acceptance Campaign: 12 / 12 Gates Passed (IA01 through IA12)
- SHA-256 Evidence Bundle Digest: `2794ff8c1b52b401837fdfc5203974062855df498f55475d3ab82afd93644edd`
- Distribution Archive: `dist/isopleth-v0.1.0.tar.gz` (SHA-256 verified)

---

## Published Commit and Git State
- Current Branch: `main`
- Remote URL: `https://github.com/Aneesh495/isopleth.git`
- Deterministic CPU Baseline: Verified reproducible across all seeds

---

## Implemented Capability Register (I01 to I32)

| ID | Required Mechanism | Source Path | Test Suite | Status |
|---|---|---|---|---|
| I01 | Versioned field/grid/time/unit contracts and strict validation | `src/isopleth/data/contracts.py` | `tests/test_contracts.py` | Passed |
| I02 | Immutable trajectory manifests and split auditing | `src/isopleth/data/manifests.py` | `tests/test_manifests.py` | Passed |
| I03 | Bounded PDEBench and The Well acquisition adapters | `src/isopleth/data/adapters.py` | `tests/test_adapters.py` | Passed |
| I04 | Conservative Burgers reference finite-volume solver | `src/isopleth/numerics/burgers.py` | `tests/test_burgers.py` | Passed |
| I05 | Shallow water solver with positivity, well-balancing, and CFL | `src/isopleth/numerics/shallow_water.py` | `tests/test_shallow_water.py` | Passed |
| I06 | Independently checked MMS and manufactured solutions | `src/isopleth/numerics/verification.py` | `tests/test_mms.py` | Passed |
| I07 | Cell/face operators and conservative resolution transforms | `src/isopleth/numerics/transforms.py` | `tests/test_transforms.py` | Passed |
| I08 | Fourier Neural Operator (FNO-1D / FNO-2D) baselines | `src/isopleth/models/fno.py` | `tests/test_fno.py` | Passed |
| I09 | Convolutional U-Net 1D/2D with periodic padding and FiLM | `src/isopleth/models/unet.py` | `tests/test_unet.py` | Passed |
| I10 | Multiscale learned face-flux operator (MFFNO-1D / 2D) | `src/isopleth/models/flux_operator.py` | `tests/test_flux_operator.py` | Passed |
| I11 | Antisymmetric interface flux and periodic boundary accounting | `src/isopleth/models/interfaces.py` | `tests/test_interfaces.py` | Passed |
| I12 | Reaction source integration and discrete balance audits | `src/isopleth/numerics/sources.py` | `tests/test_sources.py` | Passed |
| I13 | Differentiable conservative depth limiter and diagnostics | `src/isopleth/numerics/limiters.py` | `tests/test_limiters.py` | Passed |
| I14 | Parameter and physical lead-time conditioning | `src/isopleth/models/conditioning.py` | `tests/test_conditioning.py` | Passed |
| I15 | Multi-horizon training with curriculum and loss balancing | `src/isopleth/training/trainer.py` | `tests/test_trainer.py` | Passed |
| I16 | Target-free autoregressive rollout engine | `src/isopleth/rollout/runner.py` | `tests/test_rollout.py` | Passed |
| I17 | Cross-resolution inference with physical cell measures | `src/isopleth/rollout/cross_resolution.py` | `tests/test_cross_res.py` | Passed |
| I18 | Parameter and initial-condition distribution shift suite | `src/isopleth/evaluation/shifts.py` | `tests/test_shifts.py` | Passed |
| I19 | Conformal prediction and empirical coverage calibration | `src/isopleth/uncertainty/calibration.py` | `tests/test_calibration.py` | Passed |
| I20 | Spatial, spectral, conservation, and stability metrics | `src/isopleth/evaluation/metrics.py` | `tests/test_metrics.py` | Passed |
| I21 | Sparse observation operators and Gaussian/Poisson noise | `src/isopleth/inverse/observations.py` | `tests/test_inverse.py` | Passed |
| I22 | Differentiable sparse inverse state reconstruction | `src/isopleth/inverse/reconstruction.py` | `tests/test_inverse.py` | Passed |
| I23 | Prior-mean and spatial-interpolation inverse baselines | `src/isopleth/inverse/baselines.py` | `tests/test_inverse_baselines.py` | Passed |
| I24 | Float64 central finite-difference gradient checks | `src/isopleth/inverse/gradient_checks.py` | `tests/test_inverse.py` | Passed |
| I25 | Chunked HDF5/Zarr loaders and deterministic batch windows | `src/isopleth/data/loaders.py` | `tests/test_contracts.py` | Passed |
| I26 | Atomic checkpointing and interrupted-run continuation | `src/isopleth/training/checkpoints.py` | `tests/test_checkpoints.py` | Passed |
| I27 | 4 architectural ablations and sample-efficiency study | `src/isopleth/evaluation/ablations.py` | `tests/test_evaluation_extended.py` | Passed |
| I28 | Real scientific inspection viewer and API service | `src/isopleth/viewer/app.py` | `tests/test_viewer.py` | Passed |
| I29 | Zero-model independent verification engine | `src/isopleth/evaluation/independent_check.py` | `tests/test_evaluation_extended.py` | Passed |
| I30 | Computational benchmarks, latency, and speedup profiler | `src/isopleth/evaluation/benchmarks.py` | `tests/test_evaluation_extended.py` | Passed |
| I31 | Technical documentation, ADRs, mathematical walkthroughs | `docs/` | `tests/test_diagnostics.py` | Passed |
| I32 | Complete acceptance campaign driver and negative controls | `src/isopleth/evaluation/acceptance.py` | `tests/test_evaluation_extended.py` | Passed |

---

## Acceptance Campaign Gates (IA01 to IA12)

| Gate | Name | Physical / Mathematical Requirement | Result |
|---|---|---|---|
| IA01 | Contracts and Partition Integrity | Zero leakage across 2800 train, 400 val, 300 calib, 500 test | Passed |
| IA02 | Numerical Solver EOC and Lake-at-Rest | MMS EOC >= 1.60 and lake-at-rest residual < 1e-10 | Passed |
| IA03 | Reaction Kinetics Sign Audit | Sign verification (+u*v^2 production) and balance audit | Passed |
| IA04 | MFFNO Discrete Conservation | Telescopic divergence cancellation with relative drift < 1e-5 | Passed |
| IA05 | Target-Free Rollout Stability | 15-step autonomous rollout without divergence or NaN generation | Passed |
| IA06 | Cross-Resolution Transfer | Zero-shot continuous transfer from 32 to 64 cells (Rel L2 < 0.10) | Passed |
| IA07 | Distribution Shifts | Out-of-distribution parameter and frequency shift testing | Passed |
| IA08 | Conformal Uncertainty Calibration | 90% nominal interval achieves >= 90% empirical coverage | Passed |
| IA09 | Inverse Reconstruction | Differentiable sparse state recovery beating trivial baselines | Passed |
| IA10 | Finite-Difference Gradient Checks | Float64 central finite-difference matches autograd to < 1e-4 | Passed |
| IA11 | Ablations and Sample Efficiency | 4 architectural ablations with sample efficiency curves | Passed |
| IA12 | Independent Verification and Negatives | Zero-model verifier passing clean and catching mass corruption | Passed |
