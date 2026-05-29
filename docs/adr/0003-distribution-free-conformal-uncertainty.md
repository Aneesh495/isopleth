# ADR 0003: Distribution-Free Conformal Uncertainty Quantification

## Status
Accepted

## Context
Quantifying trajectory prediction uncertainty in physical systems is critical for safety and inverse estimation. Existing methods (Bayesian Neural Networks, deep ensembles, MC dropout) require strong Gaussian or parametric assumptions, carry severe compute overhead during multi-step rollouts, and lack formal finite-sample coverage guarantees.

## Decision
We adopt Distribution-Free Split Conformal Prediction and Conformal Risk Control (CRC) as the primary uncertainty quantification framework in Isopleth:
1. A dedicated, held-out calibration partition of 300 trajectories is reserved exclusively for calibration.
2. Nonconformity scores are computed over trajectory rollout errors.
3. Quantiles q_alpha are calibrated to provide rigorous finite-sample coverage guarantees:
   ```
   P( y_{t+h} in C(x) ) >= 1 - alpha
   ```
4. For secondary analytical aleatoric vs epistemic decomposition, we supplement conformal sets with Evidential Deep Learning (Normal-Inverse-Gamma priors) which evaluates uncertainty in a single forward pass without Monte Carlo sampling.

## Consequences
### Positive:
- Distribution-free finite-sample mathematical validity without Gaussianity assumptions.
- Fast evaluation (one quantile lookup per lead time, zero MCMC or ensemble overhead).
- Conformal risk control extends guarantees to arbitrary bounded physical loss metrics.

### Negative:
- Interval widths can be conservative if the calibration partition does not capture out-of-distribution extremes.
