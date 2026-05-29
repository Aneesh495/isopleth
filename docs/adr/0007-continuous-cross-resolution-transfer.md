# ADR 0007: Continuous Cross-Resolution Transfer and Scaling

## Status
Accepted

## Context
Standard neural network architectures (such as CNNs and U-Nets) are tied to a fixed spatial discretization: a model trained on a 64-element grid cannot process a 128-element grid without ad-hoc bicubic interpolation. Neural operators claim continuous mesh independence, but naive implementations fail when spatial derivatives or interface fluxes are not scaled by physical grid cell widths dx and dt.

## Decision
We implement continuous cross-resolution evaluation using physical cell measures:
1. Continuous Coordinate Channels: Operators receive continuous spatial coordinates x in [0, 1) rather than discrete cell indices.
2. Frequency Padding in Spectral Layers: FNO and MFFNO evaluate Fourier modes up to cutoff mode k_max, padding higher frequencies with zeros when evaluating on fine grids.
3. Explicit Cell Measure Scaling: Interface fluxes and discrete divergences are scaled explicitly by target grid metrics dx_target and dt_target:
   ```
   u^{n+1} = u^n - (dt_target / dx_target) * div( F_target )
   ```
4. Zero-Shot Verification Gate IA06: Models trained on Nx = 32 must evaluate zero-shot on Nx = 64 with relative L2 error < 0.10 without any fine-tuning.

## Consequences
### Positive:
- Operators can be trained cheaply on coarse grids and deployed directly on fine grids.
- Preserves physical wave propagation speeds regardless of evaluation grid resolution.

### Negative:
- High-frequency details beyond the training cutoff mode are not resolved unless supplemented by local convolutional stencils.
