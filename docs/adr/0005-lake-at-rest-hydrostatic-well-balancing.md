# ADR 0005: Lake-at-Rest Hydrostatic Well-Balancing in Shallow Water

## Status
Accepted

## Context
In the 2D Shallow Water equations with variable bathymetry b(x, y), the hydrostatic balance (lake-at-rest equilibrium) corresponds to quiescent water (u = v = 0) where fluid depth h plus bed elevation b is spatially constant:
```
eta = h(x, y) + b(x, y) = const
```
In this steady state, the pressure gradient flux divergence exactly cancels the bathymetry bottom slope source term:
```
d/dx ( 0.5 * g * h^2 ) = - g * h * db/dx
```
If a numerical or neural scheme differentiates flux and evaluates the source term independently without exact algebraic cancellation, truncation errors generate spurious non-physical waves with velocities up to O(sqrt(g*h)), destroying the steady state.

## Decision
We enforce hydrostatic well-balancing in both the numerical solver and neural surrogate models:
1. Surface Elevation Variable: We reformulate the hydrostatic pressure balance in terms of total water surface elevation eta = h + b.
2. Hydrostatic Reconstruction: Bed slope gradients and pressure fluxes are reconstructed using Audusse hydrostatic interface reconstructions.
3. Verification Gate IA02: Every solver and model configuration must achieve lake-at-rest residual < 1e-10 over 30 simulation steps.

## Consequences
### Positive:
- Preserves unperturbed quiescent water bodies indefinitely without artificial wave generation.
- Correctly models small-amplitude perturbations (such as tsunamis) over complex underwater topography.

### Negative:
- Slightly increases interface reconstruction complexity at wet-dry boundaries.
