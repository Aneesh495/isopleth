# ADR 0002: Staggered Interface Flux Heads and Antisymmetric Coupling

## Status
Accepted

## Context
When computing spatial flux divergence, numerical finite-volume schemes evaluate fluxes across cell boundaries (faces), whereas physical conserved variables (mass, momentum, concentrations) reside at cell centroids. If flux values are collocated at cell centers and differentiated via naive central differences, numerical odd-even decoupling and checkerboard pressure oscillations occur.

## Decision
We implement dedicated staggered interface projection heads:
1. In 1D: The head maps latent cell features to face fluxes F_{i+1/2} located midway between cell i and cell i+1.
2. In 2D: The head outputs two separate staggered flux tensors:
   - F_x located at vertical interfaces (i+1/2, j).
   - F_y located at horizontal interfaces (i, j+1/2).
3. Antisymmetric Coupling: Interface flux evaluations between adjacent cells satisfy anti-symmetry under coordinate reflection, preventing artificial directional bias in wave propagation.

## Consequences
### Positive:
- Eliminates checkerboard grid decoupling.
- Discrete divergence matches high-order finite-volume numerical formulations exactly.
- Natural alignment with staggered Arakawa C-grid hydrodynamics.

### Negative:
- 2D architectures require separate horizontal and vertical face projection channels.
