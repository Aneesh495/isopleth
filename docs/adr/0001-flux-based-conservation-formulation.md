# ADR 0001: Flux-Based Conservation Formulation for Neural Operators

## Status
Accepted

## Context
Standard neural operator frameworks (Fourier Neural Operators, U-Nets, and DeepONets) directly parameterize the next-step solution operator:
```
u^{n+1} = G_theta( u^n )
```
When evaluated over multi-step autoregressive rollouts, small per-step prediction errors accumulate. In conservation laws governed by divergence forms:
```
du/dt + div( F(u) ) = S(u)
```
direct state updating routinely violates physical invariants: total fluid mass drifts, momentum decays or blows up unphysically, and shocks develop severe Gibbs oscillations.

## Decision
We formulate neural operators in the space of spatial interface fluxes rather than state updates:
```
F_{i+1/2} = N_theta( u^n, dx, dt )
u^{n+1}_i = u^n_i - (dt / dx) * ( F_{i+1/2} - F_{i-1/2} ) + dt * S_i
```
By enforcing that interface flux F_{i+1/2} is shared between cell i and cell i+1, the spatial sum over any periodic or closed domain telescopically cancels identically to machine precision.

## Consequences
### Positive:
- Machine-precision mass conservation holds for any network weights, layers, and activations.
- Long-horizon autoregressive rollouts remain stable without exponential mass growth or dissipation.
- Physical flux quantities are interpretable and inspectable.

### Negative:
- The operator must output staggered face quantities (dimension N for 1D, dimension (Nx, Ny) for 2D horizontal/vertical faces), requiring dedicated interface head layers.
- Open boundary conditions require explicit boundary flux accounting.
