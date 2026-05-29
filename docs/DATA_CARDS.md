# Isopleth Dataset Cards

## Dataset Overview

The Isopleth laboratory maintains three canonical physical dataset collections spanning hyperbolic, parabolic, and reaction-diffusion conservation laws. Every dataset follows strict zero-leakage partitions (2800 train, 400 validation, 300 calibration, 500 test) with cryptographic SHA-256 binary verification.

---

## 1. Dataset Card: 1D Viscous Periodic Burgers Flow

### 1.1 Dataset Summary
- **Physical Family**: Non-linear hyperbolic advection with viscous diffusion.
- **Governing Equation**: `du/dt + d/dx (0.5 * u^2) = nu * d^2 u / dx^2`
- **Spatial Domain**: Unit periodic domain x in [0, 1), periodic boundary conditions.
- **Temporal Horizon**: t in [0.0, 1.0], discretized with dt = 0.01 (100 steps per trajectory).
- **Physical Invariants**: Total mass M = integral(u dx) is invariant over time.
- **Energy Accounting**: Kinetic energy E = 0.5 * integral(u^2 dx) dissipates monotonically.

### 1.2 Initial Condition Generation
Initial states are generated via two complementary stochastic field samplers:
1. **Gaussian Random Field (GRF)**: Matérn covariance kernel with smoothness parameter nu_matern in {1.5, 2.5} and correlation length l_corr in [0.1, 0.3].
2. **Karhunen-Loeve Expansion**: Truncated modal expansion with decaying spectral amplitudes `c_k \sim k^{-alpha}` for alpha in [1.5, 3.0].

### 1.3 Solver Parameters and Verification
- **Discretization**: Finite-volume method with Godunov interface numerical flux.
- **Reconstruction**: Second-order MUSCL with van Leer slope limiter.
- **Time Integration**: Strong Stability Preserving Runge-Kutta 2 (SSP-RK2).
- **Convergence**: Verified by Method of Manufactured Solutions (MMS) achieving EOC = 2.16 (second-order).

### 1.4 Data Partitioning and Splits
- Total Trajectories: 4,000
- Train Split: 2,800 trajectories (70%)
- Validation Split: 400 trajectories (10%)
- Calibration Split: 300 trajectories (7.5%)
- Held-Out Test Split: 500 trajectories (12.5%)
- Storage Format: Chunked HDF5 (`compression="gzip"`) and Zarr v3 (`zstd` codec).

---

## 2. Dataset Card: 2D Shallow Water System (Saint-Venant)

### 2.1 Dataset Summary
- **Physical Family**: Non-linear hyperbolic conservation law with geometric source terms.
- **Governing Equations**:
  - Mass conservation: `dh/dt + d/dx (h * u) + d/dy (h * v) = 0`
  - Zonal momentum: `d(h*u)/dt + d/dx (h*u^2 + 0.5*g*h^2) + d/dy (h*u*v) = - g * h * db/dx`
  - Meridional momentum: `d(h*v)/dt + d/dx (h*u*v) + d/dy (h*v^2 + 0.5*g*h^2) = - g * h * db/dy`
- **Spatial Domain**: Square domain [0, 1) x [0, 1) with periodic or reflective boundary conditions.
- **Physical Invariants**: Total water volume V = integral(h dx dy) is strictly conserved.
- **Physical Invariants**: Total surface elevation eta = h + b satisfies hydrostatic balance.

### 2.2 Initial Condition Generation and Bathymetry
1. **Circular Dambreak Configurations**: Discontinuous cylinder of elevated fluid collapsing under gravity.
2. **Random Gaussian Mounds**: Multiple interacting fluid mounds generating complex wave interference.
3. **Synthetic Bathymetry**: Gaussian seamounts and undulating cosine sea floor topographies.

### 2.3 Numerical Verification
- Positivity Preservation: Differentiable depth limiter enforcing h >= 1e-6.
- Lake-at-Rest Well-Balancing: Hydrostatic equilibrium verified to residual < 1e-10.
- CFL Stability: Adaptive sub-cycling partitioning macro-steps into sub-steps satisfying advective-wave speed CFL <= 0.45.

### 2.4 Data Partitioning
- Total Trajectories: 4,000
- Train Split: 2,800 trajectories
- Validation Split: 400 trajectories
- Calibration Split: 300 trajectories
- Test Split: 500 trajectories
- Storage: Chunked HDF5 / Zarr arrays with shape (4000, 50, 3, 32, 32).

---

## 3. Dataset Card: 2D Gray-Scott Reaction-Diffusion

### 3.1 Dataset Summary
- **Physical Family**: Stiff parabolic reaction-diffusion system with non-linear cubic kinetics.
- **Governing Equations**:
  - `du/dt = D_u * laplacian(u) - u * v^2 + F * (1 - u)`
  - `dv/dt = D_v * laplacian(v) + u * v^2 - (F + k) * v`
- **Spatial Domain**: Periodic square [0, 1) x [0, 1).
- **Physical Constraints**: Concentrations u, v constrained to [0, 1].

### 3.2 Parameter Regimes (Pearson Classification)
Datasets span 6 canonical morphological regimes:
- **Solitons (alpha)**: Feed F = 0.030, Kill k = 0.062 (localized traveling spots).
- **Moving Spots (beta)**: Feed F = 0.014, Kill k = 0.054 (self-replicating spots).
- **Spirals (gamma)**: Feed F = 0.018, Kill k = 0.051 (rotating scroll waves).
- **Stripes / Worms (delta)**: Feed F = 0.034, Kill k = 0.065 (labyrinthine patterns).
- **Chaos (epsilon)**: Feed F = 0.026, Kill k = 0.055 (spatiotemporal turbulence).
- **U-Skate (zeta)**: Feed F = 0.062, Kill k = 0.061 (transient wave rings).

### 3.3 Balance Audit
Discrete balance equations are audited with the `BalanceAuditor` engine verifying that net species changes equal integrated diffusion flux plus chemical reaction sources to machine precision.
