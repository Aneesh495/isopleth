# ADR 0004: Differentiable Sparse Inverse State Reconstruction

## Status
Accepted

## Context
A major promise of neural operator surrogates is accelerating inverse problems (reconstructing hidden initial states u_0 or physical parameters from sparse sensor measurements). However, when surrogates are used inside iterative gradient descent or MCMC loops:
1. Do autograd gradients match true physical adjoints?
2. Do surrogate forward errors introduce bias or spurious minima in the reconstructed state?

## Decision
We implement a fully differentiable inverse reconstruction engine and couple it with rigorous gradient verification:
1. Differentiable Observation Model: Sparse observation matrices and sensor noise are integrated into the PyTorch computational graph.
2. Adjoint Optimization: Invert state via backpropagation-through-time (BPTT) using Adam, projected gradient descent, and second-order L-BFGS.
3. Gradient Verification: Every inverse setup is verified against double-precision central finite differences with step size epsilon = 1e-6, requiring relative gradient error <= 1e-4.
4. Distortion Diagnostics: Quantify surrogate distortion by comparing surrogate-inverted states against reference numerical solver inversions and Hamiltonian Monte Carlo posterior chains.

## Consequences
### Positive:
- Accelerates sparse inverse reconstruction by 4x to 20x compared to high-resolution numerical PDE solves.
- Finite-difference checks catch vanishing or exploding gradients before running production inversions.
- Provides quantitative diagnostics of surrogate-induced inverse bias.

### Negative:
- Backpropagation through multi-step rollouts requires caching intermediate hidden activations in memory.
