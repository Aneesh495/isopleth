# Isopleth Model Cards

## Model Zoo Overview

Isopleth implements five neural operator families to systematically evaluate physical conservation, cross-resolution scaling, and inverse distortion.

---

## 1. Model Card: Multiscale Face-Flux Neural Operator (MFFNO)

### 1.1 Architecture Summary
- **Model Type**: Conservative learned numerical face-flux operator.
- **Inductive Bias**: Predicts numerical fluxes at staggered cell interfaces rather than state updates; exactly satisfies telescopic divergence cancellation.
- **Components**:
  - Global Branch: 1D/2D Spectral Fourier convolution extracting large-scale waves.
  - Local Branch: Depthwise convolutional stencils (k=3, 5) capturing shock gradients.
  - Conditioning: Feature-wise Linear Modulation (FiLM) for viscosity nu, lead time dt, and Reynolds numbers.
  - Interface Head: Staggered projection mapping latent features to cell face fluxes F_{i+1/2}.
- **Parameter Count**: 111,456 (1D configuration with 32 hidden channels, 8 Fourier modes).

### 1.2 Forward Mapping
```
Input: State u^n in R^(B, C, N), grid spacing dx, time step dt
1. F_{i+1/2} = NeuralFluxHead( u^n, dx, dt )
2. Div_i = - (dt / dx) * ( F_{i+1/2} - F_{i-1/2} )
3. Output: u^{n+1} = u^n + Div_i
```

### 1.3 Physical Invariant Guarantees
- Discrete Mass Conservation: Machine precision (relative drift < 1e-12 under double precision, < 1e-6 under single precision).
- Zero-Divergence Property: Domain integral of flux divergence vanishes identically under periodic boundary conditions.
- Positivity Preservation: Differentiable depth projection guarantees h >= h_min.

### 1.4 Limitations
- Periodic and closed domain formulations are exact; open boundary inflow/outflow requires explicit boundary flux logging.

---

## 2. Model Card: Fourier Neural Operator (FNO Baseline)

### 2.1 Architecture Summary
- **Model Type**: Classical spectral Fourier neural operator (Li et al., 2020).
- **Inductive Bias**: Mesh-independent kernel integral parameterized in Fourier space.
- **Components**:
  - Lifting MLP mapping physical channels to hidden representation.
  - 4 Spectral Convolution Blocks: Real FFT -> truncated frequency multiplier -> Inverse Real FFT + residual linear bypass.
  - Projection MLP mapping latent representation to updated state.
- **Parameter Count**: 140,513 (1D configuration with 32 hidden channels, 8 modes).

### 2.2 Forward Mapping
```
Input: State u^n in R^(B, C, N)
Output: u^{n+1} = Projection( SpectralBlocks( Lifting( u^n ) ) )
```

### 2.3 Physical Invariants
- Direct state update does not enforce spatial flux balance.
- Unconstrained cumulative mass drift: O(1e-2) over 50 autoregressive steps.

---

## 3. Model Card: Convolutional U-Net Baseline

### 3.1 Architecture Summary
- **Model Type**: Multi-scale convolutional encoder-decoder with skip connections (Ronneberger et al., 2015).
- **Inductive Bias**: Hierarchical spatial feature aggregation with localized receptive fields.
- **Components**:
  - Encoder: 2 resolution-halving stages with strided convolutions.
  - Decoder: 2 transposed convolutional upsampling stages with skip concatenations.
  - Boundary Padding: Circular padding ensuring periodic domain continuity.
  - Conditioning: FiLM scale-and-shift modulation in convolutional residual blocks.
- **Parameter Count**: 189,409 (1D configuration with 16 base channels).

### 3.2 Limitations
- Multi-scale spatial pooling breaks continuous cross-resolution transfer when evaluated on unseen grid resolutions without interpolation.

---

## 4. Model Card: Deep Operator Network (DeepONet Baseline)

### 4.1 Architecture Summary
- **Model Type**: Universal operator approximator with dual-network topology (Lu et al., 2021).
- **Inductive Bias**: Bilinear dot product between input function sensor measurements and query coordinates.
- **Components**:
  - Branch Network: MLP processing m discrete sensor measurements u(x_j).
  - Trunk Network: MLP evaluating continuous coordinate basis functions phi_k(y).
  - Output: Dot product sum_k b_k(u) * phi_k(y) + bias.
- **Parameter Count**: 91,137 (Branch MLP 64->128->64, Trunk MLP 1->128->64).

### 4.2 Conservative Variant
Isopleth provides `ConservativeFluxDeepONet1D`, where the Branch and Trunk networks output interface flux representations F_{i+1/2}, restoring exact telescopic discrete conservation to the DeepONet formulation.

---

## 5. Model Card: Wavelet Neural Operator (WNO Baseline)

### 5.1 Architecture Summary
- **Model Type**: Multi-resolution wavelet neural operator (Tripura & Chakraborty, 2023).
- **Inductive Bias**: Compactly supported spatiotemporal localization via Discrete Wavelet Transform (DWT).
- **Components**:
  - 1D/2D DWT Decomposition: Haar and Daubechies D4 wavelet filters.
  - Wavelet Domain Conv1d/Conv2d transforming detail and approximation coefficients.
  - Inverse DWT (IDWT) reconstructing spatial fields.
  - Staggered Conservative Face-Flux head (optional).
- **Parameter Count**: 13,857 (compact, highly parameter-efficient).

### 5.2 Strengths and Limitations
- Excellent shock localization without global Fourier spectral ringing (Gibbs phenomenon).
- Fast O(N) wavelet transform complexity.
