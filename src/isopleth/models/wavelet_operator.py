"""Wavelet Neural Operator (WNO) baseline models.

Implements multi-resolution Wavelet Neural Operators using 1D and 2D
Discrete Wavelet Transforms (Haar and Daubechies D4 filters). WNO operates
in multi-scale wavelet coefficient space, providing spatial-frequency
localization beneficial for sharp shock fronts and localized gradients.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from isopleth.models.interfaces import (
    DiscreteFluxDivergence,
    InterfaceFluxHead1D,
    InterfaceFluxHead2D,
)


def get_haar_filters_1d() -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns 1D Haar scaling (lowpass) and wavelet (highpass) filters."""
    # Lowpass: [1/sqrt(2), 1/sqrt(2)], Highpass: [-1/sqrt(2), 1/sqrt(2)]
    s = 1.0 / math.sqrt(2.0)
    lo = torch.tensor([s, s], dtype=torch.float32)
    hi = torch.tensor([-s, s], dtype=torch.float32)
    return lo, hi


def get_daub4_filters_1d() -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns 1D Daubechies D4 scaling and wavelet filters."""
    sqrt3 = math.sqrt(3.0)
    denom = 4.0 * math.sqrt(2.0)
    h0 = (1.0 + sqrt3) / denom
    h1 = (3.0 + sqrt3) / denom
    h2 = (3.0 - sqrt3) / denom
    h3 = (1.0 - sqrt3) / denom
    lo = torch.tensor([h0, h1, h2, h3], dtype=torch.float32)
    hi = torch.tensor([h3, -h2, h1, -h0], dtype=torch.float32)
    return lo, hi


class DWT1D(nn.Module):
    """1D Discrete Wavelet Transform layer with circular padding."""

    def __init__(self, wavelet: str = "haar") -> None:
        super().__init__()
        self.wavelet = wavelet.lower()
        if self.wavelet == "haar":
            lo, hi = get_haar_filters_1d()
        elif self.wavelet in ("daub4", "db4", "d4"):
            lo, hi = get_daub4_filters_1d()
        else:
            raise ValueError(f"Unsupported wavelet: {wavelet}")

        # Shape for depthwise conv1d: (1, 1, filter_len)
        self.register_buffer("lo_filter", lo.view(1, 1, -1))
        self.register_buffer("hi_filter", hi.view(1, 1, -1))
        self.filter_len = lo.shape[0]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform single-level 1D DWT.

        Args:
            x: Input tensor, shape (B, C, N).

        Returns:
            Tuple of (approx, detail) each of shape (B, C, N // 2).
        """
        batch_size, channels, n = x.shape
        pad = self.filter_len - 1

        # Circular padding for periodic conservation laws
        x_pad = F.pad(x, (0, pad), mode="circular")

        # Repeat filters for grouped convolution across channels
        lo_weight = self.lo_filter.repeat(channels, 1, 1).to(
            device=x.device, dtype=x.dtype
        )
        hi_weight = self.hi_filter.repeat(channels, 1, 1).to(
            device=x.device, dtype=x.dtype
        )

        approx = F.conv1d(x_pad, lo_weight, stride=2, groups=channels)
        detail = F.conv1d(x_pad, hi_weight, stride=2, groups=channels)
        return approx, detail


class IDWT1D(nn.Module):
    """1D Inverse Discrete Wavelet Transform layer."""

    def __init__(self, wavelet: str = "haar") -> None:
        super().__init__()
        self.wavelet = wavelet.lower()
        if self.wavelet == "haar":
            lo, hi = get_haar_filters_1d()
        elif self.wavelet in ("daub4", "db4", "d4"):
            lo, hi = get_daub4_filters_1d()
        else:
            raise ValueError(f"Unsupported wavelet: {wavelet}")

        self.register_buffer("lo_filter", lo.view(1, 1, -1))
        self.register_buffer("hi_filter", hi.view(1, 1, -1))
        self.filter_len = lo.shape[0]

    def forward(self, approx: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        """Perform single-level 1D inverse DWT.

        Args:
            approx: Low-frequency approximation, shape (B, C, N // 2).
            detail: High-frequency detail, shape (B, C, N // 2).

        Returns:
            Reconstructed tensor of shape (B, C, N).
        """
        batch_size, channels, half_n = approx.shape
        lo_weight = self.lo_filter.repeat(channels, 1, 1).to(
            device=approx.device, dtype=approx.dtype
        )
        hi_weight = self.hi_filter.repeat(channels, 1, 1).to(
            device=detail.device, dtype=detail.dtype
        )

        # Transposed conv1d upsamples by factor 2
        recon_lo = F.conv_transpose1d(approx, lo_weight, stride=2, groups=channels)
        recon_hi = F.conv_transpose1d(detail, hi_weight, stride=2, groups=channels)

        pad = self.filter_len - 1
        recon = recon_lo + recon_hi
        # Trim circular padding effect if filter_len > 2
        if pad > 0 and recon.shape[-1] > 2 * half_n:
            recon = recon[..., : 2 * half_n]
        return recon


class WaveletConv1D(nn.Module):
    """Multi-scale wavelet convolution block in 1D."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wavelet: str = "haar",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dwt = DWT1D(wavelet=wavelet)
        self.idwt = IDWT1D(wavelet=wavelet)

        # Learned linear transforms in approximation and detail spaces
        self.approx_weight = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.detail_weight = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # Physical space residual bypass
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of 1D wavelet convolution.

        Args:
            x: Input tensor, shape (B, C_in, N).

        Returns:
            Output tensor, shape (B, C_out, N).
        """
        # Multi-scale decomposition
        approx, detail = self.dwt(x)

        # Subband linear transformations
        approx_out = self.approx_weight(approx)
        detail_out = self.detail_weight(detail)

        # Wavelet synthesis
        recon = self.idwt(approx_out, detail_out)
        out = recon + self.skip(x)
        return out


class WaveletNeuralOperator1D(nn.Module):
    """Full 1D Wavelet Neural Operator (WNO).

    Operates across multiple scales using stacked wavelet convolution blocks,
    with an optional conservative flux head for exact discrete conservation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 4,
        wavelet: str = "haar",
        conservative: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.conservative = conservative

        # Lifting projection: in_channels + 1 (spatial coordinate) -> hidden_channels
        self.lifting = nn.Linear(in_channels + 1, hidden_channels)

        # Wavelet convolution layers
        self.layers = nn.ModuleList(
            [
                WaveletConv1D(hidden_channels, hidden_channels, wavelet=wavelet)
                for _ in range(num_layers)
            ]
        )

        if self.conservative:
            self.flux_head = InterfaceFluxHead1D(
                in_channels=hidden_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
            )
            self.divergence = DiscreteFluxDivergence(spatial_dim=1)
        else:
            self.projection = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, out_channels),
            )

    def forward(
        self,
        x: torch.Tensor,
        dt: float = 0.01,
        dx: Optional[float] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Evaluate 1D WNO.

        Args:
            x: Input state tensor, shape (B, C_in, N).
            dt: Timestep for conservative update.
            dx: Cell size. If None, 1.0 / N is used.

        Returns:
            If conservative: Tuple of (x_next, face_fluxes).
            Otherwise: Predicted state x_next of shape (B, C_out, N).
        """
        batch_size, _, n_cells = x.shape
        if dx is None:
            dx = 1.0 / float(n_cells)

        # Append normalized coordinate grid [0, 1]
        coords = torch.linspace(0.0, 1.0, n_cells, device=x.device, dtype=x.dtype)
        coords = coords.view(1, 1, n_cells).repeat(batch_size, 1, 1)
        x_with_coords = torch.cat([x, coords], dim=1)  # (B, C+1, N)

        # Lifting
        h = self.lifting(x_with_coords.transpose(1, 2)).transpose(1, 2)

        # Stacked wavelet blocks with GELU activations
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if idx < len(self.layers) - 1:
                h = F.gelu(h)

        if self.conservative:
            fluxes = self.flux_head(h)
            div = self.divergence.forward_1d(fluxes, dx=dx)
            x_next = x[:, : self.out_channels, :] + float(dt) * div
            return x_next, fluxes
        else:
            out = self.projection(h.transpose(1, 2)).transpose(1, 2)
            return out


class WaveletConv2D(nn.Module):
    """2D Wavelet Convolution block using 2D Haar decomposition."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # In 2D Haar DWT, one level produces 4 subbands: LL, LH, HL, HH
        # Each subband has shape (B, C, H // 2, W // 2)
        self.subband_conv = nn.Conv2d(
            in_channels * 4, out_channels * 4, kernel_size=1
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of 2D Wavelet Convolution."""
        batch_size, c, h, w = x.shape
        # Haar 2D downsampling into 4 subbands
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        # LL (average), LH (horizontal), HL (vertical), HH (diagonal)
        ll = 0.5 * (x00 + x01 + x10 + x11)
        lh = 0.5 * (x00 - x01 + x10 - x11)
        hl = 0.5 * (x00 + x01 - x10 - x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)

        subbands = torch.cat([ll, lh, hl, hh], dim=1)
        transformed = self.subband_conv(subbands)

        c_out = self.out_channels
        ll_t = transformed[:, 0:c_out]
        lh_t = transformed[:, c_out : 2 * c_out]
        hl_t = transformed[:, 2 * c_out : 3 * c_out]
        hh_t = transformed[:, 3 * c_out :]

        # Inverse 2D Haar reconstruction
        y00 = 0.5 * (ll_t + lh_t + hl_t + hh_t)
        y01 = 0.5 * (ll_t - lh_t + hl_t - hh_t)
        y10 = 0.5 * (ll_t + lh_t - hl_t - hh_t)
        y11 = 0.5 * (ll_t - lh_t - hl_t + hh_t)

        recon = torch.zeros(
            (batch_size, c_out, h, w), device=x.device, dtype=x.dtype
        )
        recon[:, :, 0::2, 0::2] = y00
        recon[:, :, 0::2, 1::2] = y01
        recon[:, :, 1::2, 0::2] = y10
        recon[:, :, 1::2, 1::2] = y11

        return recon + self.skip(x)


class WaveletNeuralOperator2D(nn.Module):
    """Full 2D Wavelet Neural Operator (WNO)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 4,
        conservative: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.conservative = conservative

        # Lifting projection: in_channels + 2 (spatial coordinates x, y) -> hidden_channels
        self.lifting = nn.Conv2d(in_channels + 2, hidden_channels, kernel_size=1)

        self.layers = nn.ModuleList(
            [WaveletConv2D(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )

        if self.conservative:
            self.flux_head = InterfaceFluxHead2D(
                in_channels=hidden_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
            )
            self.divergence = DiscreteFluxDivergence(spatial_dim=2)
        else:
            self.projection = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            )

    def forward(
        self,
        x: torch.Tensor,
        dt: float = 0.01,
        dx: Optional[float] = None,
        dy: Optional[float] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Evaluate 2D WNO.

        Args:
            x: Input state tensor, shape (B, C_in, H, W).
            dt: Timestep.
            dx: Grid spacing along x.
            dy: Grid spacing along y.

        Returns:
            If conservative: Tuple of (x_next, (flux_x, flux_y)).
            Otherwise: Predicted state x_next of shape (B, C_out, H, W).
        """
        batch_size, _, h, w = x.shape
        if dx is None:
            dx = 1.0 / float(h)
        if dy is None:
            dy = 1.0 / float(w)

        grid_x = torch.linspace(0.0, 1.0, h, device=x.device, dtype=x.dtype)
        grid_y = torch.linspace(0.0, 1.0, w, device=x.device, dtype=x.dtype)
        gx, gy = torch.meshgrid(grid_x, grid_y, indexing="ij")
        coords = (
            torch.stack([gx, gy], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        )
        x_with_coords = torch.cat([x, coords], dim=1)

        h_feat = self.lifting(x_with_coords)
        for idx, layer in enumerate(self.layers):
            h_feat = layer(h_feat)
            if idx < len(self.layers) - 1:
                h_feat = F.gelu(h_feat)

        if self.conservative:
            flux_x, flux_y = self.flux_head(h_feat)
            div = self.divergence.forward_2d(flux_x, flux_y, dx=dx, dy=dy)
            x_next = x[:, : self.out_channels, :, :] + float(dt) * div
            return x_next, (flux_x, flux_y)
        else:
            return self.projection(h_feat)
