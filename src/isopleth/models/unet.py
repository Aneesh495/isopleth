"""Original convolutional U-Net baseline implementation for 1D and 2D dynamical systems.

Features multiscale encoder-decoder blocks, skip connections, coordinate channels,
and FiLM parameter conditioning.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from isopleth.models.conditioning import FiLMBlock

class ConvBlock1D(nn.Module):
    """Dual convolution residual block in 1D."""

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular")
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular")
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.film = FiLMBlock(out_channels, condition_dim)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        res = self.skip(x)
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        if cond is not None:
            h = self.film(h, cond)
        return F.gelu(h + res)

class UNet1D(nn.Module):
    """1D convolutional U-Net architecture."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        condition_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # In + 1 coordinate channel
        self.in_conv = nn.Conv1d(in_channels + 1, base_channels, kernel_size=3, padding=1, padding_mode="circular")

        # Encoder
        self.enc1 = ConvBlock1D(base_channels, base_channels, condition_dim)
        self.down1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.enc2 = ConvBlock1D(base_channels * 2, base_channels * 2, condition_dim)
        self.down2 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = ConvBlock1D(base_channels * 4, base_channels * 4, condition_dim)

        # Decoder
        self.up2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock1D(base_channels * 4, base_channels * 2, condition_dim)

        self.up1 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock1D(base_channels * 2, base_channels, condition_dim)

        # Out projection
        self.out_conv = nn.Conv1d(base_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular")

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Tensor of shape [B, C, nx] or [B, nx].
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, C, nx = x.shape

        # Grid coordinate
        grid = torch.linspace(0.0, 1.0, nx, device=x.device, dtype=x.dtype).view(1, 1, nx).repeat(B, 1, 1)
        h = self.in_conv(torch.cat([x, grid], dim=1))

        # Down
        e1 = self.enc1(h, cond)
        d1 = self.down1(e1)

        e2 = self.enc2(d1, cond)
        d2 = self.down2(e2)

        # Bottleneck
        b = self.bottleneck(d2, cond)

        # Up
        u2 = self.up2(b)
        # Handle odd dimensions
        if u2.shape[-1] != e2.shape[-1]:
            u2 = F.interpolate(u2, size=e2.shape[-1], mode="linear", align_corners=False)
        m2 = self.dec2(torch.cat([u2, e2], dim=1), cond)

        u1 = self.up1(m2)
        if u1.shape[-1] != e1.shape[-1]:
            u1 = F.interpolate(u1, size=e1.shape[-1], mode="linear", align_corners=False)
        m1 = self.dec1(torch.cat([u1, e1], dim=1), cond)

        out = self.out_conv(m1)
        return out.squeeze(1) if self.out_channels == 1 else out

class ConvBlock2D(nn.Module):
    """Dual convolution residual block in 2D."""

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular")
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular")
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.film = FiLMBlock(out_channels, condition_dim)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        res = self.skip(x)
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        if cond is not None:
            h = self.film(h, cond)
        return F.gelu(h + res)

class UNet2D(nn.Module):
    """2D convolutional U-Net architecture."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        condition_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_conv = nn.Conv2d(in_channels + 2, base_channels, kernel_size=3, padding=1, padding_mode="circular")

        self.enc1 = ConvBlock2D(base_channels, base_channels, condition_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.enc2 = ConvBlock2D(base_channels * 2, base_channels * 2, condition_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.bottleneck = ConvBlock2D(base_channels * 4, base_channels * 4, condition_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock2D(base_channels * 4, base_channels * 2, condition_dim)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock2D(base_channels * 2, base_channels, condition_dim)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular")

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Tensor of shape [B, C, ny, nx].
        """
        B, C, ny, nx = x.shape

        gx = torch.linspace(0.0, 1.0, nx, device=x.device, dtype=x.dtype)
        gy = torch.linspace(0.0, 1.0, ny, device=x.device, dtype=x.dtype)
        GY, GX = torch.meshgrid(gy, gx, indexing="ij")
        coord = torch.stack([GX, GY], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        h = self.in_conv(torch.cat([x, coord], dim=1))

        e1 = self.enc1(h, cond)
        d1 = self.down1(e1)

        e2 = self.enc2(d1, cond)
        d2 = self.down2(e2)

        b = self.bottleneck(d2, cond)

        u2 = self.up2(b)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        m2 = self.dec2(torch.cat([u2, e2], dim=1), cond)

        u1 = self.up1(m2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        m1 = self.dec1(torch.cat([u1, e1], dim=1), cond)

        return self.out_conv(m1)
