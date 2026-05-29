"""Tests for Wavelet Neural Operator (WNO) baseline models."""

import pytest
import torch

from isopleth.models.wavelet_operator import (
    DWT1D,
    IDWT1D,
    WaveletConv1D,
    WaveletConv2D,
    WaveletNeuralOperator1D,
    WaveletNeuralOperator2D,
)


def test_dwt_and_idwt_1d_haar():
    dwt = DWT1D(wavelet="haar")
    idwt = IDWT1D(wavelet="haar")

    x = torch.randn(2, 3, 32)
    approx, detail = dwt(x)
    assert approx.shape == (2, 3, 16)
    assert detail.shape == (2, 3, 16)

    recon = idwt(approx, detail)
    assert recon.shape == (2, 3, 32)
    # Haar is an orthogonal wavelet; perfect reconstruction holds on periodic signal
    assert torch.allclose(x, recon, atol=1e-5)


def test_wavelet_conv_1d():
    wconv = WaveletConv1D(in_channels=8, out_channels=16, wavelet="haar")
    x = torch.randn(2, 8, 32)
    out = wconv(x)
    assert out.shape == (2, 16, 32)


def test_wno_1d_unconstrained_and_conservative():
    # Unconstrained WNO
    wno_standard = WaveletNeuralOperator1D(
        in_channels=1,
        out_channels=1,
        hidden_channels=16,
        num_layers=2,
        conservative=False,
    )
    x = torch.randn(2, 1, 32)
    out_std = wno_standard(x)
    assert out_std.shape == (2, 1, 32)

    # Conservative WNO with exact telescopic conservation
    wno_cons = WaveletNeuralOperator1D(
        in_channels=1,
        out_channels=1,
        hidden_channels=16,
        num_layers=2,
        conservative=True,
    )
    wno_cons = wno_cons.to(dtype=torch.float64)
    x_dbl = torch.randn(2, 1, 32, dtype=torch.float64)
    out_next, fluxes = wno_cons(x_dbl, dt=0.01, dx=1.0 / 32)
    assert out_next.shape == (2, 1, 32)
    assert fluxes.shape == (2, 1, 32)

    # Telescopic cancellation check
    mass_in = x_dbl.sum(dim=-1)
    mass_out = out_next.sum(dim=-1)
    assert torch.allclose(mass_in, mass_out, atol=1e-12)


def test_wno_2d_unconstrained_and_conservative():
    # Unconstrained 2D WNO
    wno2d = WaveletNeuralOperator2D(
        in_channels=2,
        out_channels=2,
        hidden_channels=16,
        num_layers=2,
        conservative=False,
    )
    x2d = torch.randn(2, 2, 16, 16)
    out2d = wno2d(x2d)
    assert out2d.shape == (2, 2, 16, 16)

    # Conservative 2D WNO
    wno2d_cons = WaveletNeuralOperator2D(
        in_channels=2,
        out_channels=2,
        hidden_channels=16,
        num_layers=2,
        conservative=True,
    )
    wno2d_cons = wno2d_cons.to(dtype=torch.float64)
    x2d_dbl = torch.randn(2, 2, 16, 16, dtype=torch.float64)
    out2d_next, (fx, fy) = wno2d_cons(x2d_dbl, dt=0.01, dx=1.0 / 16, dy=1.0 / 16)
    assert out2d_next.shape == (2, 2, 16, 16)
    assert fx.shape == (2, 2, 16, 16)
    assert fy.shape == (2, 2, 16, 16)

    # 2D discrete mass conservation check
    mass_in = x2d_dbl.sum(dim=(-2, -1))
    mass_out = out2d_next.sum(dim=(-2, -1))
    assert torch.allclose(mass_in, mass_out, atol=1e-12)
