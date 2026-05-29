"""Non-dimensionalization, Buckingham Pi similitude, and dimensionless transforms.

Provides dimensional analysis and scale-invariance tools across PDE families:
1. Burgers equation: Reynolds number, advective and diffusive time scales.
2. Shallow Water: Froude number, Rossby deformation radius, Burgers number, Strouhal number.
3. Gray-Scott: Species Peclet numbers, Damkoehler numbers, and kinetic scales.
4. Scale-invariant state transformers converting physical units into unit computational domains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


@dataclass(frozen=True)
class BurgersDimensionalScales:
    """Characteristic dimensional scales for 1D Burgers advection-diffusion."""

    velocity_scale: float
    length_scale: float
    viscosity: float
    reynolds_number: float
    advective_time: float
    diffusive_time: float
    peclet_number: float


@dataclass(frozen=True)
class ShallowWaterDimensionalScales:
    """Characteristic dimensional scales for 2D Shallow Water equations."""

    depth_scale: float
    velocity_scale: float
    length_scale: float
    gravity: float
    coriolis_param: float
    froude_number: float
    rossby_deformation_radius: float
    rossby_number: float
    burgers_number: float
    wave_celerity: float
    advective_time: float
    gravity_wave_time: float


@dataclass(frozen=True)
class GrayScottDimensionalScales:
    """Characteristic dimensional scales for Gray-Scott reaction-diffusion."""

    domain_size: float
    diffusion_u: float
    diffusion_v: float
    feed_rate: float
    kill_rate: float
    diffusive_time_u: float
    diffusive_time_v: float
    reaction_time: float
    peclet_ratio: float
    damkoehler_number: float


class BurgersSimilitude:
    """Calculates dimensionless groups for Burgers advection-diffusion."""

    @staticmethod
    def compute_scales(
        velocity_scale: float = 1.0,
        length_scale: float = 1.0,
        viscosity: float = 0.01,
    ) -> BurgersDimensionalScales:
        """Compute characteristic scales and dimensionless numbers."""
        u0 = float(abs(velocity_scale))
        l0 = float(abs(length_scale))
        nu = float(abs(viscosity))

        # Re = U0 * L0 / nu
        reynolds = (u0 * l0) / max(nu, 1e-15)
        t_adv = l0 / max(u0, 1e-15)
        t_diff = (l0**2) / max(nu, 1e-15)
        peclet = reynolds  # Peclet identical to Reynolds when Schmidt number is 1

        return BurgersDimensionalScales(
            velocity_scale=u0,
            length_scale=l0,
            viscosity=nu,
            reynolds_number=reynolds,
            advective_time=t_adv,
            diffusive_time=t_diff,
            peclet_number=peclet,
        )

    @staticmethod
    def to_dimensionless(
        u: torch.Tensor,
        x: torch.Tensor,
        t: float,
        scales: BurgersDimensionalScales,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Convert dimensional variables to dimensionless form u* = u/U0, x* = x/L0, t* = t/T_adv."""
        u_star = u / scales.velocity_scale
        x_star = x / scales.length_scale
        t_star = t / scales.advective_time
        return u_star, x_star, t_star

    @staticmethod
    def to_dimensional(
        u_star: torch.Tensor,
        scales: BurgersDimensionalScales,
    ) -> torch.Tensor:
        """Restore dimensional velocity: u = u* * U0."""
        return u_star * scales.velocity_scale


class ShallowWaterSimilitude:
    """Calculates dimensionless hydrodynamic parameters for Shallow Water systems."""

    @staticmethod
    def compute_scales(
        depth_scale: float = 1.0,
        velocity_scale: float = 1.0,
        length_scale: float = 1.0,
        gravity: float = 9.81,
        coriolis_param: float = 0.0,
    ) -> ShallowWaterDimensionalScales:
        """Compute Froude, Rossby, and Burgers numbers."""
        h0 = float(abs(depth_scale))
        u0 = float(abs(velocity_scale))
        l0 = float(abs(length_scale))
        g = float(abs(gravity))
        f = float(abs(coriolis_param))

        # Celerity c = sqrt(g * H0)
        celerity = math.sqrt(g * h0)

        # Froude number Fr = U0 / c
        froude = u0 / max(celerity, 1e-15)

        # Rossby deformation radius R_d = c / f
        if f > 1e-12:
            r_d = celerity / f
            rossby = u0 / (f * l0)
            burgers = (r_d / l0) ** 2
        else:
            r_d = float("inf")
            rossby = float("inf")
            burgers = float("inf")

        t_adv = l0 / max(u0, 1e-15)
        t_wave = l0 / max(celerity, 1e-15)

        return ShallowWaterDimensionalScales(
            depth_scale=h0,
            velocity_scale=u0,
            length_scale=l0,
            gravity=g,
            coriolis_param=f,
            froude_number=froude,
            rossby_deformation_radius=r_d,
            rossby_number=rossby,
            burgers_number=burgers,
            wave_celerity=celerity,
            advective_time=t_adv,
            gravity_wave_time=t_wave,
        )

    @staticmethod
    def to_dimensionless(
        h: torch.Tensor,
        hu: torch.Tensor,
        hv: torch.Tensor,
        scales: ShallowWaterDimensionalScales,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert dimensional shallow water state to dimensionless form."""
        h_star = h / scales.depth_scale
        # Discharge hu_star = hu / (H0 * U0)
        discharge_scale = scales.depth_scale * scales.velocity_scale
        hu_star = hu / discharge_scale
        hv_star = hv / discharge_scale
        return h_star, hu_star, hv_star

    @staticmethod
    def to_dimensional(
        h_star: torch.Tensor,
        hu_star: torch.Tensor,
        hv_star: torch.Tensor,
        scales: ShallowWaterDimensionalScales,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Restore dimensional depth and momentum discharges."""
        h = h_star * scales.depth_scale
        discharge_scale = scales.depth_scale * scales.velocity_scale
        hu = hu_star * discharge_scale
        hv = hv_star * discharge_scale
        return h, hu, hv


class GrayScottSimilitude:
    """Calculates chemical time scales and Damkoehler numbers for Gray-Scott morphogenesis."""

    @staticmethod
    def compute_scales(
        domain_size: float = 1.0,
        diffusion_u: float = 2e-5,
        diffusion_v: float = 1e-5,
        feed_rate: float = 0.03,
        kill_rate: float = 0.06,
    ) -> GrayScottDimensionalScales:
        """Compute chemical reaction rates and diffusion time scales."""
        l0 = float(domain_size)
        du = float(diffusion_u)
        dv = float(diffusion_v)
        f = float(feed_rate)
        k = float(kill_rate)

        # Diffusive time scales T_diff = L^2 / D
        t_diff_u = (l0**2) / max(du, 1e-15)
        t_diff_v = (l0**2) / max(dv, 1e-15)

        # Typical reaction time scale T_rxn = 1 / (F + k)
        t_rxn = 1.0 / max(f + k, 1e-15)

        # Ratio of diffusion coefficients
        peclet_ratio = du / max(dv, 1e-15)

        # Damkoehler number Da = T_diff / T_rxn (reaction rate vs diffusion rate)
        damkoehler = t_diff_u / max(t_rxn, 1e-15)

        return GrayScottDimensionalScales(
            domain_size=l0,
            diffusion_u=du,
            diffusion_v=dv,
            feed_rate=f,
            kill_rate=k,
            diffusive_time_u=t_diff_u,
            diffusive_time_v=t_diff_v,
            reaction_time=t_rxn,
            peclet_ratio=peclet_ratio,
            damkoehler_number=damkoehler,
        )
