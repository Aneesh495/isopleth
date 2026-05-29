"""High-resolution approximate and exact Riemann solvers for conservation laws.

Includes:
1. Exact Shallow Water Riemann solver using two-wave fan Newton-Raphson iteration.
2. HLL (Harten-Lax-van Leer) Riemann solver with Einfeldt wave speed estimates.
3. HLLC (HLL with Contact) Riemann solver restoring shear and contact discontinuities.
4. Roe Riemann solver with Harten-Hyman sonic entropy fix.
5. Audusse well-balanced hydrostatic reconstruction for non-smooth topography.
6. Desingularized wetting-and-drying momentum limiter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch


@dataclass
class RiemannState1D:
    """Conserved state vector for 1D shallow water [h, hu]."""

    h: torch.Tensor
    hu: torch.Tensor

    @property
    def u(self) -> torch.Tensor:
        """Velocity u = hu / h with desingularization."""
        return torch.where(self.h > 1e-6, self.hu / self.h, torch.zeros_like(self.hu))


@dataclass
class RiemannFlux1D:
    """Interfacial flux vector [F_mass, F_momentum]."""

    f_mass: torch.Tensor
    f_momentum: torch.Tensor


class ExactShallowWaterRiemannSolver:
    """Exact Riemann solver for 1D Saint-Venant shallow water equations.

    Solves the nonlinear algebraic equation for intermediate water depth h*
    across left and right shock or rarefaction waves using Newton-Raphson iteration.
    """

    def __init__(self, gravity: float = 9.81, max_iter: int = 25, tolerance: float = 1e-8) -> None:
        self.g = gravity
        self.max_iter = max_iter
        self.tol = tolerance

    def _wave_function(
        self,
        h_star: float,
        h_k: float,
        c_k: float,
    ) -> Tuple[float, float]:
        """Evaluates wave curve f_k(h*) and its derivative f_k'(h*)."""
        g = self.g
        if h_star <= h_k:
            # Rarefaction wave
            c_star = math.sqrt(g * h_star)
            f_val = 2.0 * (c_star - c_k)
            df_val = math.sqrt(g / h_star)
        else:
            # Shock wave
            q_k = math.sqrt(0.5 * g * (h_star + h_k) / (h_star * h_k))
            f_val = (h_star - h_k) * q_k
            dq_dh = 0.25 * g * (1.0 / (h_star * h_k) - (h_star + h_k) / (h_star**2 * h_k)) / q_k
            df_val = q_k + (h_star - h_k) * dq_dh
        return f_val, df_val

    def solve_star_state(
        self,
        h_L: float,
        u_L: float,
        h_R: float,
        u_R: float,
    ) -> Tuple[float, float]:
        """Solves for (h_star, u_star) using Newton-Raphson iteration."""
        g = self.g
        c_L = math.sqrt(g * h_L)
        c_R = math.sqrt(g * h_R)

        # Check for dry bed generation
        if (u_R - u_L) >= 2.0 * (c_L + c_R):
            # Vacuum/dry bed forms in intermediate state
            return 0.0, 0.5 * (u_L + u_R)

        # Two-rarefaction approximation as initial guess
        c_star_guess = 0.5 * (c_L + c_R) - 0.25 * (u_R - u_L)
        h_star = max(1e-4, (c_star_guess**2) / g)

        delta_u = u_R - u_L

        for _ in range(self.max_iter):
            f_L, df_L = self._wave_function(h_star, h_L, c_L)
            f_R, df_R = self._wave_function(h_star, h_R, c_R)

            phi = f_L + f_R + delta_u
            dphi = df_L + df_R

            if abs(dphi) < 1e-12:
                break

            h_next = h_star - phi / dphi
            h_next = max(1e-6, h_next)

            if abs(h_next - h_star) < self.tol:
                h_star = h_next
                break
            h_star = h_next

        # Velocity in star state
        f_L_final, _ = self._wave_function(h_star, h_L, c_L)
        f_R_final, _ = self._wave_function(h_star, h_R, c_R)
        u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R_final - f_L_final)

        return h_star, u_star


class HLLRiemannSolver1D:
    """HLL (Harten-Lax-van Leer) two-wave approximate Riemann solver for 1D shallow water."""

    def __init__(self, gravity: float = 9.81) -> None:
        self.g = gravity

    def compute_flux(
        self,
        h_L: torch.Tensor,
        hu_L: torch.Tensor,
        h_R: torch.Tensor,
        hu_R: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates mass and momentum fluxes at cell interface."""
        g = self.g
        eps = 1e-6

        u_L = torch.where(h_L > eps, hu_L / h_L, torch.zeros_like(hu_L))
        u_R = torch.where(h_R > eps, hu_R / h_R, torch.zeros_like(hu_R))

        c_L = torch.sqrt(torch.clamp(g * h_L, min=0.0))
        c_R = torch.sqrt(torch.clamp(g * h_R, min=0.0))

        # Roe-averaged wave speeds
        sqrt_h_L = torch.sqrt(torch.clamp(h_L, min=0.0))
        sqrt_h_R = torch.sqrt(torch.clamp(h_R, min=0.0))
        denom = sqrt_h_L + sqrt_h_R + 1e-12
        u_roe = (sqrt_h_L * u_L + sqrt_h_R * u_R) / denom
        c_roe = torch.sqrt(0.5 * g * (h_L + h_R))

        # Einfeldt wave speed estimates
        s_L = torch.minimum(u_L - c_L, u_roe - c_roe)
        s_R = torch.maximum(u_R + c_R, u_roe + c_roe)

        # Physical left and right fluxes
        f_mass_L = hu_L
        f_mom_L = hu_L * u_L + 0.5 * g * (h_L**2)

        f_mass_R = hu_R
        f_mom_R = hu_R * u_R + 0.5 * g * (h_R**2)

        # HLL intermediate state flux
        ds = s_R - s_L + 1e-12
        hll_mass = (s_R * f_mass_L - s_L * f_mass_R + s_L * s_R * (h_R - h_L)) / ds
        hll_mom = (s_R * f_mom_L - s_L * f_mom_R + s_L * s_R * (hu_R - hu_L)) / ds

        # Branching based on wave speed signs
        flux_mass = torch.where(s_L >= 0.0, f_mass_L, torch.where(s_R <= 0.0, f_mass_R, hll_mass))
        flux_mom = torch.where(s_L >= 0.0, f_mom_L, torch.where(s_R <= 0.0, f_mom_R, hll_mom))

        return flux_mass, flux_mom


class HLLCRiemannSolver2D:
    """HLLC (HLL with Contact) Riemann solver for 2D Shallow Water equations.

    Restores the contact discontinuity / transverse shear wave S_*
    to preserve rotational shear flows and contact fronts across grid interfaces.
    """

    def __init__(self, gravity: float = 9.81) -> None:
        self.g = gravity

    def compute_flux_x(
        self,
        h_L: torch.Tensor,
        hu_L: torch.Tensor,
        hv_L: torch.Tensor,
        h_R: torch.Tensor,
        hu_R: torch.Tensor,
        hv_R: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes [F_h, F_hu, F_hv] along x-normal cell face."""
        g = self.g
        eps = 1e-6

        u_L = torch.where(h_L > eps, hu_L / h_L, torch.zeros_like(hu_L))
        v_L = torch.where(h_L > eps, hv_L / h_L, torch.zeros_like(hv_L))
        u_R = torch.where(h_R > eps, hu_R / h_R, torch.zeros_like(hu_R))
        v_R = torch.where(h_R > eps, hv_R / h_R, torch.zeros_like(hv_R))

        c_L = torch.sqrt(torch.clamp(g * h_L, min=0.0))
        c_R = torch.sqrt(torch.clamp(g * h_R, min=0.0))

        # Einfeldt wave estimates
        sqrt_h_L = torch.sqrt(torch.clamp(h_L, min=0.0))
        sqrt_h_R = torch.sqrt(torch.clamp(h_R, min=0.0))
        denom = sqrt_h_L + sqrt_h_R + 1e-12
        u_roe = (sqrt_h_L * u_L + sqrt_h_R * u_R) / denom
        c_roe = torch.sqrt(0.5 * g * (h_L + h_R))

        s_L = torch.minimum(u_L - c_L, u_roe - c_roe)
        s_R = torch.maximum(u_R + c_R, u_roe + c_roe)

        # Contact speed S_*
        num_star = s_L * h_R * (u_R - s_R) - s_R * h_L * (u_L - s_L)
        den_star = h_R * (u_R - s_R) - h_L * (u_L - s_L) + 1e-12
        s_star = num_star / den_star

        # Normal physical fluxes
        f_h_L = hu_L
        f_hu_L = hu_L * u_L + 0.5 * g * (h_L**2)
        f_hv_L = hu_L * v_L

        f_h_R = hu_R
        f_hu_R = hu_R * u_R + 0.5 * g * (h_R**2)
        f_hv_R = hu_R * v_R

        # Intermediate star states U_*L and U_*R
        factor_L = (s_L - u_L) / (s_L - s_star + 1e-12)
        h_star_L = h_L * factor_L
        hu_star_L = h_star_L * s_star
        hv_star_L = h_star_L * v_L

        factor_R = (s_R - u_R) / (s_R - s_star + 1e-12)
        h_star_R = h_R * factor_R
        hu_star_R = h_star_R * s_star
        hv_star_R = h_star_R * v_R

        # Star fluxes via Rankine-Hugoniot condition
        f_h_star_L = f_h_L + s_L * (h_star_L - h_L)
        f_hu_star_L = f_hu_L + s_L * (hu_star_L - hu_L)
        f_hv_star_L = f_hv_L + s_L * (hv_star_L - hv_L)

        f_h_star_R = f_h_R + s_R * (h_star_R - h_R)
        f_hu_star_R = f_hu_R + s_R * (hu_star_R - hu_R)
        f_hv_star_R = f_hv_R + s_R * (hv_star_R - hv_R)

        # Region selection
        f_h = torch.where(
            s_L >= 0.0,
            f_h_L,
            torch.where(
                s_star >= 0.0,
                f_h_star_L,
                torch.where(s_R >= 0.0, f_h_star_R, f_h_R),
            ),
        )
        f_hu = torch.where(
            s_L >= 0.0,
            f_hu_L,
            torch.where(
                s_star >= 0.0,
                f_hu_star_L,
                torch.where(s_R >= 0.0, f_hu_star_R, f_hu_R),
            ),
        )
        f_hv = torch.where(
            s_L >= 0.0,
            f_hv_L,
            torch.where(
                s_star >= 0.0,
                f_hv_star_L,
                torch.where(s_R >= 0.0, f_hv_star_R, f_hv_R),
            ),
        )

        return f_h, f_hu, f_hv


class RoeRiemannSolver1D:
    """Roe approximate Riemann solver with Harten-Hyman entropy fix."""

    def __init__(self, gravity: float = 9.81, delta_entropy: float = 0.1) -> None:
        self.g = gravity
        self.delta = delta_entropy

    def compute_flux(
        self,
        h_L: torch.Tensor,
        hu_L: torch.Tensor,
        h_R: torch.Tensor,
        hu_R: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Roe flux with wave-by-wave sonic entropy fix."""
        g = self.g
        eps = 1e-6

        u_L = torch.where(h_L > eps, hu_L / h_L, torch.zeros_like(hu_L))
        u_R = torch.where(h_R > eps, hu_R / h_R, torch.zeros_like(hu_R))

        sqrt_h_L = torch.sqrt(torch.clamp(h_L, min=0.0))
        sqrt_h_R = torch.sqrt(torch.clamp(h_R, min=0.0))
        denom = sqrt_h_L + sqrt_h_R + 1e-12

        # Roe averages
        u_tilde = (sqrt_h_L * u_L + sqrt_h_R * u_R) / denom
        c_tilde = torch.sqrt(0.5 * g * (h_L + h_R))

        # Wave speeds (eigenvalues)
        lambda_1 = u_tilde - c_tilde
        lambda_2 = u_tilde + c_tilde

        # Harten-Hyman entropy fix for sonic rarefactions
        def entropy_fix(lam: torch.Tensor) -> torch.Tensor:
            eps_sonic = self.delta * c_tilde
            return torch.where(
                torch.abs(lam) >= eps_sonic,
                torch.abs(lam),
                (lam**2 + eps_sonic**2) / (2.0 * eps_sonic),
            )

        abs_lam_1 = entropy_fix(lambda_1)
        abs_lam_2 = entropy_fix(lambda_2)

        # Wave strengths (wave amplitudes alpha)
        dh = h_R - h_L
        dhu = hu_R - hu_L

        alpha_1 = (lambda_2 * dh - dhu) / (2.0 * c_tilde + 1e-12)
        alpha_2 = (dhu - lambda_1 * dh) / (2.0 * c_tilde + 1e-12)

        # Left and right physical fluxes
        f_mass_L = hu_L
        f_mom_L = hu_L * u_L + 0.5 * g * (h_L**2)

        f_mass_R = hu_R
        f_mom_R = hu_R * u_R + 0.5 * g * (h_R**2)

        # Roe dissipation: sum |lambda_p| * alpha_p * R_p
        # Right eigenvector R1 = [1, lambda_1]^T
        # Right eigenvector R2 = [1, lambda_2]^T
        diss_mass = abs_lam_1 * alpha_1 + abs_lam_2 * alpha_2
        diss_mom = abs_lam_1 * alpha_1 * lambda_1 + abs_lam_2 * alpha_2 * lambda_2

        flux_mass = 0.5 * (f_mass_L + f_mass_R) - 0.5 * diss_mass
        flux_mom = 0.5 * (f_mom_L + f_mom_R) - 0.5 * diss_mom

        return flux_mass, flux_mom


class AudusseWellBalancedReconstruction:
    """Audusse hydrostatic reconstruction for topography preserving lake-at-rest to machine precision."""

    def __init__(self, gravity: float = 9.81) -> None:
        self.g = gravity

    def reconstruct_1d(
        self,
        h_L: torch.Tensor,
        h_R: torch.Tensor,
        b_L: torch.Tensor,
        b_R: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs hydrostatic reconstruction over cell face topography.

        Returns:
            (h_L_star, h_R_star, b_face, source_topography)
        """
        g = self.g
        b_face = torch.maximum(b_L, b_R)
        eta_L = h_L + b_L
        eta_R = h_R + b_R

        h_L_star = torch.clamp(eta_L - b_face, min=0.0)
        h_R_star = torch.clamp(eta_R - b_face, min=0.0)

        # Well-balanced source term discretization matching hydrostatic pressure flux
        s_topo_L = 0.5 * g * (h_L_star**2 - h_L**2)
        s_topo_R = 0.5 * g * (h_R**2 - h_R_star**2)

        return h_L_star, h_R_star, b_face, (s_topo_L + s_topo_R)


class DesingularizedMomentumLimiter:
    """Desingularizes velocity calculation in thin and wetting-and-drying flow regimes."""

    def __init__(self, threshold_eps: float = 1e-5) -> None:
        self.eps = threshold_eps

    def compute_velocity(self, h: torch.Tensor, hu: torch.Tensor) -> torch.Tensor:
        """Evaluates desingularized velocity u avoiding division by zero in dry zones.

        u = sqrt(2) * h * hu / sqrt(h^4 + max(h, eps)^4)
        """
        denom = torch.sqrt(h**4 + torch.clamp(h, min=self.eps)**4)
        return math.sqrt(2.0) * h * hu / denom
