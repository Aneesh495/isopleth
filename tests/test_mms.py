"""Tests for Method of Manufactured Solutions and grid refinement convergence (I06)."""

import pytest
import math

from isopleth.numerics.verification import BurgersMMS, ShallowWaterVerification

def test_burgers_mms_convergence():
    mms = BurgersMMS(c_speed=0.2, viscosity=0.01)
    table = mms.run_refinement_study(resolutions=(16, 32, 64), t_final=0.05)
    
    assert len(table.entries) == 3
    # Check that error decreases monotonically
    e16 = table.entries[0].l2_error
    e32 = table.entries[1].l2_error
    e64 = table.entries[2].l2_error
    assert e32 < e16
    assert e64 < e32

    # EOC should be around 1.8 to 2.1 for second-order TVD on smooth solution
    eoc_final = table.final_eoc
    assert eoc_final >= 1.6, f"Expected second order EOC >= 1.6, got {eoc_final:.2f}"

def test_lake_at_rest_equilibrium():
    passed, eta_res, vel_res = ShallowWaterVerification.verify_lake_at_rest(
        resolution=(16, 16), steps=30, dt=0.002
    )
    assert passed is True
    assert eta_res < 1e-10
    assert vel_res < 1e-10
