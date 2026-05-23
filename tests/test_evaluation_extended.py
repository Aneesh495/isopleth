"""Tests for ablation suite, performance benchmarks, independent verifier, and acceptance campaign."""

import numpy as np
import pytest
import torch

from isopleth.evaluation import (
    AblationExperimentSuite,
    AblationVariant,
    AcceptanceCampaignRunner,
    IndependentTrajectoryVerifier,
    NeuralOperatorBenchmarkSuite,
)
from isopleth.models.flux_operator import MultiscaleFaceFluxOperator1D
from isopleth.numerics.burgers import Burgers1DSolver, BurgersSolverConfig


def test_ablation_variants_forward_and_study() -> None:
    """Verifies that all 5 ablation variants instantiate and evaluate rollouts."""
    suite = AblationExperimentSuite()
    nx = 32
    u0 = torch.sin(2 * np.pi * torch.linspace(0, 1, nx)).view(1, 1, nx)
    traj = u0.repeat(1, 4, 1)

    for variant in AblationVariant:
        model = suite.build_ablation_model(variant, hidden_channels=16, modes=4)
        stable, rel_l2, drift, low_e, high_e = suite.evaluate_model_rollout(
            model=model,
            initial_state=u0,
            true_trajectory=traj,
            num_steps=3,
            dx=1.0/nx,
            dt=0.01,
        )
        assert stable > 0
        assert not np.isnan(rel_l2)
        assert not np.isnan(drift)

    study = suite.run_study(test_initial=u0, test_true_traj=traj, num_steps=3)
    assert len(study.variant_results) == 5
    assert len(study.sample_efficiency_curve) == 3
    assert len(study.executive_findings) >= 4


def test_performance_benchmarks_and_matched_speedup() -> None:
    """Verifies latency, memory scaling, and matched-accuracy speedup profiling."""
    bench = NeuralOperatorBenchmarkSuite(device=torch.device("cpu"))
    model = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16, modes=4)
    solver = Burgers1DSolver(BurgersSolverConfig(viscosity=0.01))

    total_p, train_p = bench.count_parameters(model)
    assert total_p > 0
    assert train_p == total_p

    nx = 32
    u0 = torch.sin(2 * np.pi * torch.linspace(0, 1, nx)).view(1, 1, nx)
    matched = bench.compute_matched_accuracy_speedup(
        surrogate_model=model,
        solver=solver,
        u0=u0,
        num_coarse_steps=3,
        coarse_dt=0.01,
    )

    assert matched.surrogate_steps_required == 3
    assert matched.solver_steps_required >= 3
    assert matched.speedup_factor > 0.0

    report = bench.run_full_suite(sample_model=model, sample_solver=solver, sample_u0=u0)
    assert len(report.resolution_scaling) == 2
    assert report.total_model_parameters == total_p


def test_independent_trajectory_verifier_and_negatives() -> None:
    """Verifies that the independent verifier certifies clean runs and rejects negative controls."""
    verifier = IndependentTrajectoryVerifier(
        tolerance_rel_l2=0.05,
        tolerance_mass_drift=1e-5,
        enforce_positivity=True,
    )

    # 1. Clean synthetic run
    T, nx = 5, 32
    clean_true = np.ones((1, T, nx)) * 2.0
    clean_pred = np.ones((1, T, nx)) * 2.0
    res_clean = verifier.verify_arrays(clean_pred, clean_true, dataset_name="clean_test")
    assert res_clean.passed_all_gates
    assert len(res_clean.failure_reasons) == 0
    assert len(res_clean.certificate_sha256) == 64

    # 2. Negative control: mass drift injection
    bad_mass_pred = clean_pred.copy()
    bad_mass_pred[:, -1, :] += 0.5  # Mass injection at final step
    res_drift = verifier.verify_arrays(bad_mass_pred, clean_true, dataset_name="drift_negative")
    assert not res_drift.passed_all_gates
    assert any("mass drift" in r.lower() for r in res_drift.failure_reasons)

    # 3. Negative control: positivity violation
    bad_pos_pred = clean_pred.copy()
    bad_pos_pred[0, 2, 5] = -0.1
    res_pos = verifier.verify_arrays(bad_pos_pred, clean_true, dataset_name="pos_negative")
    assert not res_pos.passed_all_gates
    assert any("positivity violation" in r.lower() for r in res_pos.failure_reasons)

    # 4. Negative control: NaN injection
    bad_nan_pred = clean_pred.copy()
    bad_nan_pred[0, 1, 1] = np.nan
    res_nan = verifier.verify_arrays(bad_nan_pred, clean_true, dataset_name="nan_negative")
    assert not res_nan.passed_all_gates
    assert res_nan.nan_or_inf_detected


def test_acceptance_campaign_execution() -> None:
    """Executes complete 12-gate acceptance campaign driver."""
    runner = AcceptanceCampaignRunner()
    report = runner.execute_full_campaign()

    assert report.total_gates == 12
    assert report.gates_passed == 12
    assert report.gates_failed == 0
    assert report.all_passed
    assert len(report.sha256_bundle_digest) == 64

    # Check each gate ID presence
    gate_ids = [outcome.gate_id for outcome in report.outcomes]
    for i in range(1, 13):
        expected_id = f"IA{i:02d}"
        assert expected_id in gate_ids
