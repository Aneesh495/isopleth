"""Tests for non-parametric statistical hypothesis testing and multi-model ranking."""

from __future__ import annotations

import numpy as np
import pytest

from isopleth.evaluation.statistical_testing import (
    BootstrapMetricEstimator,
    MultiModelRankReport,
    MultiModelStatisticalRanker,
    PairwiseEquivalenceTester,
    PermutationSignificanceTester,
)


def test_multi_model_statistical_ranker() -> None:
    rng = np.random.default_rng(42)
    # 30 seeds, 3 models with distinct means
    m1 = rng.normal(loc=0.05, scale=0.01, size=(30, 1))
    m2 = rng.normal(loc=0.10, scale=0.01, size=(30, 1))
    m3 = rng.normal(loc=0.15, scale=0.01, size=(30, 1))
    err_mat = np.hstack([m1, m2, m3])

    report = MultiModelStatisticalRanker.rank_models(
        error_matrix=err_mat,
        model_names=["ModelA", "ModelB", "ModelC"],
        alpha=0.05,
    )

    assert isinstance(report, MultiModelRankReport)
    assert report.is_globally_significant is True
    assert report.average_ranks["ModelA"] < report.average_ranks["ModelC"]


def test_pairwise_equivalence_tester() -> None:
    rng = np.random.default_rng(123)
    # Errors differ by only 0.001 on average, well within margin 0.05
    err_a = rng.normal(loc=0.05, scale=0.01, size=40)
    err_b = err_a + rng.normal(loc=0.001, scale=0.002, size=40)

    is_equiv, p1, p2 = PairwiseEquivalenceTester.test_equivalence(
        err_a, err_b, equivalence_margin=0.05, alpha=0.05
    )
    assert is_equiv is True
    assert p1 < 0.05
    assert p2 < 0.05


def test_bootstrap_metric_estimator() -> None:
    estimator = BootstrapMetricEstimator(num_resamples=500, seed=42)
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, lower, upper = estimator.estimate_mean_ci(vals, confidence_level=0.90)

    assert mean == pytest.approx(3.0)
    assert lower <= mean <= upper


def test_permutation_significance_tester() -> None:
    tester = PermutationSignificanceTester(num_permutations=1000, seed=99)
    err_a = [0.10, 0.12, 0.09, 0.11, 0.13, 0.10, 0.12, 0.09, 0.11, 0.14]
    err_b = [0.50, 0.48, 0.52, 0.49, 0.51, 0.50, 0.48, 0.52, 0.49, 0.53]

    mean_diff, p_val = tester.test_paired_difference(err_a, err_b)
    assert mean_diff < 0.0
    assert p_val < 0.01
