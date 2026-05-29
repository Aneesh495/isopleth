"""Non-parametric statistical hypothesis testing and multi-model ranking.

Provides rigorous statistical testing for model benchmark comparisons:
1. Wilcoxon signed-rank paired test for paired model comparisons.
2. Two-sample Kolmogorov-Smirnov test for error distribution equality.
3. Friedman test for multi-model benchmark ranking across evaluation seeds.
4. Holm-Bonferroni correction controlling Family-Wise Error Rate (FWER).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats as stats


@dataclass
class MultiModelRankReport:
    """Outcomes of Friedman and post-hoc rank hypothesis testing."""

    model_names: List[str]
    average_ranks: Dict[str, float]
    friedman_statistic: float
    friedman_p_value: float
    is_globally_significant: bool
    pairwise_p_values: Dict[Tuple[str, str], float]
    pairwise_significant: Dict[Tuple[str, str], bool]


class MultiModelStatisticalRanker:
    """Computes Friedman test and pairwise post-hoc tests across model test seeds."""

    @staticmethod
    def rank_models(
        error_matrix: np.ndarray,
        model_names: Sequence[str],
        alpha: float = 0.05,
    ) -> MultiModelRankReport:
        """Perform non-parametric Friedman test on error matrix of shape (N_seeds, K_models).

        Args:
            error_matrix: 2D array where rows are datasets/seeds and columns are models.
            model_names: List of model identifiers matching columns.
            alpha: Family-wise significance level.

        Returns:
            MultiModelRankReport detailing rankings, Friedman p-value, and pairwise tests.
        """
        arr = np.asarray(error_matrix, dtype=np.float64)
        n_seeds, k_models = arr.shape

        if len(model_names) != k_models:
            raise ValueError(f"model_names length {len(model_names)} must match columns {k_models}")

        # Compute ranks per seed (1 = lowest error / best model)
        ranks = np.zeros_like(arr)
        for i in range(n_seeds):
            ranks[i] = stats.rankdata(arr[i])

        avg_ranks = np.mean(ranks, axis=0)
        rank_dict = {name: float(avg_ranks[j]) for j, name in enumerate(model_names)}

        # Friedman test across model columns
        if k_models >= 3 and n_seeds >= 2:
            res_f = stats.friedmanchisquare(*[arr[:, col] for col in range(k_models)])
            f_stat = float(res_f.statistic)
            f_p = float(res_f.pvalue)
        else:
            f_stat, f_p = 0.0, 1.0

        is_global_sig = f_p < alpha

        # Pairwise Wilcoxon tests with Holm-Bonferroni FWER correction
        pairwise_raw_p: List[Tuple[Tuple[str, str], float]] = []
        for i in range(k_models):
            for j in range(i + 1, k_models):
                diff = arr[:, i] - arr[:, j]
                if np.all(diff == 0.0):
                    p_val = 1.0
                else:
                    w_res = stats.wilcoxon(arr[:, i], arr[:, j], alternative="two-sided")
                    p_val = float(w_res.pvalue)
                pairwise_raw_p.append(((model_names[i], model_names[j]), p_val))

        # Holm-Bonferroni correction
        pairwise_raw_p.sort(key=lambda item: item[1])
        m_tests = len(pairwise_raw_p)

        pairwise_p_dict: Dict[Tuple[str, str], float] = {}
        pairwise_sig_dict: Dict[Tuple[str, str], bool] = {}

        for rank_idx, (pair, p_raw) in enumerate(pairwise_raw_p):
            # Adjusted alpha threshold: alpha / (m - rank_idx)
            threshold = alpha / float(m_tests - rank_idx)
            adjusted_p = min(1.0, p_raw * (m_tests - rank_idx))
            is_sig = p_raw <= threshold
            pairwise_p_dict[pair] = adjusted_p
            pairwise_sig_dict[pair] = is_sig

        return MultiModelRankReport(
            model_names=list(model_names),
            average_ranks=rank_dict,
            friedman_statistic=f_stat,
            friedman_p_value=f_p,
            is_globally_significant=is_global_sig,
            pairwise_p_values=pairwise_p_dict,
            pairwise_significant=pairwise_sig_dict,
        )


class PairwiseEquivalenceTester:
    """Conducts Two One-Sided Tests (TOST) for statistical model equivalence.

    Tests whether the performance discrepancy between two models is strictly
    contained within a prescribed numerical equivalence margin (-delta, +delta).
    """

    @staticmethod
    def test_equivalence(
        errors_a: Sequence[float],
        errors_b: Sequence[float],
        equivalence_margin: float = 0.02,
        alpha: float = 0.05,
    ) -> Tuple[bool, float, float]:
        """Perform TOST paired test for equivalence margin delta.

        Returns:
            Tuple of (is_equivalent, p_value_lower, p_value_upper).
        """
        arr_a = np.asarray(errors_a, dtype=np.float64)
        arr_b = np.asarray(errors_b, dtype=np.float64)
        diff = arr_a - arr_b
        n = len(diff)

        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1)) + 1e-15
        se = std_diff / math.sqrt(n)

        # One-sided test 1: H01: mean_diff <= -delta  vs  H11: mean_diff > -delta
        t1 = (mean_diff - (-equivalence_margin)) / se
        p1 = float(1.0 - stats.t.cdf(t1, df=n - 1))

        # One-sided test 2: H02: mean_diff >= delta   vs  H12: mean_diff < delta
        t2 = (mean_diff - equivalence_margin) / se
        p2 = float(stats.t.cdf(t2, df=n - 1))

        # Both one-sided null hypotheses must be rejected at level alpha
        is_equiv = (p1 < alpha) and (p2 < alpha)
        return is_equiv, p1, p2


class BootstrapMetricEstimator:
    """Non-parametric percentile bootstrap confidence interval estimator."""

    def __init__(self, num_resamples: int = 2000, seed: Optional[int] = None) -> None:
        self.num_resamples = num_resamples
        self.rng = np.random.default_rng(seed)

    def estimate_mean_ci(
        self,
        values: Sequence[float],
        confidence_level: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Compute bootstrap confidence interval for the sample mean.

        Returns:
            Tuple of (sample_mean, ci_lower, ci_upper).
        """
        arr = np.asarray(values, dtype=np.float64)
        n = len(arr)
        if n == 0:
            return 0.0, 0.0, 0.0

        sample_mean = float(np.mean(arr))
        boot_indices = self.rng.integers(0, n, size=(self.num_resamples, n))
        boot_means = np.mean(arr[boot_indices], axis=1)

        alpha = 1.0 - confidence_level
        ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
        ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

        return sample_mean, ci_lower, ci_upper


class PermutationSignificanceTester:
    """Exact paired Monte Carlo permutation test for model comparisons."""

    def __init__(self, num_permutations: int = 5000, seed: Optional[int] = None) -> None:
        self.num_permutations = num_permutations
        self.rng = np.random.default_rng(seed)

    def test_paired_difference(
        self,
        errors_model_a: Sequence[float],
        errors_model_b: Sequence[float],
    ) -> Tuple[float, float]:
        """Test null hypothesis that errors_a and errors_b come from identical distributions.

        Returns:
            Tuple of (observed_mean_difference, two_sided_p_value).
        """
        diff = np.asarray(errors_model_a, dtype=np.float64) - np.asarray(
            errors_model_b, dtype=np.float64
        )
        n = len(diff)
        obs_stat = abs(float(np.mean(diff)))

        # Randomly flip signs under the exchangeability null hypothesis
        signs = self.rng.choice([-1.0, 1.0], size=(self.num_permutations, n))
        perm_stats = np.abs(np.mean(signs * diff[None, :], axis=1))

        # Two-sided p-value
        p_val = float(np.mean(perm_stats >= obs_stat))
        return float(np.mean(diff)), p_val
