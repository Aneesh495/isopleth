"""Evaluation suites, metrics, shift benchmarks, diagnostics, testing, and acceptance campaign."""

from isopleth.evaluation.ablations import (
    AblationExperimentSuite,
    AblationMetrics,
    AblationStudySummary,
    AblationVariant,
    SampleEfficiencyPoint,
)
from isopleth.evaluation.acceptance import (
    AcceptanceCampaignReport,
    AcceptanceCampaignRunner,
    GateExecutionOutcome,
)
from isopleth.evaluation.benchmarks import (
    BenchmarkSuiteReport,
    MatchedAccuracyComparison,
    NeuralOperatorBenchmarkSuite,
    ResolutionBenchmarkPoint,
)
from isopleth.evaluation.diagnostics import (
    ConservationDriftPowerSpectrum,
    DriftPowerSpectrumMetrics,
    HessianDistortionReport,
    SpectralDissipationAnalyzer,
    SpectralDissipationMetrics,
    StatisticalComparisonReport,
    StatisticalHypothesisTester,
    SurrogateHessianDistortionAnalyzer,
)
from isopleth.evaluation.independent_check import (
    IndependentTrajectoryVerifier,
    IndependentVerificationResult,
)
from isopleth.evaluation.metrics import (
    EvaluationMetrics,
    TrajectoryEvaluator,
)
from isopleth.evaluation.shifts import (
    DistributionShiftEvaluator,
    ShiftExperimentReport,
    ShiftExperimentSuite,
)
from isopleth.evaluation.statistical_testing import (
    BootstrapMetricEstimator,
    MultiModelRankReport,
    MultiModelStatisticalRanker,
    PairwiseEquivalenceTester,
    PermutationSignificanceTester,
)

__all__ = [
    "AblationExperimentSuite",
    "AblationMetrics",
    "AblationStudySummary",
    "AblationVariant",
    "AcceptanceCampaignReport",
    "AcceptanceCampaignRunner",
    "BenchmarkSuiteReport",
    "BootstrapMetricEstimator",
    "ConservationDriftPowerSpectrum",
    "DistributionShiftEvaluator",
    "DriftPowerSpectrumMetrics",
    "EvaluationMetrics",
    "GateExecutionOutcome",
    "HessianDistortionReport",
    "IndependentTrajectoryVerifier",
    "IndependentVerificationResult",
    "MatchedAccuracyComparison",
    "MultiModelRankReport",
    "MultiModelStatisticalRanker",
    "NeuralOperatorBenchmarkSuite",
    "PairwiseEquivalenceTester",
    "PermutationSignificanceTester",
    "ResolutionBenchmarkPoint",
    "SampleEfficiencyPoint",
    "ShiftExperimentReport",
    "ShiftExperimentSuite",
    "SpectralDissipationAnalyzer",
    "SpectralDissipationMetrics",
    "StatisticalComparisonReport",
    "StatisticalHypothesisTester",
    "SurrogateHessianDistortionAnalyzer",
    "TrajectoryEvaluator",
]
