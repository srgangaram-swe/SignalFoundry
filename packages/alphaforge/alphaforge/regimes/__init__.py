"""Regime detection, change-point models, and incremental-value evidence.

Public API (SF-S3-MR4):

- :class:`RegimeModel` — the shared contract: causal fit/filter, posterior
  outputs, canonical state ordering, and deterministic identity.
- :func:`expanding_state_probabilities` — the only sanctioned way to turn a
  regime model into a feature; refuses non-causal models.
- rule-based, Gaussian-mixture, hidden-Markov, CUSUM, and Bayesian online
  change-point models.
- :func:`segment_bayesian` / :func:`segment_kernel` — retrospective segmenters,
  deliberately *not* ``RegimeModel`` subclasses so they cannot be used causally.
- :func:`evaluate_regime_value` — incremental value against a no-regime
  baseline, which is what replaces "the chart looks right".
"""

from alphaforge.regimes.base import (
    MAX_STATES,
    ExpandingFitPlan,
    NotFittedError,
    RegimeError,
    RegimeLeakageError,
    RegimeModel,
    RegimeReport,
    RegimeStates,
    expanding_state_probabilities,
)
from alphaforge.regimes.changepoint import (
    BayesianOnlineChangePoint,
    CusumRegime,
    Segmentation,
    segment_bayesian,
    segment_kernel,
)
from alphaforge.regimes.evidence import (
    RegimeValueEvidence,
    compare_regime_models,
    evaluate_regime_value,
    forward_returns,
)
from alphaforge.regimes.mixture import GaussianHMMRegime, GaussianMixtureRegime
from alphaforge.regimes.rules import RuleBasedRegime, rule_statistic

__all__ = [
    "MAX_STATES",
    "BayesianOnlineChangePoint",
    "CusumRegime",
    "ExpandingFitPlan",
    "GaussianHMMRegime",
    "GaussianMixtureRegime",
    "NotFittedError",
    "RegimeError",
    "RegimeLeakageError",
    "RegimeModel",
    "RegimeReport",
    "RegimeStates",
    "RegimeValueEvidence",
    "RuleBasedRegime",
    "Segmentation",
    "compare_regime_models",
    "evaluate_regime_value",
    "expanding_state_probabilities",
    "forward_returns",
    "rule_statistic",
    "segment_bayesian",
    "segment_kernel",
]
