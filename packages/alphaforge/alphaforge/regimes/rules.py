"""Interpretable rule-based regimes (SF-S3-MR4).

The baseline every probabilistic state model has to beat. A trend/volatility
rule needs no EM, no seed, and no convergence check; a fitted HMM that cannot
outperform one has bought its extra parameters and failure modes for nothing.

Rules are expressed as *soft* memberships rather than hard thresholds. A hard
cut at the 70th percentile of volatility makes the regime flip on a rounding
difference when volatility sits at the boundary, and that flicker propagates
straight into position sizing. A logistic transition of declared width turns the
same rule into a graded call whose ambiguity is visible in the posterior.

Thresholds are quantiles of the **training window only**, stored at fit time and
applied unchanged afterwards. Recomputing a quantile over the full sample is the
subtle leak here: a "high volatility" label would then be defined partly by
volatility that had not happened yet.
"""

from __future__ import annotations

from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from alphaforge.regimes.base import RegimeError, RegimeModel, RegimeReport, RegimeStates

FloatArray = NDArray[np.float64]

RuleName = Literal["volatility", "trend", "drawdown", "volatility_change"]


def rule_statistic(
    series: pd.Series, rule: RuleName, *, window: int = 21, long_window: int = 63
) -> pd.Series:
    """Return the trailing statistic a rule thresholds.

    Every statistic is trailing and uses ``min_periods=window``, so a partially
    warmed value is NaN rather than an average of whatever happens to exist.

    * ``volatility`` — trailing standard deviation of the series.
    * ``trend`` — trailing mean scaled by trailing standard deviation, i.e. a
      rolling t-like ratio, so the label means "trending relative to its own
      noise" rather than "went up a lot".
    * ``drawdown`` — depth below the trailing running peak of the cumulative
      sum, which is the state a risk manager actually reacts to.
    * ``volatility_change`` — log ratio of short to long trailing volatility;
      positive means volatility is expanding. This is the leading one of the
      four, since expansion typically precedes the stress a level rule only
      confirms after the fact.
    """
    if window < 2:
        raise RegimeError("window must be at least two observations")
    if long_window <= window:
        raise RegimeError("long_window must exceed window")
    values = series.astype(float)
    if rule == "volatility":
        return values.rolling(window, min_periods=window).std()
    if rule == "trend":
        mean = values.rolling(window, min_periods=window).mean()
        deviation = values.rolling(window, min_periods=window).std()
        return mean / deviation.replace(0.0, np.nan)
    if rule == "drawdown":
        cumulative = values.fillna(0.0).cumsum()
        peak = cumulative.rolling(long_window, min_periods=window).max()
        return cumulative - peak
    if rule == "volatility_change":
        short = values.rolling(window, min_periods=window).std()
        long = values.rolling(long_window, min_periods=long_window).std()
        ratio = short / long.replace(0.0, np.nan)
        return np.log(ratio.where(ratio > 0.0))
    raise RegimeError(f"unknown rule {rule!r}")


class RuleBasedRegime(RegimeModel):
    """Two-state regime from a trailing statistic and a training quantile.

    Args:
        rule: Which trailing statistic to threshold.
        quantile: Training quantile defining the boundary.
        width: Logistic transition width as a fraction of the training
            inter-quartile range. Smaller is closer to a hard threshold; ``0``
            is not permitted, because a hard rule cannot express ambiguity.
        window: Short trailing window.
        long_window: Long trailing window, for drawdown and volatility change.
        elevated_above: Whether the second state means "statistic above the
            threshold". ``drawdown`` sets this ``False`` — a deeper drawdown is
            a *lower* number, and getting this backwards silently inverts the
            regime.
    """

    name = "rule_regime"
    causal = True
    ordering_rule = "state 0 base, state 1 elevated (fixed by the rule's direction)"

    def __init__(
        self,
        rule: RuleName = "volatility",
        *,
        quantile: float = 0.7,
        width: float = 0.25,
        window: int = 21,
        long_window: int = 63,
    ) -> None:
        super().__init__()
        if not 0.0 < quantile < 1.0:
            raise RegimeError("quantile must lie strictly in (0, 1)")
        if width <= 0.0 or not np.isfinite(width):
            raise RegimeError("width must be finite and positive; a hard rule cannot be ambiguous")
        self.rule: RuleName = rule
        self.quantile = quantile
        self.width = width
        self.window = window
        self.long_window = long_window
        self.elevated_above = rule != "drawdown"
        self.threshold_: float = 0.0
        self.scale_: float = 1.0
        self.n_states = 2

    def fit_statistic(self, series: pd.Series) -> pd.Series:
        """Return the trailing statistic this rule thresholds."""
        return rule_statistic(series, self.rule, window=self.window, long_window=self.long_window)

    def fit(self, series: pd.Series | FloatArray) -> Self:
        """Fit the threshold on the training window's statistic only."""
        if not isinstance(series, pd.Series):
            series = pd.Series(np.asarray(series, dtype=float))
        statistic = self.fit_statistic(series)
        values = statistic.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size < 10:
            raise RegimeError(
                f"rule {self.rule!r} produced only {finite.size} finite training values; "
                "the trailing window is longer than the training sample"
            )
        self.threshold_ = float(np.quantile(finite, self.quantile))
        spread = float(np.subtract(*np.quantile(finite, [0.75, 0.25])))
        # A degenerate spread would make the logistic a step function; fall back
        # to the standard deviation and then to unity rather than dividing by ~0.
        self.scale_ = max(abs(spread) * self.width, float(np.std(finite)) * 1e-3, 1e-12)
        self._report = RegimeReport(
            model=self.name,
            n_states=2,
            iterations=1,
            max_iterations=1,
            converged=True,
            stopping_reason="closed_form",
            n_train_observations=int(finite.size),
            notes=(f"rule={self.rule}", f"quantile={self.quantile}"),
        )
        self._labels = self.state_labels
        self._fitted = True
        return self

    def filter(self, series: pd.Series) -> RegimeStates:
        """Return soft memberships for ``series`` under the fitted threshold."""
        self._ensure_fitted()
        if not isinstance(series, pd.Series):
            raise RegimeError("filter expects a pandas Series")
        statistic = self.fit_statistic(series).to_numpy(dtype=float)
        probabilities = np.full((statistic.size, 2), np.nan)
        observed = np.isfinite(statistic)
        if observed.any():
            direction = 1.0 if self.elevated_above else -1.0
            exponent = direction * (statistic[observed] - self.threshold_) / self.scale_
            elevated = 1.0 / (1.0 + np.exp(-np.clip(exponent, -700.0, 700.0)))
            probabilities[observed, 0] = 1.0 - elevated
            probabilities[observed, 1] = elevated
        return RegimeStates(
            probabilities=probabilities,
            labels=self._labels,
            index=series.index,
            causal=True,
        )

    def _fit(self, values: FloatArray) -> RegimeReport:  # pragma: no cover - fit is overridden
        raise NotImplementedError("RuleBasedRegime overrides fit directly")

    def _filter(self, values: FloatArray) -> FloatArray:  # pragma: no cover - overridden
        raise NotImplementedError("RuleBasedRegime overrides filter directly")

    @property
    def state_labels(self) -> tuple[str, ...]:
        return ("base", f"{self.rule}_elevated")

    def configuration(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "quantile": self.quantile,
            "width": self.width,
            "window": self.window,
            "long_window": self.long_window,
            "elevated_above": self.elevated_above,
        }

    def fitted_parameters(self) -> dict[str, Any]:
        return {"threshold": self.threshold_, "scale": self.scale_}
