"""Ranking portfolios and uncertainty-aware volatility targeting (SF-S4-MR1).

Six allocation policies over a cross-section of predicted scores, plus two
orthogonal sizing layers — uncertainty shrinkage and volatility targeting —
that can be applied to any of them.

The policies differ in **how much of the score they trust**, which is the axis
that actually matters when scores are noisy:

* ``top_k`` and ``quantile`` trust only the *ordering*, and only near the ends.
* ``rank_weighted`` trusts the full ordering but not the magnitudes, so a single
  extreme score cannot dominate the book.
* ``score_weighted`` trusts the magnitudes, which is right when the score is a
  calibrated expected return and wrong when it is an arbitrary signal strength.
* ``inverse_volatility`` ignores the score for sizing and uses it only for
  direction.
* ``long_short_spread`` is ``top_k`` on both tails, sized to a target net.

Two invariants hold for every policy:

**Determinism under ties.** Ranking breaks ties on the symbol name, so two
assets with identical scores always produce the same book. Without this, an
equal-score pair reorders between runs and the backtest is irreproducible —
a failure that only appears intermittently, which is the worst kind.

**Nothing is returned unprojected.** Every policy ends in
:func:`~alphaforge.portfolio.contracts.project_to_feasible`, so a returned book
has already satisfied every declared limit or the call raised.

Explicit non-goals from the issue, honoured here: no unrestricted leverage (a
leverage ceiling is a required constraint), no hidden cash (the cash weight is
carried on the result), and **no full Kelly** — uncertainty sizing is fractional
and capped, because full Kelly on estimated moments is a route to ruin when the
estimates are wrong.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from alphaforge.portfolio.contracts import (
    AllocationResult,
    PortfolioConstraints,
    PortfolioError,
    project_to_feasible,
)

FloatArray = NDArray[np.float64]

PolicyName = Literal[
    "top_k",
    "long_short_spread",
    "quantile",
    "score_weighted",
    "rank_weighted",
    "inverse_volatility",
]

#: Ceiling on the uncertainty-sizing tilt. A position may be scaled down freely
#: but never scaled *up* beyond this multiple of its unsized weight: an
#: apparently very-low-uncertainty name is usually an estimation artefact, and
#: uncapped confidence sizing turns that artefact into concentration.
MAX_CONFIDENCE_MULTIPLE = 2.0


def _validated_scores(scores: pd.Series) -> pd.Series:
    """Return finite scores on a unique, deterministically ordered index."""
    if not isinstance(scores, pd.Series):
        raise PortfolioError("scores must be a pandas Series indexed by symbol")
    if scores.index.has_duplicates:
        raise PortfolioError("scores must have a unique symbol index")
    clean = scores.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise PortfolioError("no finite scores to allocate over")
    # Sort by symbol first so every downstream operation is order-stable, then
    # rely on a stable sort for the score ordering. Ties therefore break on the
    # symbol name, deterministically, rather than on input order.
    return clean.sort_index()


def _deterministic_rank(scores: pd.Series) -> pd.Series:
    """Return dense ascending ranks with ties broken on the symbol name."""
    ordered = scores.sort_values(kind="stable")
    return pd.Series(np.arange(1, len(ordered) + 1, dtype=float), index=ordered.index).reindex(
        scores.index
    )


def _finalize(
    raw: pd.Series,
    *,
    policy: str,
    constraints: PortfolioConstraints,
    previous: pd.Series | None,
    liquidity_caps: pd.Series | None,
    diagnostics: dict[str, Any] | None = None,
) -> AllocationResult:
    """Project raw target weights and package the accounting."""
    symbols = raw.index
    previous_aligned = (
        np.zeros(len(symbols))
        if previous is None
        else previous.reindex(symbols).fillna(0.0).to_numpy(dtype=float)
    )
    caps = None
    if liquidity_caps is not None:
        caps = liquidity_caps.reindex(symbols).fillna(0.0).to_numpy(dtype=float)

    projected = project_to_feasible(
        raw.to_numpy(dtype=float),
        constraints,
        previous=previous_aligned if constraints.max_turnover is not None else None,
        liquidity_caps=caps,
    )
    weights = pd.Series(projected, index=symbols, name="target_weight")
    gross = float(np.sum(np.abs(projected)))
    # Turnover is measured over the union of the previous and current universes,
    # so a name that left the universe is charged for being exited rather than
    # silently vanishing at zero cost.
    if previous is None:
        turnover = gross
    else:
        union = previous.index.union(symbols)
        turnover = float(
            (weights.reindex(union).fillna(0.0) - previous.reindex(union).fillna(0.0)).abs().sum()
        )
    # A policy that selects few names under a tight position cap cannot reach
    # its gross target: k names can carry at most k * max_position. The book is
    # still valid, but the shortfall is unintended cash and must be impossible
    # to miss rather than merely derivable from `cash_weight`.
    gross_target = min(constraints.max_gross, constraints.deployable)
    active = int(np.count_nonzero(np.abs(raw.to_numpy(dtype=float)) > 0.0))
    record = dict(diagnostics or {})
    record["deployed_fraction"] = float(gross / gross_target) if gross_target > 0.0 else 0.0
    record["gross_shortfall"] = bool(gross < gross_target - 1e-9)
    if record["gross_shortfall"]:
        record["shortfall_reason"] = (
            f"{active} selected names at a {constraints.max_position} position cap reach "
            f"gross {active * constraints.max_position:.4f} against a {gross_target:.4f} target"
        )
    return AllocationResult(
        weights=weights,
        policy=policy,
        cash_weight=float(1.0 - gross) if constraints.long_only else float(constraints.cash_buffer),
        gross=gross,
        net=float(np.sum(projected)),
        turnover=turnover,
        n_positions=int(np.count_nonzero(np.abs(projected) > 1e-12)),
        constraints=constraints,
        diagnostics=record,
    )


# ---------------------------------------------------------------------------
# Ranking policies
# ---------------------------------------------------------------------------


def top_k_portfolio(
    scores: pd.Series,
    constraints: PortfolioConstraints,
    *,
    k: int,
    weighting: Literal["equal", "score", "rank"] = "equal",
    previous: pd.Series | None = None,
    liquidity_caps: pd.Series | None = None,
) -> AllocationResult:
    """Long the ``k`` highest-scoring names.

    The most common research portfolio, and the most honest about what a noisy
    score supports: it uses the ordering at the top of the cross-section and
    discards everything else.

    Args:
        k: Names to hold. Must not exceed the universe.
        weighting: ``equal`` sizes them identically, ``score`` in proportion to
            score above the selection boundary, ``rank`` in proportion to rank.
    """
    clean = _validated_scores(scores)
    if not 1 <= k <= len(clean):
        raise PortfolioError(f"k must be in [1, {len(clean)}], got {k}")
    selected = clean.sort_values(ascending=False, kind="stable").head(k).sort_index()
    raw = _weight_selection(selected, weighting, constraints)
    full = pd.Series(0.0, index=clean.index)
    full.loc[raw.index] = raw
    return _finalize(
        full,
        policy=f"top_k[{weighting}]",
        constraints=constraints,
        previous=previous,
        liquidity_caps=liquidity_caps,
        diagnostics={"k": k, "weighting": weighting},
    )


def _weight_selection(
    selected: pd.Series, weighting: str, constraints: PortfolioConstraints
) -> pd.Series:
    """Size a selected long sleeve to the deployable gross target."""
    target = min(constraints.max_gross, constraints.deployable)
    if weighting == "equal":
        base = pd.Series(1.0, index=selected.index)
    elif weighting == "rank":
        base = _deterministic_rank(selected)
    elif weighting == "score":
        # Shift so the weakest selected name gets ~zero weight rather than a
        # negative one; a raw score can be negative even at the top of a weak
        # cross-section.
        base = selected - selected.min()
        if float(base.sum()) <= 0.0:
            base = pd.Series(1.0, index=selected.index)
    else:
        raise PortfolioError(f"unsupported weighting {weighting!r}")
    return base / float(base.sum()) * target


def long_short_spread_portfolio(
    scores: pd.Series,
    constraints: PortfolioConstraints,
    *,
    k: int,
    previous: pd.Series | None = None,
    liquidity_caps: pd.Series | None = None,
) -> AllocationResult:
    """Long the top ``k`` and short the bottom ``k``, dollar-balanced.

    Each sleeve is sized to half the gross target, so the raw book is
    market-neutral by construction before constraints act on it. Whether it
    *stays* neutral depends on ``max_net``, which is the caller's declared
    policy rather than an assumption made here.

    Raises:
        PortfolioError: If shorting is disallowed, or the universe cannot supply
            two disjoint sleeves of ``k`` names.
    """
    clean = _validated_scores(scores)
    if constraints.long_only:
        raise PortfolioError("a long-short spread requires long_only=False")
    if not 1 <= k <= len(clean) // 2:
        raise PortfolioError(f"k must be in [1, {len(clean) // 2}] for disjoint sleeves, got {k}")
    ordered = clean.sort_values(ascending=False, kind="stable")
    target = min(constraints.max_gross, constraints.deployable)
    weights = pd.Series(0.0, index=clean.index)
    weights.loc[ordered.head(k).index] = target / (2.0 * k)
    weights.loc[ordered.tail(k).index] = -target / (2.0 * k)
    return _finalize(
        weights,
        policy="long_short_spread",
        constraints=constraints,
        previous=previous,
        liquidity_caps=liquidity_caps,
        diagnostics={"k": k},
    )


def quantile_portfolio(
    scores: pd.Series,
    constraints: PortfolioConstraints,
    *,
    quantile: float = 0.2,
    previous: pd.Series | None = None,
    liquidity_caps: pd.Series | None = None,
) -> AllocationResult:
    """Long the top ``quantile`` of the cross-section, short the bottom.

    The scale-free sibling of :func:`long_short_spread_portfolio`: sleeve size
    tracks the universe, so the book does not silently concentrate when the
    universe shrinks — which a fixed ``k`` does.
    """
    clean = _validated_scores(scores)
    if not 0.0 < quantile <= 0.5:
        raise PortfolioError("quantile must lie in (0, 0.5]")
    count = max(int(np.floor(len(clean) * quantile)), 1)
    if constraints.long_only:
        return top_k_portfolio(
            clean,
            constraints,
            k=count,
            weighting="equal",
            previous=previous,
            liquidity_caps=liquidity_caps,
        )
    if count > len(clean) // 2:
        raise PortfolioError("quantile selects overlapping sleeves; reduce it")
    return long_short_spread_portfolio(
        clean, constraints, k=count, previous=previous, liquidity_caps=liquidity_caps
    )


def score_weighted_portfolio(
    scores: pd.Series,
    constraints: PortfolioConstraints,
    *,
    previous: pd.Series | None = None,
    liquidity_caps: pd.Series | None = None,
) -> AllocationResult:
    """Weight in proportion to the demeaned score.

    Trusts score *magnitudes*, which is correct when the score is a calibrated
    expected return and wrong when it is an arbitrary strength. Demeaning makes
    the raw book dollar-neutral; for a long-only mandate the negative side is
    dropped by the projection's box constraint.
    """
    clean = _validated_scores(scores)
    centred = clean - float(clean.mean())
    magnitude = float(centred.abs().sum())
    if magnitude <= 0.0:
        # A flat cross-section carries no information; equal weight is the only
        # defensible response, and it is recorded in the diagnostics.
        return equal_weight_fallback(clean, constraints, previous, liquidity_caps)
    target = min(constraints.max_gross, constraints.deployable)
    return _finalize(
        centred / magnitude * target,
        policy="score_weighted",
        constraints=constraints,
        previous=previous,
        liquidity_caps=liquidity_caps,
    )


def rank_weighted_portfolio(
    scores: pd.Series,
    constraints: PortfolioConstraints,
    *,
    previous: pd.Series | None = None,
    liquidity_caps: pd.Series | None = None,
) -> AllocationResult:
    """Weight in proportion to demeaned rank.

    The robust counterpart to score weighting: it keeps the full ordering but
    discards magnitudes, so one extreme score cannot dominate the book. On noisy
    predictions this is usually the better bet, and the pair is worth comparing
    precisely because the difference measures how much the magnitudes are worth.
    """
    clean = _validated_scores(scores)
    ranks = _deterministic_rank(clean)
    centred = ranks - float(ranks.mean())
    magnitude = float(centred.abs().sum())
    if magnitude <= 0.0:
        return equal_weight_fallback(clean, constraints, previous, liquidity_caps)
    target = min(constraints.max_gross, constraints.deployable)
    return _finalize(
        centred / magnitude * target,
        policy="rank_weighted",
        constraints=constraints,
        previous=previous,
        liquidity_caps=liquidity_caps,
    )


def inverse_volatility_portfolio(
    scores: pd.Series,
    volatility: pd.Series,
    constraints: PortfolioConstraints,
    *,
    previous: pd.Series | None = None,
    liquidity_caps: pd.Series | None = None,
) -> AllocationResult:
    """Take the score's *direction* and size by inverse volatility.

    Deliberately discards score magnitude. Volatility is far more estimable than
    expected return, so a policy that only asks the score which way to lean and
    lets risk decide the size is a serious competitor to anything that trusts
    the magnitudes.

    A name with zero, missing, or non-finite volatility is **excluded**, not
    given the median: an unmeasurable risk is not a small risk, and substituting
    a plausible number is how an untradeable name acquires a full position.
    """
    clean = _validated_scores(scores)
    vol = volatility.reindex(clean.index).astype(float)
    usable = np.isfinite(vol.to_numpy()) & (vol.to_numpy() > 0.0)
    if not usable.any():
        raise PortfolioError("no asset has a usable positive volatility estimate")
    excluded = int((~usable).sum())
    selected = clean[usable]
    inverse = 1.0 / vol[usable]
    direction = np.sign(selected)
    direction[direction == 0.0] = 1.0
    raw = direction * inverse
    magnitude = float(raw.abs().sum())
    target = min(constraints.max_gross, constraints.deployable)
    weights = pd.Series(0.0, index=clean.index)
    weights.loc[raw.index] = raw / magnitude * target
    if constraints.long_only:
        weights = weights.clip(lower=0.0)
    return _finalize(
        weights,
        policy="inverse_volatility",
        constraints=constraints,
        previous=previous,
        liquidity_caps=liquidity_caps,
        diagnostics={"excluded_unmeasurable_volatility": excluded},
    )


def equal_weight_fallback(
    scores: pd.Series,
    constraints: PortfolioConstraints,
    previous: pd.Series | None = None,
    liquidity_caps: pd.Series | None = None,
) -> AllocationResult:
    """Equal-weight book, recorded as a fallback rather than a silent default."""
    clean = _validated_scores(scores)
    target = min(constraints.max_gross, constraints.deployable)
    weights = pd.Series(target / len(clean), index=clean.index)
    return _finalize(
        weights,
        policy="equal_weight",
        constraints=constraints,
        previous=previous,
        liquidity_caps=liquidity_caps,
        diagnostics={"reason": "cross-section carried no dispersion"},
    )


# ---------------------------------------------------------------------------
# Sizing layers
# ---------------------------------------------------------------------------


def apply_uncertainty_sizing(
    weights: pd.Series,
    uncertainty: pd.Series,
    *,
    strength: float = 1.0,
    max_multiple: float = MAX_CONFIDENCE_MULTIPLE,
) -> pd.Series:
    """Shrink positions whose predictions are least certain.

    The tilt is ``(median uncertainty / uncertainty) ** strength``, normalized to
    preserve gross exposure, then **capped**. Scaling by the ratio to the median
    keeps the book's size unchanged and only redistributes it, so this layer
    changes *concentration* rather than leverage.

    ``strength`` interpolates continuously from no tilt (``0``) to inverse-
    uncertainty weighting (``1``). It is capped at ``max_multiple`` because an
    apparently very-low-uncertainty name is usually an estimation artefact, and
    uncapped confidence sizing converts that artefact into concentration. This
    is fractional and bounded by construction — **not** full Kelly, which the
    issue names as a non-goal and which is a route to ruin on estimated moments.

    Names with missing or non-positive uncertainty keep their original weight
    rather than being tilted on a number that does not exist.
    """
    if not 0.0 <= strength <= 1.0:
        raise PortfolioError("strength must lie in [0, 1]")
    if max_multiple < 1.0:
        raise PortfolioError("max_multiple must be at least 1.0")
    aligned = uncertainty.reindex(weights.index).astype(float)
    values = aligned.to_numpy()
    usable = np.isfinite(values) & (values > 0.0)
    if not usable.any():
        return weights.copy()
    reference = float(np.median(values[usable]))
    tilt = np.ones(len(weights))
    tilt[usable] = np.clip(
        (reference / values[usable]) ** strength, 1.0 / max_multiple, max_multiple
    )
    tilted = weights.to_numpy(dtype=float) * tilt
    gross_before = float(np.sum(np.abs(weights.to_numpy(dtype=float))))
    gross_after = float(np.sum(np.abs(tilted)))
    if gross_after > 0.0 and gross_before > 0.0:
        tilted *= gross_before / gross_after
    return pd.Series(tilted, index=weights.index, name=weights.name)


def apply_volatility_target(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    target_volatility: float,
    max_leverage: float,
    periods_per_year: int = 252,
) -> tuple[pd.Series, dict[str, Any]]:
    """Scale the whole book so its ex-ante volatility meets a target.

    Ex-ante portfolio volatility is ``sqrt(w' Σ w)`` annualized. The scale factor
    is ``target / realized_ex_ante``, **capped by ``max_leverage``** — that cap
    is the whole reason this is safe to run. In a quiet regime the unconstrained
    factor grows without bound, and a volatility target without a leverage
    ceiling is precisely the mechanism that turns a calm market into a
    catastrophic one when the estimate is stale.

    Returns:
        ``(scaled_weights, diagnostics)`` where the diagnostics record the
        ex-ante volatility before and after, the raw factor, the applied factor,
        and whether the leverage cap bound — so a book that silently failed to
        reach its target is visible rather than assumed to have reached it.
    """
    if target_volatility <= 0.0 or not np.isfinite(target_volatility):
        raise PortfolioError("target_volatility must be finite and positive")
    if max_leverage <= 0.0 or not np.isfinite(max_leverage):
        raise PortfolioError("max_leverage must be finite and positive")
    symbols = weights.index
    matrix = covariance.reindex(index=symbols, columns=symbols)
    if matrix.isna().to_numpy().any():
        raise PortfolioError("covariance must cover every allocated symbol")
    vector = weights.to_numpy(dtype=float)
    variance = float(vector @ matrix.to_numpy(dtype=float) @ vector)
    if variance < 0.0:
        raise PortfolioError("covariance produced a negative portfolio variance")
    ex_ante = float(np.sqrt(variance) * np.sqrt(periods_per_year))
    if ex_ante <= 0.0:
        return weights.copy(), {
            "ex_ante_volatility": 0.0,
            "raw_factor": float("nan"),
            "applied_factor": 1.0,
            "leverage_capped": False,
            "target_met": False,
        }
    raw_factor = target_volatility / ex_ante
    gross = float(np.sum(np.abs(vector)))
    leverage_limit = max_leverage / gross if gross > 0.0 else raw_factor
    applied = min(raw_factor, leverage_limit)
    scaled = weights * applied
    return scaled, {
        "ex_ante_volatility": ex_ante,
        "scaled_ex_ante_volatility": ex_ante * applied,
        "raw_factor": raw_factor,
        "applied_factor": applied,
        "leverage_capped": bool(applied < raw_factor - 1e-12),
        "target_met": bool(abs(ex_ante * applied - target_volatility) < 1e-9),
    }


POLICIES: dict[str, Any] = {
    "top_k": top_k_portfolio,
    "long_short_spread": long_short_spread_portfolio,
    "quantile": quantile_portfolio,
    "score_weighted": score_weighted_portfolio,
    "rank_weighted": rank_weighted_portfolio,
    "inverse_volatility": inverse_volatility_portfolio,
}
