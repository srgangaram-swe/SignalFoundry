"""Causal intraday diagnostics, fixed candidate budget and block uncertainty.

Both candidates use only completed, consecutive one-minute observations. A
decision at a close executes at the next bar's open, including both sides of
turnover costs. Missing minutes and session boundaries reset the window. OHLC
cannot prove executable fills, so this report alone never authorizes a broker.
"""

from __future__ import annotations

import math
import random
from collections import deque
from datetime import timedelta
from decimal import Decimal
from statistics import fmean
from typing import Any
from zoneinfo import ZoneInfo

from signal_foundry.boundary import FoundryError
from signal_foundry.trading.models import Bar, Plan

NEW_YORK = ZoneInfo("America/New_York")
CANDIDATES = ("momentum", "mean_reversion")


def target(prices: tuple[Decimal, ...], *, candidate: str, threshold_bps: int) -> int:
    """One whole-share long or flat target, O(window) precision-safe arithmetic."""
    if candidate not in CANDIDATES or len(prices) < 2 or any(p <= 0 for p in prices):
        raise FoundryError("strategy_contract", "Invalid causal strategy window.")
    mean = sum(prices, Decimal(0)) / len(prices)
    return _decision(prices[-1], mean, candidate, threshold_bps)


def _decision(last: Decimal, mean: Decimal, candidate: str, threshold: int) -> int:
    distance = (last / mean - 1) * 10000
    return int(
        distance > threshold if candidate == "momentum" else distance < -threshold
    )


def daily_pnl(
    bars: tuple[Bar, ...], plan: Plan, candidate: str, cost_bps: int
) -> dict[str, dict[str, float]]:
    """Replay one-share positions; overnight exposure is excluded by liquidation.

    O(n) time, O(days + window) memory; window <= 120 and n <= 20,000.
    Gaps liquidate at the last observed close with explicit costs. That convention
    is a diagnostic bound, not a claim such an exit was available in real time.
    """
    days: dict[str, dict[str, float]] = {}
    window: deque[Decimal] = deque(maxlen=plan.window)
    rolling_sum = Decimal(0)
    previous: Bar | None = None
    position = 0
    fee = Decimal(cost_bps) / 10000
    for bar in bars:
        day = bar.at.astimezone(NEW_YORK).date().isoformat()
        row = days.setdefault(
            day,
            {"gross": 0.0, "cost": 0.0, "turnover": 0.0, "bars": 0.0, "baseline": 0.0},
        )
        contiguous = (
            previous is not None
            and bar.at - previous.at == timedelta(minutes=1)
            and day == previous.at.astimezone(NEW_YORK).date().isoformat()
        )
        if not contiguous:
            if previous is not None and position:
                prior_day = previous.at.astimezone(NEW_YORK).date().isoformat()
                days[prior_day]["cost"] += float(previous.close * fee)
                days[prior_day]["turnover"] += float(previous.close)
            position = 0
            window.clear()
            rolling_sum = Decimal(0)
        else:
            assert previous is not None
            row["gross"] += position * float(bar.open - previous.close)
        proposed = (
            _decision(
                window[-1], rolling_sum / plan.window, candidate, plan.threshold_bps
            )
            if len(window) == plan.window
            else 0
        )
        turnover = abs(proposed - position) * bar.open
        row["cost"] += float(turnover * fee)
        row["turnover"] += float(turnover)
        row["gross"] += proposed * float(bar.close - bar.open)
        row["baseline"] += float(bar.close - bar.open)
        if contiguous and previous is not None:
            row["baseline"] += float(bar.open - previous.close)
        else:
            row["baseline"] -= float(bar.open * fee)
        # Baseline also closes at each gap/session boundary, below.
        if previous is not None and not contiguous:
            prior_day = previous.at.astimezone(NEW_YORK).date().isoformat()
            days[prior_day]["baseline"] -= float(previous.close * fee)
        position = proposed
        if len(window) == plan.window:
            rolling_sum -= window[0]
        rolling_sum += bar.close
        window.append(bar.close)
        row["bars"] += 1
        previous = bar
    if previous is not None:
        day = previous.at.astimezone(NEW_YORK).date().isoformat()
        days[day]["cost"] += position * float(previous.close * fee)
        days[day]["turnover"] += position * float(previous.close)
        days[day]["baseline"] -= float(previous.close * fee)
    for row in days.values():
        row["net"] = row["gross"] - row["cost"]
    return days


def interval(values: list[float], seed: int) -> tuple[float, float] | None:
    """500 moving-block resamples, five-day blocks, Bonferroni for two candidates.

    Twenty dates is a minimum for emitting an interval, not evidence of adequate
    statistical power. No independence assumption is made between adjacent days.
    """
    if len(values) < 20:
        return None
    if len(values) > 366 or not all(math.isfinite(value) for value in values):
        raise FoundryError("research_sample", "Invalid daily sample.")
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(500):
        sample: list[float] = []
        while len(sample) < len(values):
            start = rng.randrange(len(values) - 4)
            sample.extend(values[start : start + 5])
        means.append(fmean(sample[: len(values)]))
    means.sort()
    return means[6], means[493]


def evaluate(bars: tuple[Bar, ...], plan: Plan) -> dict[str, Any]:
    """Retain both candidates and cost stress; final data never selects a winner."""
    if not bars or len(bars) > 20000:
        raise FoundryError("research_sample", "Expected 1–20,000 intraday bars.")
    if any(a.at >= b.at for a, b in zip(bars, bars[1:], strict=False)):
        raise FoundryError("research_order", "Research bars must be strictly ordered.")
    if any(not plan.start <= b.at < plan.end for b in bars):
        raise FoundryError(
            "research_split", "Observation is outside the frozen interval."
        )
    # An entire exchange date belongs to one split. No within-day boundary leak.
    split = plan.selection_end.astimezone(NEW_YORK).date().isoformat()
    records: list[dict[str, Any]] = []
    selection: dict[str, float] = {}
    for candidate in CANDIDATES:
        for multiplier in (1, 2, 3):
            daily = daily_pnl(
                bars, plan, candidate, plan.cost_bps_per_side * multiplier
            )
            training = [row["net"] for day, row in daily.items() if day < split]
            heldout = [row["net"] for day, row in daily.items() if day >= split]
            excess = [
                row["net"] - row["baseline"]
                for day, row in daily.items()
                if day >= split
            ]
            if multiplier == 1:
                selection[candidate] = fmean(training) if training else -math.inf
            records.append(
                {
                    "candidate": candidate,
                    "cost_multiplier": multiplier,
                    "selection_days": len(training),
                    "test_days": len(heldout),
                    "test_mean_net_dollars_per_share": (
                        fmean(heldout) if heldout else None
                    ),
                    "net_interval": interval(heldout, plan.seed),
                    "excess_over_baseline_interval": interval(excess, plan.seed),
                    "daily": daily,
                }
            )
    winner = (
        max(CANDIDATES, key=lambda c: selection[c])
        if any(math.isfinite(v) for v in selection.values())
        else None
    )
    return {
        "schema_version": "intraday-diagnostics-1",
        "plan_identity": plan.identity,
        "selected_on_pretest_dates": winner,
        "records": records,
        "decision": "NO_GO",
        "broker_authorized": False,
        "limitations": [
            "OHLC/gap exits are diagnostic assumptions, not observed fills.",
            "History is current-vintage; universe/revisions remain incomplete.",
            "Source qualification and execution/capacity evidence are required.",
            "Paper fills do not measure market impact or queue position.",
        ],
    }
