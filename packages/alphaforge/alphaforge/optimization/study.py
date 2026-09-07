"""Reproducible, aggregate-only evidence publication for SF-S4-MR2.

The publisher deliberately owns its synthetic development data. It has no
argument through which a protected holdout, vendor dataset, credential, order,
or fill can enter. The configured holdout size is recorded as a reservation
only; those observations are never generated. Consequently this module can
exercise optimizer mechanics and attribution reconciliation, but it cannot
select a trading candidate or support a profit or deployment claim.

Every run is bounded by schema-level dimensions, an estimated solver-call
ceiling, phase deadline checks, artifact-count and byte ceilings, and an atomic
no-overwrite directory publication. The manifest hashes the exact UTF-8 YAML
bytes and every published payload artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alphaforge.optimization.evidence import (
    compare_markowitz_variants,
    sensitivity_to_input_error,
)
from alphaforge.optimization.mean_variance import (
    Formulation,
    MeanVarianceProblem,
    OptimizerError,
    solve_mean_variance,
)
from alphaforge.optimization.risk_attribution import (
    attribute_ex_ante_risk,
    attribute_exposure_drift,
    attribute_realized_performance,
    evaluate_return_scenarios,
)
from alphaforge.optimization.risk_model import estimate_factor_risk_model
from alphaforge.portfolio.contracts import PortfolioConstraints
from alphaforge.portfolio.evidence import (
    BacktestPanel,
    chronological_folds,
    volatility_regimes,
)

STUDY_SCHEMA_VERSION = "1.0.0"
EXPECTED_FORMULATIONS: tuple[Formulation, ...] = (
    "minimum_variance",
    "target_return",
    "maximum_utility",
    "alpha_risk_cost",
)
EXPECTED_COMPARISON_ARMS: frozenset[str] = frozenset(
    {
        *EXPECTED_FORMULATIONS,
        "equal_weight",
        "inverse_volatility",
        "rank_weighted",
        "rank_uncertainty_vol_target",
        "no_trade",
    }
)

MAX_CONFIG_BYTES = 64 * 1024
MAX_SYNTHETIC_OBSERVATIONS = 2_000
MAX_FRAME_ROWS = 50_000
MAX_FRAME_COLUMNS = 128
MIN_OUTPUT_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 50 * 1024 * 1024

BoundedSeed = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
PositiveFinite = Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]


class MeanVarianceStudyError(ValueError):
    """Raised when a study input or publication invariant is violated."""


class _StrictModel(BaseModel):
    """Shared immutable, deny-unknown configuration boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SyntheticDevelopmentConfig(_StrictModel):
    """Bounded synthetic market whose final reserved interval is not generated."""

    seed: BoundedSeed
    start_date: date
    n_assets: Annotated[int, Field(strict=True, ge=6, le=64)]
    n_factors: Annotated[int, Field(strict=True, ge=1, le=8)]
    history_observations: Annotated[int, Field(strict=True, ge=260, le=1_500)]
    development_observations: Annotated[int, Field(strict=True, ge=12, le=250)]
    reserved_holdout_observations: Annotated[int, Field(strict=True, ge=1, le=500)]
    periods_per_year: Annotated[int, Field(strict=True, ge=1, le=366)] = 252
    factor_volatility: Annotated[float, Field(strict=True, gt=0.0, le=0.10, allow_inf_nan=False)]
    specific_volatility: Annotated[float, Field(strict=True, gt=0.0, le=0.10, allow_inf_nan=False)]
    signal_volatility: Annotated[float, Field(strict=True, gt=0.0, le=0.05, allow_inf_nan=False)]
    signal_persistence: Annotated[float, Field(strict=True, ge=-0.95, le=0.95, allow_inf_nan=False)]
    planted_signal_strength: Annotated[
        float, Field(strict=True, ge=0.0, le=2.0, allow_inf_nan=False)
    ]
    average_daily_volume_dollars: Annotated[
        float, Field(strict=True, gt=1_000.0, le=1.0e12, allow_inf_nan=False)
    ]

    @model_validator(mode="after")
    def _validate_dimensions(self) -> SyntheticDevelopmentConfig:
        if self.n_factors >= self.n_assets:
            raise ValueError("n_factors must be smaller than n_assets")
        generated = self.history_observations + self.development_observations
        if generated > MAX_SYNTHETIC_OBSERVATIONS:
            raise ValueError(
                f"generated observations exceed the {MAX_SYNTHETIC_OBSERVATIONS}-row ceiling"
            )
        return self


class PortfolioConstraintConfig(_StrictModel):
    """Serializable subset of the portfolio-constraint contract."""

    max_position: Annotated[float, Field(strict=True, gt=0.0, le=10.0, allow_inf_nan=False)]
    max_gross: Annotated[float, Field(strict=True, gt=0.0, le=10.0, allow_inf_nan=False)]
    max_net: Annotated[float, Field(strict=True, ge=0.0, le=10.0, allow_inf_nan=False)]
    max_leverage: Annotated[float, Field(strict=True, gt=0.0, le=10.0, allow_inf_nan=False)]
    cash_buffer: Annotated[float, Field(strict=True, ge=0.0, lt=1.0, allow_inf_nan=False)]
    long_only: bool

    @model_validator(mode="after")
    def _validate_relationships(self) -> PortfolioConstraintConfig:
        self.to_domain()
        return self

    def to_domain(self) -> PortfolioConstraints:
        """Return the immutable domain constraint record."""
        return PortfolioConstraints(
            max_position=self.max_position,
            max_gross=self.max_gross,
            max_net=self.max_net,
            max_turnover=None,
            max_leverage=self.max_leverage,
            cash_buffer=self.cash_buffer,
            long_only=self.long_only,
        )


class EvidenceGridConfig(_StrictModel):
    """Complete development-only comparison and perturbation grid."""

    folds: Annotated[int, Field(strict=True, ge=1, le=10)]
    capital_levels: tuple[PositiveFinite, ...]
    turnover_budgets: tuple[
        Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)] | None, ...
    ]
    formulations: tuple[Formulation, ...]
    perturbations: tuple[
        Annotated[float, Field(strict=True, ge=0.0, le=2.0, allow_inf_nan=False)], ...
    ]
    sensitivity_formulation: Literal["maximum_utility"]
    cost_bps: Annotated[float, Field(strict=True, ge=10.0, le=10.0, allow_inf_nan=False)]
    risk_aversion: Annotated[float, Field(strict=True, gt=0.0, le=1.0e6, allow_inf_nan=False)]
    budget: Annotated[float, Field(strict=True, gt=0.0, le=10.0, allow_inf_nan=False)]
    target_return: Annotated[float, Field(strict=True, gt=0.0, le=0.10, allow_inf_nan=False)]
    participation: Annotated[float, Field(strict=True, gt=0.0, le=1.0, allow_inf_nan=False)]
    uncertainty_strength: Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
    volatility_target: Annotated[float, Field(strict=True, gt=0.0, le=5.0, allow_inf_nan=False)]
    regime_window: Annotated[int, Field(strict=True, ge=2, le=252)]

    @field_validator("capital_levels")
    @classmethod
    def _validate_capital_levels(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not 1 <= len(value) <= 4:
            raise ValueError("capital_levels must hold 1..4 entries")
        if len(set(value)) != len(value):
            raise ValueError("capital_levels must be unique")
        return value

    @field_validator("turnover_budgets")
    @classmethod
    def _validate_turnover_budgets(
        cls, value: tuple[float | None, ...]
    ) -> tuple[float | None, ...]:
        if not 1 <= len(value) <= 4:
            raise ValueError("turnover_budgets must hold 1..4 entries")
        if len(set(value)) != len(value):
            raise ValueError("turnover_budgets must be unique")
        return value

    @field_validator("formulations")
    @classmethod
    def _validate_formulations(cls, value: tuple[Formulation, ...]) -> tuple[Formulation, ...]:
        if value != EXPECTED_FORMULATIONS:
            raise ValueError(
                "formulations must contain every MR2 formulation once in the frozen order"
            )
        return value

    @field_validator("perturbations")
    @classmethod
    def _validate_perturbations(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not 2 <= len(value) <= 8:
            raise ValueError("perturbations must hold 2..8 entries")
        if tuple(sorted(value)) != value or len(set(value)) != len(value):
            raise ValueError("perturbations must be sorted and unique")
        if value[0] != 0.0:
            raise ValueError("perturbations must begin with the unperturbed 0.0 reference")
        return value


class AttributionStudyConfig(_StrictModel):
    """Reference-book attribution and bounded scenario settings."""

    initial_capital: PositiveFinite
    realized_cost_return: Annotated[float, Field(strict=True, ge=0.0, le=0.10, allow_inf_nan=False)]
    factor_window: Annotated[int, Field(strict=True, ge=10, le=1_000)]
    factor_shrinkage: Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
    specific_shrinkage: Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
    broad_drawdown_return: Annotated[
        float, Field(strict=True, ge=-0.99, lt=0.0, allow_inf_nan=False)
    ]
    broad_rally_return: Annotated[float, Field(strict=True, gt=0.0, le=2.0, allow_inf_nan=False)]
    factor_rotation_magnitude: Annotated[
        float, Field(strict=True, gt=0.0, le=1.0, allow_inf_nan=False)
    ]


class PublicationBoundsConfig(_StrictModel):
    """Fail-closed publication resource ceilings."""

    max_solver_calls: Annotated[int, Field(strict=True, ge=1, le=4_096)]
    max_runtime_seconds: Annotated[float, Field(strict=True, gt=0.0, le=600.0, allow_inf_nan=False)]
    max_artifacts: Annotated[int, Field(strict=True, ge=7, le=16)]
    max_output_bytes: Annotated[int, Field(strict=True, ge=MIN_OUTPUT_BYTES, le=MAX_OUTPUT_BYTES)]
    plot_dpi: Annotated[int, Field(strict=True, ge=72, le=300)]


class MeanVarianceStudyConfig(_StrictModel):
    """Frozen Pydantic schema for one MR2 engineering study."""

    schema_version: Literal["1.0.0"]
    profile_id: Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-z0-9-]+$")]
    scope: Literal["synthetic_development_only"]
    synthetic: SyntheticDevelopmentConfig
    constraints: PortfolioConstraintConfig
    evidence: EvidenceGridConfig
    attribution: AttributionStudyConfig
    publication: PublicationBoundsConfig

    @model_validator(mode="after")
    def _validate_cross_section(self) -> MeanVarianceStudyConfig:
        evaluations = self.synthetic.development_observations - 1
        if self.evidence.folds > evaluations:
            raise ValueError("fold count exceeds available development evaluation dates")
        generated = self.synthetic.history_observations + self.synthetic.development_observations
        if self.evidence.regime_window > generated:
            raise ValueError("regime_window exceeds generated development history")
        if self.attribution.factor_window > self.synthetic.history_observations:
            raise ValueError("factor_window exceeds pre-development history")
        if self.constraints.max_position * self.synthetic.n_assets < self.evidence.budget:
            raise ValueError("position caps cannot reach the configured budget")
        effective_gross = min(
            self.constraints.max_gross,
            self.constraints.max_leverage,
        ) * (1.0 - self.constraints.cash_buffer)
        if self.evidence.budget > min(effective_gross, self.constraints.max_net):
            raise ValueError("budget exceeds a configured portfolio exposure limit")
        estimated_calls = _estimated_solver_calls(self)
        if estimated_calls > self.publication.max_solver_calls:
            raise ValueError(
                f"study requires {estimated_calls} solver calls, above max_solver_calls "
                f"{self.publication.max_solver_calls}"
            )
        return self


@dataclass(frozen=True)
class SyntheticDevelopmentData:
    """Generated development data; intentionally has no holdout field."""

    returns: pd.DataFrame
    panel: BacktestPanel
    factor_exposures: pd.DataFrame
    seed_map: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class AttributionEvidence:
    """Aggregate attribution table plus deterministic reference identities."""

    summary: pd.DataFrame
    solver_identity: str
    problem_identity: str
    asset_risk_model_identity: str
    factor_model_identity: str
    solver_name: str
    solver_version: str


@dataclass(frozen=True)
class MeanVarianceStudyResult:
    """Published path and content identities for one completed study."""

    output_dir: Path
    config_sha256: str
    manifest_sha256: str
    artifacts: tuple[tuple[str, str], ...]
    comparison_rows: int
    sensitivity_rows: int
    attribution_rows: int


@dataclass(frozen=True)
class _ConfigSnapshot:
    config: MeanVarianceStudyConfig
    source_sha256: str
    source_bytes: int


def _estimated_solver_calls(config: MeanVarianceStudyConfig) -> int:
    evaluations = config.synthetic.development_observations - 1
    comparison = (
        len(config.evidence.formulations)
        * len(config.evidence.capital_levels)
        * len(config.evidence.turnover_budgets)
        * evaluations
    )
    sensitivity = 2 * len(config.evidence.perturbations) * evaluations
    return comparison + sensitivity + 1


def _read_config_snapshot(path: str | Path) -> _ConfigSnapshot:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise MeanVarianceStudyError("study config must be a regular, non-symlink file")
    size = source.stat().st_size
    if not 1 <= size <= MAX_CONFIG_BYTES:
        raise MeanVarianceStudyError(f"study config size must lie in [1, {MAX_CONFIG_BYTES}] bytes")
    payload = source.read_bytes()
    if len(payload) != size:
        raise MeanVarianceStudyError("study config changed while it was being read")
    try:
        document = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise MeanVarianceStudyError("study config must be valid UTF-8 YAML") from exc
    if not isinstance(document, dict):
        raise MeanVarianceStudyError("study config root must be a mapping")
    config = MeanVarianceStudyConfig.model_validate(document)
    return _ConfigSnapshot(
        config=config,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        source_bytes=len(payload),
    )


def load_mean_variance_study_config(path: str | Path) -> MeanVarianceStudyConfig:
    """Load a bounded, immutable, deny-unknown study configuration."""
    return _read_config_snapshot(path).config


def _named_seed(root_seed: int, stream: str) -> int:
    digest = hashlib.sha256(
        f"alphaforge-mean-variance-study-v1:{root_seed}:{stream}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def build_synthetic_development_data(
    config: SyntheticDevelopmentConfig,
) -> SyntheticDevelopmentData:
    """Generate deterministic, redistribution-safe development data only.

    The configured holdout count does not influence any random stream or array
    dimension. Changing that reservation cannot change development evidence,
    and no holdout row exists to access.
    """
    stream_names = (
        "factor_loadings",
        "factor_returns",
        "specific_returns",
        "predictive_signal",
        "liquidity",
    )
    seeds = {name: _named_seed(config.seed, name) for name in stream_names}
    generators = {name: np.random.default_rng(seed) for name, seed in seeds.items()}
    observations = config.history_observations + config.development_observations
    assets = tuple(f"SYNTH_{index:02d}" for index in range(config.n_assets))
    factors = tuple(f"factor_{index:02d}" for index in range(config.n_factors))
    dates = pd.bdate_range(config.start_date, periods=observations)

    loadings = generators["factor_loadings"].normal(
        0.0, 0.65, size=(config.n_assets, config.n_factors)
    )
    factor_returns = generators["factor_returns"].normal(
        0.0, config.factor_volatility, size=(observations, config.n_factors)
    )
    specific_returns = generators["specific_returns"].normal(
        0.0, config.specific_volatility, size=(observations, config.n_assets)
    )
    innovations = generators["predictive_signal"].normal(
        0.0, config.signal_volatility, size=(observations, config.n_assets)
    )
    signal = np.empty_like(innovations)
    signal[0] = innovations[0]
    innovation_scale = float(np.sqrt(1.0 - config.signal_persistence**2))
    for row in range(1, observations):
        signal[row] = (
            config.signal_persistence * signal[row - 1] + innovation_scale * innovations[row]
        )
    planted = np.vstack([np.zeros((1, config.n_assets)), signal[:-1]])
    values = (
        factor_returns @ loadings.T + specific_returns + config.planted_signal_strength * planted
    )
    if not np.isfinite(values).all() or (values < -1.0).any():
        raise MeanVarianceStudyError("synthetic return construction violated simple-return bounds")
    returns = pd.DataFrame(values, index=dates, columns=assets, dtype=np.float64)

    evaluation_positions = np.arange(
        config.history_observations,
        observations - 1,
        dtype=np.int64,
    )
    evaluation_dates = dates[evaluation_positions]
    trailing = returns.rolling(63, min_periods=20).std(ddof=1)
    volatility = trailing.loc[evaluation_dates]
    if not np.isfinite(volatility.to_numpy(dtype=np.float64)).all():
        raise MeanVarianceStudyError("synthetic volatility warm-up did not produce finite values")
    liquidity_scale = generators["liquidity"].uniform(0.75, 1.25, config.n_assets)
    adv = pd.DataFrame(
        np.broadcast_to(
            config.average_daily_volume_dollars * liquidity_scale,
            (len(evaluation_dates), config.n_assets),
        ).copy(),
        index=evaluation_dates,
        columns=assets,
        dtype=np.float64,
    )
    panel = BacktestPanel(
        scores=pd.DataFrame(
            signal[evaluation_positions],
            index=evaluation_dates,
            columns=assets,
            dtype=np.float64,
        ),
        forward_returns=pd.DataFrame(
            values[evaluation_positions + 1],
            index=evaluation_dates,
            columns=assets,
            dtype=np.float64,
        ),
        volatility=volatility,
        adv=adv,
    )
    exposures = pd.DataFrame(loadings, index=assets, columns=factors, dtype=np.float64)
    return SyntheticDevelopmentData(
        returns=returns,
        panel=panel,
        factor_exposures=exposures,
        seed_map=tuple(sorted(seeds.items())),
    )


def _check_deadline(started: float, config: MeanVarianceStudyConfig, phase: str) -> None:
    elapsed = time.monotonic() - started
    if elapsed > config.publication.max_runtime_seconds:
        raise MeanVarianceStudyError(
            f"study exceeded its {config.publication.max_runtime_seconds:g}-second "
            f"runtime ceiling after {phase}"
        )


def _run_comparison(
    data: SyntheticDevelopmentData,
    config: MeanVarianceStudyConfig,
) -> pd.DataFrame:
    regimes = volatility_regimes(
        data.returns,
        window=config.evidence.regime_window,
    ).reindex(data.panel.scores.index)
    comparison = compare_markowitz_variants(
        data.panel,
        data.returns,
        config.constraints.to_domain(),
        folds=chronological_folds(data.panel.scores.index, n_folds=config.evidence.folds),
        capital_levels=config.evidence.capital_levels,
        turnover_budgets=config.evidence.turnover_budgets,
        formulations=cast(tuple[Formulation, ...], config.evidence.formulations),
        cost_bps=config.evidence.cost_bps,
        risk_aversion=config.evidence.risk_aversion,
        budget=config.evidence.budget,
        target_return=config.evidence.target_return,
        regimes=regimes,
        uncertainty_strength=config.evidence.uncertainty_strength,
        volatility_target=config.evidence.volatility_target,
        participation=config.evidence.participation,
        include_no_trade=True,
    )
    observed = frozenset(comparison["arm"].astype(str)) if not comparison.empty else frozenset()
    if observed != EXPECTED_COMPARISON_ARMS:
        raise MeanVarianceStudyError(
            f"comparison arm coverage mismatch: expected {sorted(EXPECTED_COMPARISON_ARMS)}, "
            f"observed {sorted(observed)}"
        )
    return comparison


def _run_sensitivity(
    data: SyntheticDevelopmentData,
    config: MeanVarianceStudyConfig,
) -> pd.DataFrame:
    result = sensitivity_to_input_error(
        data.panel,
        data.returns,
        config.constraints.to_domain(),
        capital=config.evidence.capital_levels[0],
        perturbations=config.evidence.perturbations,
        seed=config.synthetic.seed,
        formulation=config.evidence.sensitivity_formulation,
        budget=config.evidence.budget,
        risk_aversion=config.evidence.risk_aversion,
    )
    observed = tuple(result["alpha_error"].to_numpy(dtype=np.float64))
    if observed != config.evidence.perturbations:
        raise MeanVarianceStudyError(
            "sensitivity output does not cover the frozen perturbation grid"
        )
    return result


def _attribution_row(
    section: str,
    name: str,
    metric: str,
    value: float,
    unit: str,
) -> dict[str, str | float]:
    if not np.isfinite(value):
        raise MeanVarianceStudyError(f"non-finite attribution value for {section}/{name}/{metric}")
    return {
        "section": section,
        "name": name,
        "metric": metric,
        "value": float(value),
        "unit": unit,
        "scope": "synthetic_development_only",
    }


def _run_attribution(
    data: SyntheticDevelopmentData,
    config: MeanVarianceStudyConfig,
) -> AttributionEvidence:
    decision_date = pd.Timestamp(data.panel.scores.index[-1])
    factor_model = estimate_factor_risk_model(
        data.returns,
        data.factor_exposures,
        as_of=decision_date,
        exposure_vintage=decision_date,
        exposure_available_at=decision_date,
        window=config.attribution.factor_window,
        factor_shrinkage=config.attribution.factor_shrinkage,
        specific_shrinkage=config.attribution.specific_shrinkage,
        periods_per_year=config.synthetic.periods_per_year,
    )
    problem = MeanVarianceProblem(
        risk_model=factor_model.risk_model,
        constraints=config.constraints.to_domain(),
        budget=config.evidence.budget,
        decision_timestamp=decision_date,
    )
    solved = solve_mean_variance(problem, formulation="minimum_variance")
    if solved.status != "optimal" or not solved.audit_passed or not solved.kkt_passed:
        raise OptimizerError(
            "attribution reference requires one independently certified minimum-variance book"
        )
    weights = solved.weights
    ex_ante = attribute_ex_ante_risk(factor_model, weights)
    decision_position = data.returns.index.get_loc(decision_date)
    if not isinstance(decision_position, int):
        raise MeanVarianceStudyError("attribution decision date is not unique")
    realized_returns = data.returns.iloc[decision_position + 1].reindex(weights.index)
    capital = config.attribution.initial_capital
    cost_return = config.attribution.realized_cost_return
    cash_weight = float(1.0 - math.fsum(float(value) for value in weights.to_numpy()))
    cash_return = 0.0
    # This caller-owned ledger is deliberately independent of the attribution
    # implementation. The boundary must reject a contribution model that does
    # not reproduce the observed ending value supplied here.
    observed_ending_equity = math.fsum(
        [
            *(
                capital * float(weight) * (1.0 + float(asset_return))
                for weight, asset_return in zip(
                    weights.to_numpy(),
                    realized_returns.to_numpy(),
                    strict=True,
                )
            ),
            capital * cash_weight * (1.0 + cash_return),
            -capital * cost_return,
        ]
    )
    realized = attribute_realized_performance(
        weights,
        realized_returns,
        initial_capital=capital,
        observed_ending_equity=observed_ending_equity,
        cash_weight=cash_weight,
        cash_return=cash_return,
        cost_return=cost_return,
    )
    drift = attribute_exposure_drift(
        factor_model,
        weights,
        realized_returns,
        cash_weight=cash_weight,
    )
    first_loading = data.factor_exposures.iloc[:, 0].reindex(weights.index)
    rotation = -config.attribution.factor_rotation_magnitude * np.sign(first_loading)
    scenarios = evaluate_return_scenarios(
        weights,
        {
            "broad_drawdown": pd.Series(
                config.attribution.broad_drawdown_return,
                index=weights.index,
                dtype=np.float64,
            ),
            "broad_rally": pd.Series(
                config.attribution.broad_rally_return,
                index=weights.index,
                dtype=np.float64,
            ),
            "factor_rotation": pd.Series(rotation, index=weights.index, dtype=np.float64),
        },
        initial_capital=capital,
        cash_weight=cash_weight,
        cash_return=cash_return,
        cost_return=cost_return,
    )

    rows: list[dict[str, str | float]] = [
        _attribution_row(
            "ex_ante",
            "portfolio",
            "periodic_variance",
            ex_ante.portfolio_variance,
            "squared_simple_return",
        ),
        _attribution_row(
            "ex_ante",
            "portfolio",
            "annualized_volatility",
            ex_ante.annualized_volatility,
            "annualized_decimal",
        ),
        _attribution_row(
            "reconciliation",
            "asset_risk",
            "variance_error",
            ex_ante.variance_reconciliation_error,
            "squared_simple_return",
        ),
        _attribution_row(
            "realized",
            "portfolio",
            "gross_return",
            realized.gross_return,
            "simple_return",
        ),
        _attribution_row(
            "realized", "portfolio", "net_return", realized.net_return, "simple_return"
        ),
        _attribution_row("realized", "portfolio", "net_pnl", realized.net_pnl, "currency_units"),
        _attribution_row(
            "realized",
            "portfolio",
            "observed_ending_equity",
            realized.observed_ending_equity,
            "currency_units",
        ),
        _attribution_row(
            "realized",
            "cash",
            "return_contribution",
            realized.cash_return_contribution,
            "simple_return",
        ),
        _attribution_row(
            "realized",
            "cost",
            "return_contribution",
            realized.cost_return_contribution,
            "simple_return",
        ),
        _attribution_row(
            "reconciliation",
            "realized",
            "return_error",
            realized.return_reconciliation_error,
            "simple_return",
        ),
        _attribution_row(
            "reconciliation",
            "realized",
            "ending_equity_error",
            realized.ending_equity_reconciliation_error,
            "currency_units",
        ),
        _attribution_row(
            "drift",
            "portfolio",
            "gross_exposure_change",
            drift.gross_drift,
            "weight",
        ),
        _attribution_row("drift", "portfolio", "net_exposure_change", drift.net_drift, "weight"),
        _attribution_row(
            "reconciliation",
            "drift",
            "accounting_error",
            drift.accounting_reconciliation_error,
            "weight",
        ),
    ]
    rows.extend(
        _attribution_row(
            "asset_risk",
            item.asset,
            "component_variance",
            item.component_variance,
            "squared_simple_return",
        )
        for item in ex_ante.asset_contributions
    )
    rows.extend(
        _attribution_row(
            "factor_risk",
            item.factor,
            "component_variance",
            item.component_variance,
            "squared_simple_return",
        )
        for item in ex_ante.factor_contributions
    )
    rows.extend(
        _attribution_row(
            "factor_drift", item.factor, "exposure_change", item.drift, "factor_exposure"
        )
        for item in drift.factor_drifts
    )
    for scenario in scenarios.results:
        rows.extend(
            (
                _attribution_row(
                    "scenario",
                    scenario.name,
                    "modeled_portfolio_return",
                    scenario.modeled_portfolio_return,
                    "simple_return",
                ),
                _attribution_row(
                    "scenario",
                    scenario.name,
                    "modeled_scenario_pnl",
                    scenario.modeled_scenario_pnl,
                    "currency_units",
                ),
                _attribution_row(
                    "scenario",
                    scenario.name,
                    "modeled_ending_equity",
                    scenario.modeled_ending_equity,
                    "currency_units",
                ),
                _attribution_row(
                    "scenario_cash",
                    scenario.name,
                    "return_contribution",
                    scenario.cash_return_contribution,
                    "simple_return",
                ),
                _attribution_row(
                    "scenario_cost",
                    scenario.name,
                    "return_contribution",
                    scenario.cost_return_contribution,
                    "simple_return",
                ),
                _attribution_row(
                    "reconciliation",
                    scenario.name,
                    "scenario_return_error",
                    scenario.return_reconciliation_error,
                    "simple_return",
                ),
                _attribution_row(
                    "reconciliation",
                    scenario.name,
                    "scenario_ending_value_error",
                    scenario.ending_value_reconciliation_error,
                    "currency_units",
                ),
            )
        )
    summary = pd.DataFrame.from_records(rows).sort_values(
        ["section", "name", "metric"], ignore_index=True
    )
    return AttributionEvidence(
        summary=summary,
        solver_identity=solved.solver_identity,
        problem_identity=solved.problem_identity,
        asset_risk_model_identity=factor_model.risk_model.identity,
        factor_model_identity=factor_model.identity,
        solver_name=solved.solver_name,
        solver_version=solved.solver_version,
    )


def _plot_study(
    comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    destination: Path,
    *,
    seed: int,
    dpi: int,
) -> None:
    all_regimes = comparison.loc[comparison["regime"].eq("all")].copy()
    if all_regimes.empty:
        raise MeanVarianceStudyError("comparison has no all-regime rows to plot")
    count_keys = [
        column for column in ("arm", "capital", "turnover_budget") if column in all_regimes
    ]
    sample_counts = set(all_regimes.groupby(count_keys, dropna=False)["n_dates"].sum())
    if len(sample_counts) != 1:
        raise MeanVarianceStudyError("comparison grid cells do not share one sample count")
    development_dates = int(next(iter(sample_counts)))
    fold_count = int(all_regimes["fold"].nunique())
    sensitivity_long = pd.concat(
        [
            sensitivity.loc[:, ["alpha_error", "net_return"]]
            .rename(columns={"alpha_error": "input_error", "net_return": "annualized_return"})
            .assign(input="alpha"),
            sensitivity.loc[:, ["covariance_error", "covariance_net_return"]]
            .rename(
                columns={
                    "covariance_error": "input_error",
                    "covariance_net_return": "annualized_return",
                }
            )
            .assign(input="covariance"),
        ],
        ignore_index=True,
    )
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    figure, axes = plt.subplots(1, 4, figsize=(29, 7))
    sns.barplot(
        data=all_regimes,
        x="net_return",
        y="arm",
        hue="turnover_budget",
        errorbar=None,
        ax=axes[0],
    )
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set(
        title="Development-fold net return",
        xlabel="Annualized arithmetic return (decimal)",
        ylabel="Evidence arm",
    )
    axes[0].legend(
        title="Turnover budget",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
    )
    sns.barplot(
        data=all_regimes,
        x="cost_drag",
        y="arm",
        hue="turnover_budget",
        errorbar=None,
        ax=axes[1],
    )
    axes[1].set(
        title="Modelled transaction-cost drag",
        xlabel="Annualized cost drag (decimal)",
        ylabel="Evidence arm",
    )
    axes[1].legend().remove()
    sns.lineplot(
        data=sensitivity_long,
        x="input_error",
        y="annualized_return",
        hue="input",
        style="input",
        markers=True,
        dashes=False,
        errorbar=None,
        ax=axes[2],
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set(
        title="Input-error sensitivity",
        xlabel="Multiplicative error scale (fraction)",
        ylabel="Annualized arithmetic return (decimal)",
    )
    sns.barplot(
        data=all_regimes,
        x="feasible_fraction",
        y="arm",
        hue="turnover_budget",
        errorbar=None,
        ax=axes[3],
    )
    axes[3].axvline(1.0, color="black", linewidth=1)
    axes[3].set_xlim(0.0, 1.02)
    axes[3].set(
        title="Feasible evaluation coverage",
        xlabel="Feasible fraction (0–1; failures retained)",
        ylabel="Evidence arm",
    )
    axes[3].legend().remove()
    figure.suptitle(
        "SF-S4-MR2 synthetic development evidence — "
        f"n={development_dates} dates; {fold_count} folds; seed={seed}; holdout inaccessible",
        fontsize=18,
    )
    figure.text(
        0.5,
        0.005,
        "Synthetic engineering scope only; returns with incomplete feasible coverage are not comparable "
        "without the coverage panel; no candidate selection, profit, paper-trading, or live-trading claim.",
        ha="center",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    figure.savefig(
        destination,
        dpi=dpi,
        bbox_inches="tight",
        metadata={"Software": "AlphaForge deterministic evidence publisher"},
    )
    plt.close(figure)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MeanVarianceStudyError("JSON evidence must be finite and serializable") from exc
    return (encoded + "\n").encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"artifact already exists: {path}")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_bytes(path, _json_bytes(payload))


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    if len(frame) > MAX_FRAME_ROWS or len(frame.columns) > MAX_FRAME_COLUMNS:
        raise MeanVarianceStudyError(
            f"artifact frame exceeds {MAX_FRAME_ROWS} rows or {MAX_FRAME_COLUMNS} columns"
        )
    encoded = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    ).encode("utf-8")
    _write_bytes(path, encoded)


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise MeanVarianceStudyError(f"artifact is not a regular file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_records(staging: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    total = 0
    for path in sorted(staging.iterdir(), key=lambda item: item.name):
        if path.name == "manifest.json":
            continue
        if path.is_symlink() or not path.is_file():
            raise MeanVarianceStudyError("publication staging may contain regular files only")
        size = path.stat().st_size
        total += size
        records.append({"path": path.name, "bytes": size, "sha256": _sha256_file(path)})
    return records, total


def _summary_payload(
    snapshot: _ConfigSnapshot,
    data: SyntheticDevelopmentData,
    comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    attribution: AttributionEvidence,
) -> dict[str, Any]:
    config = snapshot.config
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "profile_id": config.profile_id,
        "scope": {
            "data": "deterministic_redistribution_safe_synthetic",
            "development_folds_only": True,
            "holdout_accessible": False,
            "holdout_observations_generated": 0,
            "holdout_observations_reserved": config.synthetic.reserved_holdout_observations,
            "candidate_selected": False,
            "profit_claim": False,
            "paper_or_live_readiness_claim": False,
        },
        "configuration": {
            "exact_source_sha256": snapshot.source_sha256,
            "exact_source_bytes": snapshot.source_bytes,
            "estimated_solver_calls": _estimated_solver_calls(config),
        },
        "data": {
            "assets": config.synthetic.n_assets,
            "factors": config.synthetic.n_factors,
            "history_observations": config.synthetic.history_observations,
            "development_observations": config.synthetic.development_observations,
            "evaluation_dates": len(data.panel.scores),
            "start_date": str(data.returns.index.min().date()),
            "development_end_date": str(data.returns.index.max().date()),
            "raw_rows_published": 0,
            "seed_map": dict(data.seed_map),
        },
        "evidence": {
            "comparison_arms": sorted(set(comparison["arm"].astype(str))),
            "comparison_rows": len(comparison),
            "folds": config.evidence.folds,
            "sensitivity_levels": list(config.evidence.perturbations),
            "sensitivity_rows": len(sensitivity),
            "attribution_rows": len(attribution.summary),
            "row_level_returns_published": 0,
            "row_level_predictions_published": 0,
        },
        "attribution_reference": {
            "book": "certified_minimum_variance_synthetic_reference",
            "problem_identity": attribution.problem_identity,
            "solver_identity": attribution.solver_identity,
            "asset_risk_model_identity": attribution.asset_risk_model_identity,
            "factor_model_identity": attribution.factor_model_identity,
            "solver_name": attribution.solver_name,
            "solver_version": attribution.solver_version,
        },
        "publication_bounds": config.publication.model_dump(mode="json"),
        "limitations": [
            "All observations are deterministic synthetic engineering data with a planted signal.",
            "Only development folds are generated and evaluated; the reserved holdout is inaccessible.",
            "No arm is selected, promoted, or represented as an investable candidate.",
            "Historical or synthetic performance does not imply profit and cannot qualify paper or live trading.",
            "Costs are the bounded MR2 research model, not order-, fill-, tax-, borrow-, or broker-level reconciliation.",
        ],
    }


def publish_mean_variance_study(
    config_path: str | Path,
    output_dir: str | Path,
) -> MeanVarianceStudyResult:
    """Run and atomically publish one frozen synthetic development study.

    Args:
        config_path: Regular UTF-8 YAML file validated by the strict schema.
        output_dir: New directory. Existing destinations and staging paths are
            refused rather than overwritten.

    Returns:
        Content identities and aggregate row counts for the completed output.

    Raises:
        MeanVarianceStudyError: On violated schema, evidence, resource, or
            publication invariants.
        FileExistsError: When output or staging already exists.
    """
    snapshot = _read_config_snapshot(config_path)
    config = snapshot.config
    destination = Path(output_dir)
    if not destination.name or destination.exists() or destination.is_symlink():
        raise FileExistsError(f"study output already exists or is invalid: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".publishing-{destination.name}"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"study staging already exists: {staging}")
    staging.mkdir(mode=0o700)
    started = time.monotonic()
    try:
        data = build_synthetic_development_data(config.synthetic)
        _check_deadline(started, config, "synthetic data generation")
        comparison = _run_comparison(data, config)
        _check_deadline(started, config, "comparison grid")
        sensitivity = _run_sensitivity(data, config)
        _check_deadline(started, config, "sensitivity grid")
        attribution = _run_attribution(data, config)
        _check_deadline(started, config, "attribution")

        _write_frame(staging / "comparison.csv", comparison)
        _write_frame(staging / "sensitivity.csv", sensitivity)
        _write_frame(staging / "attribution_summary.csv", attribution.summary)
        _write_json(staging / "resolved_config.json", config.model_dump(mode="json"))
        _plot_study(
            comparison,
            sensitivity,
            staging / "mean_variance_evidence.png",
            seed=config.synthetic.seed,
            dpi=config.publication.plot_dpi,
        )
        _check_deadline(started, config, "plot rendering")
        _write_json(
            staging / "summary.json",
            _summary_payload(snapshot, data, comparison, sensitivity, attribution),
        )

        records, payload_bytes = _artifact_records(staging)
        if len(records) + 1 > config.publication.max_artifacts:
            raise MeanVarianceStudyError("artifact count exceeds the configured ceiling")
        manifest = {
            "schema_version": STUDY_SCHEMA_VERSION,
            "profile_id": config.profile_id,
            "config_source": {
                "bytes": snapshot.source_bytes,
                "sha256": snapshot.source_sha256,
            },
            "artifacts": records,
            "payload_bytes_excluding_manifest": payload_bytes,
            "publication": {
                "atomic_directory_rename": True,
                "overwrite_allowed": False,
                "synthetic_only": True,
                "development_only": True,
                "holdout_accessible": False,
            },
        }
        _write_json(staging / "manifest.json", manifest)
        total_bytes = payload_bytes + (staging / "manifest.json").stat().st_size
        if total_bytes > config.publication.max_output_bytes:
            raise MeanVarianceStudyError(
                f"publication requires {total_bytes} bytes, above max_output_bytes "
                f"{config.publication.max_output_bytes}"
            )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"study output appeared during publication: {destination}")
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    artifact_pairs = tuple((str(item["path"]), str(item["sha256"])) for item in records)
    return MeanVarianceStudyResult(
        output_dir=destination,
        config_sha256=snapshot.source_sha256,
        manifest_sha256=_sha256_file(destination / "manifest.json"),
        artifacts=artifact_pairs,
        comparison_rows=len(comparison),
        sensitivity_rows=len(sensitivity),
        attribution_rows=len(attribution.summary),
    )
