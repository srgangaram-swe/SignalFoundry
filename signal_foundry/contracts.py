"""Immutable, bounded wire contracts shared by HTTP, workers and generated clients.

Numbers are finite, identifiers are path-segment safe, and unknown fields fail
closed. Every research outcome is development simulation, never an order or a
qualification decision. Bounds are interactive resource policy, not market limits.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Text = Annotated[str, Field(min_length=1, max_length=512)]
Scalar = Annotated[str, Field(max_length=128)] | float | int | bool | None


class Contract(BaseModel):
    """Frozen strict model; JSON tuples are accepted through JSON validation."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
        validate_default=True,
    )

    def canonical(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        ).encode()

    def digest(self) -> str:
        return hashlib.sha256(self.canonical()).hexdigest()


class Parameter(Contract):
    """One scalar hyperparameter; executable/path parameters are not supported."""

    name: Identifier
    value: (
        Annotated[float, Field(ge=-1e6, le=1e6)]
        | Annotated[int, Field(ge=-1_000_000, le=1_000_000)]
        | bool
        | Annotated[str, Field(max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")]
    )


class ModelChoice(Contract):
    name: Identifier = "ridge"
    parameters: tuple[Parameter, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def unique_parameters(self) -> Self:
        if len({item.name for item in self.parameters}) != len(self.parameters):
            raise ValueError("duplicate model parameter")
        return self


class DataChoice(Contract):
    kind: Literal["synthetic", "bundle"] = "synthetic"
    bundle_id: Digest | None = None
    benchmark: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9.-]{0,15}$")] = "BENCH"
    symbols: int = Field(default=8, ge=3, le=31)
    days: int = Field(default=500, ge=260, le=2000)

    @model_validator(mode="after")
    def source_identity(self) -> Self:
        if (self.kind == "bundle") != (self.bundle_id is not None):
            raise ValueError("only bundle data requires a bundle identity")
        if self.kind == "synthetic" and self.benchmark != "BENCH":
            raise ValueError("synthetic benchmark is BENCH")
        if (self.symbols + 1) * self.days > 40_000:
            raise ValueError("synthetic panel exceeds interactive row budget")
        return self


class FoldPolicy(Contract):
    scheme: Literal["expanding", "rolling"] = "expanding"
    train_days: int = Field(default=252, ge=126, le=1500)
    test_days: int = Field(default=63, ge=21, le=252)
    embargo_days: int = Field(default=5, ge=1, le=63)
    horizon: int = Field(default=1, ge=1, le=20)
    standardize: bool = True

    @model_validator(mode="after")
    def purged_horizon(self) -> Self:
        if self.embargo_days < self.horizon:
            raise ValueError("embargo must cover the target horizon")
        return self


class CostPolicy(Contract):
    commission_bps: float = Field(default=1.0, ge=0, le=100)
    half_spread_bps: float = Field(default=2.5, ge=0, le=100)
    slippage_bps: float = Field(default=2.0, ge=0, le=100)
    short_borrow_bps_annual: float = Field(default=1000.0, ge=0, le=10_000)
    cash_financing_bps_annual: float = Field(default=500.0, ge=0, le=10_000)
    execution_lag: int = Field(default=1, ge=1, le=5)
    rebalance_days: int = Field(default=5, ge=1, le=21)


class RiskPolicy(Contract):
    max_weight: float = Field(default=0.2, gt=0, le=1)
    max_gross: float = Field(default=1.0, gt=0, le=1)
    max_net: float = Field(default=1.0, ge=0, le=1)
    turnover_cap: float = Field(default=0.5, gt=0, le=2)
    inverse_volatility: bool = True
    drawdown_deleverage: float = Field(default=0.15, gt=0, le=0.5)
    participation_rate: float = Field(default=0.05, gt=0, le=0.1)

    @model_validator(mode="after")
    def coherent_exposure(self) -> Self:
        if self.max_weight > self.max_gross or self.max_net > self.max_gross:
            raise ValueError("name and net caps cannot exceed the gross cap")
        return self


class ResearchRequest(Contract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    feature_profile: Literal["technical_v1_no_hmm"] = "technical_v1_no_hmm"
    data: DataChoice = DataChoice()
    model: ModelChoice = ModelChoice()
    baselines: tuple[Identifier, ...] = Field(
        default=("historical_mean", "momentum_baseline"),
        min_length=1,
        max_length=3,
    )
    strategy: Identifier = "long_short"
    regime: Literal["unfiltered", "causal_volatility"] = "unfiltered"
    folds: FoldPolicy = FoldPolicy()
    costs: CostPolicy = CostPolicy()
    risk: RiskPolicy = RiskPolicy()
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def coherent_experiment(self) -> Self:
        names = (self.model.name, *self.baselines)
        if len(set(names)) != len(names):
            raise ValueError("model and baseline identities must be distinct")
        if self.data.kind == "synthetic" and self.data.days <= (
            self.folds.train_days
            + self.folds.embargo_days
            + self.folds.horizon
            + self.costs.rebalance_days
            + self.costs.execution_lag
            + 20
        ):
            raise ValueError("data leaves no untouched walk-forward test interval")
        return self


class Capability(Contract):
    name: Identifier
    available: bool
    reason: Text
    parameters: tuple[Identifier, ...] = Field(default=(), max_length=16)


class Dataset(Contract):
    bundle_id: Digest
    rows: int = Field(ge=1, le=40_000)
    symbols: tuple[str, ...] = Field(min_length=2, max_length=32)
    date_min: Annotated[str, Field(max_length=32)]
    date_max: Annotated[str, Field(max_length=32)]
    limitations: tuple[Text, ...] = Field(max_length=32)


class Catalog(Contract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    default_request: ResearchRequest = ResearchRequest()
    models: tuple[Capability, ...] = Field(max_length=64)
    strategies: tuple[Capability, ...] = Field(max_length=32)
    baselines: tuple[Identifier, ...] = Field(max_length=8)
    datasets: tuple[Dataset, ...] = Field(max_length=16)
    limitations: tuple[Text, ...] = Field(max_length=32)
    mode: Literal["development_simulation"] = "development_simulation"
    live_readiness: Literal["NOT_READY"] = "NOT_READY"


class Validation(Contract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    request_hash: Digest
    data_identity: Annotated[str, Field(max_length=128)]
    observations: int = Field(ge=1, le=40_000)
    symbols: int = Field(ge=2, le=32)
    sessions: int = Field(ge=1, le=2000)
    limitations: tuple[Text, ...] = Field(max_length=32)


class Column(Contract):
    name: Identifier
    unit: Annotated[str, Field(min_length=1, max_length=64)]


class EvidenceTable(Contract):
    name: Identifier
    description: Text
    columns: tuple[Column, ...] = Field(min_length=1, max_length=24)
    rows: tuple[tuple[Scalar, ...], ...] = Field(max_length=2048)
    total_rows: int = Field(ge=0, le=1_000_000)

    @model_validator(mode="after")
    def rectangular(self) -> Self:
        if len({column.name for column in self.columns}) != len(self.columns):
            raise ValueError("duplicate evidence column")
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("evidence rows do not align with columns")
        if self.total_rows < len(self.rows):
            raise ValueError("evidence row count contradicts truncation")
        return self


class SeedAssignment(Contract):
    name: Identifier
    value: int = Field(ge=0, le=2**32 - 1)


class ResearchEvidence(Contract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    request: ResearchRequest
    request_hash: Digest
    data_identity: Annotated[str, Field(max_length=128)]
    code_hash: Digest
    source_code_hash: Digest
    environment_hash: Digest
    seed_map: tuple[SeedAssignment, ...] = Field(min_length=1, max_length=64)
    tables: tuple[EvidenceTable, ...] = Field(min_length=1, max_length=24)
    limitations: tuple[Text, ...] = Field(max_length=32)
    mode: Literal["development_simulation"] = "development_simulation"
    live_readiness: Literal["NOT_READY"] = "NOT_READY"

    @model_validator(mode="after")
    def identities(self) -> Self:
        if self.request_hash != self.request.digest():
            raise ValueError("request identity mismatch")
        if len({table.name for table in self.tables}) != len(self.tables):
            raise ValueError("duplicate evidence table")
        return self


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Contract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    job_id: Digest
    request_hash: Digest
    state: JobState
    created_at: Annotated[str, Field(max_length=32)]
    updated_at: Annotated[str, Field(max_length=32)]
    error_code: Identifier | None = None
    evidence_hash: Digest | None = None

    @model_validator(mode="after")
    def coherent_state(self) -> Self:
        if (self.state == JobState.SUCCEEDED) != (self.evidence_hash is not None):
            raise ValueError("only succeeded jobs have published evidence")
        if (
            self.state in {JobState.QUEUED, JobState.RUNNING, JobState.SUCCEEDED}
            and self.error_code is not None
        ):
            raise ValueError("non-failure state cannot have an error")
        return self


class JobPage(Contract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    jobs: tuple[Job, ...] = Field(max_length=64)


class AuditEvent(Contract):
    sequence: int = Field(ge=1)
    state: JobState
    at: Annotated[str, Field(max_length=32)]
    code: Identifier | None = None


class AuditTrail(Contract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    job_id: Digest
    events: tuple[AuditEvent, ...] = Field(max_length=8)


class Comparison(Contract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    compatible: bool
    reason: Text
    evidence: tuple[ResearchEvidence, ResearchEvidence]


class Problem(Contract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    code: Identifier
    detail: Text
