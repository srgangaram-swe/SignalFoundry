"""Bounded distributed research execution (SF-S5-MR8).

- :mod:`~alphaforge.distributed.profiling` — measure the serial pipeline and the
  parallel fraction *before* distributing anything, with the Amdahl bound that
  caps achievable speedup regardless of worker count.
- :mod:`~alphaforge.distributed.tasks` — resource-declaring, content-addressed
  task specifications. Every task states CPU, RAM, GPU, scratch, duration, seed,
  input hash, timeout, retry budget, and cancellability.
- :mod:`~alphaforge.distributed.executor` — the local reference backend, a
  process-pool backend, and a parity check that refuses a backend which changes
  results.
- :mod:`~alphaforge.distributed.budgets` — hard resource limits admitted before
  a batch starts and enforced while it runs. No best-effort mode.
- :mod:`~alphaforge.distributed.checkpoints` — atomic versioned checkpoints that
  bind code, data, config, dependencies, seed, and task graph, and refuse to
  resume into a different world.

**Cluster access is never required for reproducibility.** The local backend is
always available and defines the correct answer; any other backend is an optional
accelerator that must match it exactly.
"""

from alphaforge.distributed.benchmark_evidence import (
    BENCHMARK_NAME,
    PRODUCTION_EXECUTION_PROFILE,
    TEST_EXECUTION_PROFILE,
    BenchmarkConfig,
    BenchmarkEnvironment,
    BenchmarkEvidence,
    BenchmarkEvidenceError,
    BenchmarkImplementation,
    BenchmarkSample,
    BenchmarkSummary,
    DistributionSummary,
    TaskGraphBinding,
    collect_benchmark_environment,
    load_benchmark_evidence,
    parse_benchmark_evidence_bytes,
    require_production_implementation,
    run_crossover_benchmark,
    summarize_benchmark,
    task_declaration_graph_sha256,
    verify_production_implementation_sources,
    write_benchmark_evidence,
)
from alphaforge.distributed.benchmark_evidence import (
    SCHEMA_VERSION as BENCHMARK_EVIDENCE_SCHEMA_VERSION,
)
from alphaforge.distributed.budgets import (
    AdmissionDecision,
    BudgetBreach,
    BudgetError,
    BudgetExceededError,
    ExperimentBudget,
    LimitKind,
    ResourceUsage,
    admit,
    breach_report,
    declared_usage,
    enforce,
)
from alphaforge.distributed.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    REQUIRED_BINDINGS,
    CheckpointCorruptError,
    CheckpointError,
    CheckpointIncompatibleError,
    CheckpointManifest,
    CheckpointStore,
    ConcurrentWriterError,
    manifest_from_payload,
    task_graph_hash,
    verify_resumable,
)
from alphaforge.distributed.executor import (
    MAX_TOTAL_SECONDS,
    MAX_WORKERS,
    BatchReport,
    ExecutionError,
    TaskOutcome,
    TaskResult,
    assert_backend_parity,
    execute_local,
    execute_process_pool,
)
from alphaforge.distributed.profiling import (
    MIN_USEFUL_PARALLEL_FRACTION,
    ProfilingError,
    SerialProfile,
    StageTiming,
    profile_stages,
)
from alphaforge.distributed.tasks import (
    MAX_RETRIES,
    MAX_TASKS_PER_BATCH,
    ResourceRequest,
    TaskContractError,
    TaskSpec,
    assert_unique_tasks,
    content_hash,
)

__all__ = [
    "AdmissionDecision",
    "BatchReport",
    "BENCHMARK_EVIDENCE_SCHEMA_VERSION",
    "BENCHMARK_NAME",
    "BenchmarkConfig",
    "BenchmarkEnvironment",
    "BenchmarkEvidence",
    "BenchmarkEvidenceError",
    "BenchmarkImplementation",
    "BenchmarkSample",
    "BenchmarkSummary",
    "BudgetBreach",
    "BudgetError",
    "BudgetExceededError",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointCorruptError",
    "CheckpointError",
    "CheckpointIncompatibleError",
    "CheckpointManifest",
    "CheckpointStore",
    "ConcurrentWriterError",
    "ExecutionError",
    "ExperimentBudget",
    "LimitKind",
    "MAX_RETRIES",
    "MAX_TASKS_PER_BATCH",
    "MAX_TOTAL_SECONDS",
    "MAX_WORKERS",
    "MIN_USEFUL_PARALLEL_FRACTION",
    "ProfilingError",
    "PRODUCTION_EXECUTION_PROFILE",
    "REQUIRED_BINDINGS",
    "ResourceRequest",
    "ResourceUsage",
    "DistributionSummary",
    "SerialProfile",
    "StageTiming",
    "TEST_EXECUTION_PROFILE",
    "TaskContractError",
    "TaskOutcome",
    "TaskResult",
    "TaskSpec",
    "TaskGraphBinding",
    "admit",
    "assert_backend_parity",
    "assert_unique_tasks",
    "breach_report",
    "content_hash",
    "declared_usage",
    "enforce",
    "execute_local",
    "execute_process_pool",
    "collect_benchmark_environment",
    "load_benchmark_evidence",
    "manifest_from_payload",
    "profile_stages",
    "parse_benchmark_evidence_bytes",
    "run_crossover_benchmark",
    "require_production_implementation",
    "summarize_benchmark",
    "task_declaration_graph_sha256",
    "task_graph_hash",
    "verify_resumable",
    "verify_production_implementation_sources",
    "write_benchmark_evidence",
]
