# Signalattice — developer convenience targets.
# Run `make help` for the list.

.DEFAULT_GOAL := help
PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
CONFIG ?= configs/example.yaml

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.PHONY: venv
venv: ## Create a virtual environment in .venv
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

.PHONY: install
install: ## Install the package (editable) with dev extras
	$(BIN)/python -m pip install -e ".[dev]"

.PHONY: install-all
install-all: ## Install every optional integration, including PyTorch
	$(BIN)/python -m pip install -e ".[dev,data,boost,torch,mlflow]"

.PHONY: lint
lint: ## Run ruff + black --check
	$(BIN)/python -m ruff check src tests scripts
	$(BIN)/python -m black --check src tests scripts

.PHONY: format
format: ## Auto-format with ruff --fix and black
	$(BIN)/python -m ruff check --fix src tests scripts
	$(BIN)/python -m black src tests scripts

.PHONY: typecheck
typecheck: ## Run mypy static type checks
	$(BIN)/python -m mypy src

.PHONY: quality
quality: lock-check lint typecheck ## Run all static quality gates

.PHONY: lock-check
lock-check: ## Verify the committed universal dependency resolution
	$(BIN)/uv lock --check

.PHONY: test
test: ## Run the test suite (skips network tests)
	$(BIN)/python -m pytest -m "not network"

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	$(BIN)/python -m pytest -m "not network" --cov=quant_platform --cov-branch \
		--cov-report=term-missing --cov-report=xml --cov-fail-under=80

.PHONY: build
build: ## Build the source distribution and wheel
	$(BIN)/python -m build
	$(BIN)/python scripts/verify_distributions.py --dist-dir dist

.PHONY: ci
ci: quality test-cov build ## Reproduce the local CI quality, test, and package gates

.PHONY: ingest
ingest: ## Ingest market data using $(CONFIG)
	$(BIN)/signalattice ingest-data --config $(CONFIG)

.PHONY: features
features: ## Build features using $(CONFIG)
	$(BIN)/signalattice build-features --config $(CONFIG)

.PHONY: train
train: ## Train models using $(CONFIG)
	$(BIN)/signalattice train-model --config $(CONFIG)

.PHONY: backtest
backtest: ## Run backtest using $(CONFIG)
	$(BIN)/signalattice run-backtest --config $(CONFIG)

.PHONY: report
report: ## Generate report using $(CONFIG)
	$(BIN)/signalattice generate-report --config $(CONFIG)

.PHONY: pipeline
pipeline: ## Run the full end-to-end pipeline using $(CONFIG)
	$(BIN)/signalattice run-full-pipeline --config $(CONFIG)

.PHONY: demo
demo: ## Run a fully offline demo (synthetic data) end-to-end
	$(BIN)/signalattice run-full-pipeline --config configs/synthetic.yaml

.PHONY: benchmark-feature-store
benchmark-feature-store: ## Regenerate synthetic feature-store JSON and Seaborn evidence
	$(BIN)/python scripts/benchmark_feature_store.py \
		--output-json docs/benchmarks/feature_store_2026-07-25.json \
		--output-plot docs/assets/feature_store_latency_2026-07-25.png \
		--output-example-manifest docs/examples/feature_store_manifest.json

.PHONY: benchmark-state-space
benchmark-state-space: ## Regenerate state-space/risk JSON and Seaborn evidence
	$(BIN)/python scripts/benchmark_state_space_baselines.py \
		--output-json docs/benchmarks/state_space_baselines_2026-07-26.json \
		--output-plot docs/assets/state_space_baselines_2026-07-26.png

.PHONY: benchmark-service-operability
benchmark-service-operability: ## Stage bounded-service JSON and Seaborn candidates for review
	@set -eu; \
	mkdir -p build; \
	service_evidence_run="$$(mktemp -d build/service-operability.XXXXXX)"; \
	$(BIN)/python scripts/benchmark_service_operability.py \
		--output "$${service_evidence_run}/candidate.json"; \
	$(BIN)/python scripts/plot_service_operability.py \
		--input "$${service_evidence_run}/candidate.json" \
		--output "$${service_evidence_run}/candidate.png"; \
	shasum -a 256 \
		"$${service_evidence_run}/candidate.json" \
		docs/benchmarks/service_operability_2026-09-06_patch1.json \
		"$${service_evidence_run}/candidate.png" \
		docs/assets/service_operability_2026-09-06_patch1.png; \
	echo "Review retained service-operability candidates in $${service_evidence_run}"; \
	echo "After review, publish only to new empty dated reference paths on a dedicated work branch; never overwrite the current references."

.PHONY: verify-service-operability
verify-service-operability: ## Verify service evidence, API bounds, telemetry, and container policy
	$(BIN)/python -m pytest -q \
		tests/test_service_admission.py \
		tests/test_service_middleware_limits.py \
		tests/test_service_telemetry_contracts.py \
		tests/test_service_telemetry_metrics.py \
		tests/test_service_telemetry_exporter.py \
		tests/test_service_telemetry_runtime.py \
		tests/test_service_operability_evidence.py \
		tests/test_service_container_contract.py

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-data
clean-data: ## Remove generated data, reports, experiments (keeps .gitkeep)
	find data -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	rm -rf reports/figures/*.png reports/*.md reports/*.html experiments *.sqlite 2>/dev/null || true

.PHONY: docker-build
docker-build: ## Build the Docker image
	docker build --pull -t signalattice:latest .

.PHONY: docker-demo
docker-demo: ## Run the synthetic demo inside Docker
	docker compose run --rm platform run-full-pipeline --config configs/synthetic.yaml

# ---------------------------------------------------------------------------
# Release machinery (SF-S5-SL-MR7)
#
# These are the authoritative entry points. `release-publish` is deliberately
# absent: publication is a separately authorized operation that runs only from
# the protected release workflow against origin/main, and a make target would
# put it one keystroke away from any developer shell.
# ---------------------------------------------------------------------------

RELEASE_STAGING ?= build/release

.PHONY: release-dry-run release-verify release-reproducible

release-dry-run: ## Build and verify a release candidate; publishes nothing
	$(BIN)/python scripts/release.py dry-run --staging $(RELEASE_STAGING)

release-verify: ## Independently verify a staged release candidate
	$(BIN)/python scripts/release.py verify --staging $(RELEASE_STAGING)

release-reproducible: ## Prove two clean builds of one commit are byte-identical
	$(BIN)/python scripts/check_release_reproducible.py

# ---------------------------------------------------------------------------
# Sprint 5 evidence dossier (SF-S5-SL-MR8)
# ---------------------------------------------------------------------------

.PHONY: sprint5-dossier sprint5-dossier-check

sprint5-dossier: ## Regenerate the Sprint 5 evidence index and figure
	$(BIN)/python scripts/build_sprint5_dossier.py
	$(BIN)/python scripts/plot_sprint5_evidence.py

sprint5-dossier-check: ## Recompute every evidence digest and refuse on drift
	$(BIN)/python scripts/build_sprint5_dossier.py --check
