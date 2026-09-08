PYTHON ?= python
UV ?= uv
SIGNAL_FOUNDRY_CONFIG ?= configs/signal_foundry_research.yaml
SPRINT_3_DECISION_CONFIG ?= configs/sprint_3_decision.yaml
MEAN_VARIANCE_STUDY_CONFIG ?= configs/mean_variance_study.yaml

.PHONY: install install-all lock-check config-check test lint format typecheck policy check download-data build-features label-evidence temporal-evidence deep-sequence-evidence time-frequency-evidence latent-representation-evidence ensemble-evidence decision-policy-evidence sprint-3-decision-evidence sprint-5-benchmark sprint-5-evidence mean-variance-evidence signal-foundry-evidence train evaluate \
        walk-forward backtest signal-foundry paper dashboard api report demo docker-build clean \
        native bench bench-native bench-event bench-execution-frictions

install:
	$(UV) sync --locked --extra dev

install-all:
	$(UV) sync --locked --all-extras

lock-check:
	$(UV) lock --check

config-check:
	$(PYTHON) scripts/validate_configs.py

test:
	$(PYTHON) -m pytest -m "not network" --cov=alphaforge --cov-branch --cov-report=term-missing --cov-fail-under=78

lint:
	$(PYTHON) -m ruff check alphaforge tests scripts apps benchmarks
	$(PYTHON) -m black --check alphaforge tests scripts apps benchmarks

format:
	$(PYTHON) -m ruff check --fix alphaforge tests scripts apps benchmarks
	$(PYTHON) -m black alphaforge tests scripts apps benchmarks

typecheck:
	$(PYTHON) -m mypy alphaforge tests scripts apps benchmarks

policy:
	$(PYTHON) -m pre_commit run --all-files

check: lock-check config-check policy typecheck test

download-data:
	$(PYTHON) scripts/download_data.py --config configs/data.yaml

build-features:
	$(PYTHON) scripts/build_features.py --config configs/features.yaml --labels-config configs/labels.yaml

# Usage: make label-evidence OUTPUT=/absolute/path/to/new/evidence-directory
label-evidence:
	@test -n "$(OUTPUT)" || (echo "OUTPUT must name a new evidence directory" >&2; exit 2)
	$(PYTHON) scripts/generate_label_evidence.py --output-dir "$(OUTPUT)"

# Usage: make temporal-evidence OUTPUT=/absolute/path/to/new/evidence-directory
temporal-evidence:
	@test -n "$(OUTPUT)" || (echo "OUTPUT must name a new evidence directory" >&2; exit 2)
	$(PYTHON) scripts/generate_temporal_validation_evidence.py --output-dir "$(OUTPUT)"

# Usage: make deep-sequence-evidence BUNDLE=/absolute/bundle OUTPUT=/new/path
deep-sequence-evidence:
	@test -n "$(BUNDLE)" || (echo "BUNDLE must name a verified Signal Foundry bundle" >&2; exit 2)
	@test -n "$(OUTPUT)" || (echo "OUTPUT must name a new evidence directory" >&2; exit 2)
	$(PYTHON) scripts/run_deep_sequence_benchmark.py "$(BUNDLE)" --output "$(OUTPUT)"

# Usage: make time-frequency-evidence OUTPUT=/absolute/path/to/new/evidence-directory
time-frequency-evidence:
	@test -n "$(OUTPUT)" || (echo "OUTPUT must name a new evidence directory" >&2; exit 2)
	$(PYTHON) scripts/run_time_frequency_vision_benchmark.py --output "$(OUTPUT)"

# Usage: make latent-representation-evidence OUTPUT=/absolute/path/to/new/evidence-directory
latent-representation-evidence:
	@test -n "$(OUTPUT)" || (echo "OUTPUT must name a new evidence directory" >&2; exit 2)
	$(PYTHON) scripts/run_latent_representation_benchmark.py --output "$(OUTPUT)"

# Usage: make ensemble-evidence OUTPUT=/absolute/path/to/new/evidence-directory
ensemble-evidence:
	@test -n "$(OUTPUT)" || (echo "OUTPUT must name a new evidence directory" >&2; exit 2)
	$(PYTHON) scripts/run_ensemble_benchmark.py --output "$(OUTPUT)"

# Usage: make decision-policy-evidence OUTPUT=/absolute/path/to/new/evidence-directory
decision-policy-evidence:
	@test -n "$(OUTPUT)" || (echo "OUTPUT must name a new evidence directory" >&2; exit 2)
	$(PYTHON) scripts/run_decision_policy_study.py --output "$(OUTPUT)"

# Usage: make sprint-3-decision-evidence OUTPUT=docs/evidence/signal_foundry_sprint_3/decision
sprint-3-decision-evidence:
	@test -n "$(OUTPUT)" || (echo "OUTPUT must name a new repository-local evidence directory" >&2; exit 2)
	$(PYTHON) scripts/publish_sprint_3_decision.py \
		--config "$(SPRINT_3_DECISION_CONFIG)" \
		--output "$(OUTPUT)"

# Usage: make sprint-5-benchmark OUTPUT=/absolute/path/to/new/raw-evidence.json
sprint-5-benchmark:
	@test -n "$(OUTPUT)" || (echo "OUTPUT must name a new raw benchmark JSON file" >&2; exit 2)
	$(PYTHON) benchmarks/benchmark_distributed_crossover.py --output "$(OUTPUT)"

# Usage: make sprint-5-evidence OUTPUT=/absolute/path/to/new/closeout-directory
sprint-5-evidence:
	@test -n "$(OUTPUT)" || (echo "OUTPUT must name a new evidence directory" >&2; exit 2)
	$(PYTHON) scripts/publish_sprint_5_evidence.py --output "$(OUTPUT)"

# Usage: make mean-variance-evidence OUTPUT=/absolute/path/to/new/evidence-directory
mean-variance-evidence:
	@test -n "$(OUTPUT)" || (echo "OUTPUT must name a new evidence directory" >&2; exit 2)
	$(PYTHON) scripts/publish_mean_variance_study.py \
		--config "$(MEAN_VARIANCE_STUDY_CONFIG)" \
		--output "$(OUTPUT)"

train:
	$(PYTHON) scripts/train_model.py --config configs/models.yaml

evaluate:
	$(PYTHON) scripts/evaluate_model.py --config configs/models.yaml

walk-forward:
	$(PYTHON) scripts/run_walk_forward.py --config configs/models.yaml

backtest:
	$(PYTHON) scripts/run_backtest.py --config configs/backtest.yaml

# Usage: make signal-foundry BUNDLE=/absolute/path/to/<bundle-id>
signal-foundry:
	@test -n "$(BUNDLE)" || (echo "BUNDLE must name a verified Signal Foundry bundle" >&2; exit 2)
	$(PYTHON) scripts/run_signal_foundry_research.py "$(BUNDLE)" --config "$(SIGNAL_FOUNDRY_CONFIG)"

# Usage: make signal-foundry-evidence RUN=/absolute/run BUNDLE=/absolute/bundle OUTPUT=/new/path
signal-foundry-evidence:
	@test -n "$(RUN)" || (echo "RUN must name an immutable governed run" >&2; exit 2)
	@test -n "$(BUNDLE)" || (echo "BUNDLE must name its verified source bundle" >&2; exit 2)
	@test -n "$(OUTPUT)" || (echo "OUTPUT must name a new evidence directory" >&2; exit 2)
	$(PYTHON) scripts/publish_signal_foundry_evidence.py \
		--run-dir "$(RUN)" \
		--bundle-dir "$(BUNDLE)" \
		--config "$(SIGNAL_FOUNDRY_CONFIG)" \
		--output-dir "$(OUTPUT)" \
		$(if $(PROFILE),--performance-profile "$(PROFILE)",)

paper:
	$(PYTHON) scripts/simulate_paper_trading.py --config configs/backtest.yaml

dashboard:
	streamlit run apps/dashboard.py

api:
	uvicorn apps.api:app --reload --port 8000

report:
	$(PYTHON) scripts/generate_report.py

# Full end-to-end pipeline on synthetic data (no network required)
demo:
	$(PYTHON) scripts/run_walk_forward.py --synthetic --fast
	$(PYTHON) scripts/run_backtest.py --latest
	$(PYTHON) scripts/generate_report.py

# --- C++ execution core ---
native:
	$(PYTHON) scripts/build_native.py

bench:
	$(PYTHON) scripts/bench_orderbook.py

bench-native:
	cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build --target bench_orderbook
	./build/bench_orderbook

bench-event:
	$(PYTHON) scripts/bench_event_engine.py

bench-execution-frictions:
	$(PYTHON) benchmarks/benchmark_execution_frictions.py \
		--output runs/execution-frictions-benchmark.json

docker-build:
	docker build -t alphaforge:latest .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
