# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0

SHELL := bash
.SHELLFLAGS := -euo pipefail -c
.DEFAULT_GOAL := help

# .env is sourced by the shell, not parsed by make, so quoted values work. Only targets that
# touch real storage load it; unit tests never do.
LOAD_ENV := set -a; [ ! -f .env ] || . ./.env; set +a;
export

COMPOSE ?= docker compose -f deploy/docker-compose.yml
UV_RUN  := uv run --locked

# NeMo Curator pin. The tutorial CLI is not shipped in the wheel, so fetch it from the
# matching tag and verify its checksum.
CURATOR_VERSION    := 1.3.0
CURATOR_COMMIT     := 6b956ce8965820de1b638fedf6de0cbcf0cc46ba
CURATOR_SCRIPT_URL := https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/$(CURATOR_COMMIT)/tutorials/video/getting-started/video_split_clip_example.py
CURATOR_SCRIPT_SHA := e6fec820b74911cd3b9a0eadc88be1c01b47dc06b49e90683a88fd584570307d
CURATOR_SCRIPT     := .cache/curator/video_split_clip_example.py

.PHONY: help setup setup-curator up attach down seed run bench verify test test-integration check lint fmt

help: ## List targets
	@awk 'BEGIN { FS = ":.*## " } /^[a-zA-Z_-]+:.*## / { printf "  %-17s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: $(CURATOR_SCRIPT) ## Dev env (any OS): package, linters, pinned Curator CLI script
	echo "$(CURATOR_SCRIPT_SHA)  $(CURATOR_SCRIPT)" | shasum -a 256 -c -
	uv sync --locked

$(CURATOR_SCRIPT):
	@mkdir -p $(@D)
	curl -fsSL -o $@.tmp $(CURATOR_SCRIPT_URL)
	echo "$(CURATOR_SCRIPT_SHA)  $@.tmp" | shasum -a 256 -c - && mv $@.tmp $@

setup-curator: setup ## GPU host (Linux x86_64, CUDA 12): NeMo Curator in .venv-curator
	@test "$$(uname -s)-$$(uname -m)" = Linux-x86_64 || { echo "Curator requires Linux x86_64 + CUDA 12" >&2; exit 1; }
	UV_PROJECT_ENVIRONMENT=$(CURDIR)/.venv-curator uv sync --locked --project deploy/curator --python 3.12

up: ## Start AIS on 127.0.0.1 and attach s3://$B2_BUCKET
	$(LOAD_ENV) $(COMPOSE) up -d --wait
	@$(MAKE) --no-print-directory attach

attach: ## Attach s3://$B2_BUCKET to the AIS cluster at $AIS_ENDPOINT (local or existing)
	$(LOAD_ENV) ./scripts/attach_b2_bucket.sh

down: ## Stop AIS (cache volume is kept)
	$(LOAD_ENV) $(COMPOSE) down

seed: ## Copy the pinned video subset from Hugging Face into B2
	$(LOAD_ENV) $(UV_RUN) python scripts/seed_b2.py

run: ## Prefetch via AIS, run NeMo Curator, publish results to B2 (ARGS="--dry-run" etc.)
	$(LOAD_ENV) $(UV_RUN) python scripts/run_pipeline.py $(ARGS)

bench: ## Read benchmark: B2 direct vs AIS cold vs AIS warm (ARGS="--prefix raw/ --limit 50")
	$(LOAD_ENV) $(UV_RUN) python benchmarks/read_throughput.py $(ARGS)

verify: ## Verify a published dataset or output (PREFIX=curated/<run>, ARGS="--direct")
	@test -n "$(PREFIX)" || { echo "set PREFIX, e.g. make verify PREFIX=curated/<run-id>" >&2; exit 2; }
	$(LOAD_ENV) $(UV_RUN) python scripts/verify_artifact.py --prefix "$(PREFIX)" $(ARGS)

test: ## Unit tests (offline, no GPU)
	$(UV_RUN) pytest

# Real B2 from .env (opt in with B2_IT=1). A fresh AIS cluster runs under a separate compose
# project, ports, and cache volume. Host cache overrides are ignored; `down -v` drops the
# test cache. B2 is untouched except for test objects under it/ (see tests/integration/test_ais_b2.py).
IT_COMPOSE := $(COMPOSE) -p aistore-b2-curation-it
test-integration: ## Real-B2 integration tests on a disposable AIS cluster (ports 51280/51281)
	@$(LOAD_ENV) \
	[ "$${B2_IT:-}" = 1 ] || { echo "set B2_IT=1 in .env to run against real B2 (see README)" >&2; exit 2; }; \
	[ -n "$${B2_KEY_ID:-}" ] && [ -n "$${B2_APP_KEY:-}" ] || { echo "B2_KEY_ID and B2_APP_KEY are required in .env" >&2; exit 2; }; \
	export AIS_ENDPOINT=http://127.0.0.1:51280 AIS_PORT=51280 AIS_TARGET_PORT=51281; \
	export AIS_CACHE_DIR=target_disk0; \
	trap '$(IT_COMPOSE) down -v >/dev/null 2>&1' EXIT; \
	$(IT_COMPOSE) up -d --wait; \
	$(UV_RUN) pytest -m integration -o addopts="" $(ARGS)

# The pinned Curator script backs an upstream contract test in tests/test_run_pipeline.py.
check: $(CURATOR_SCRIPT) lint test ## Lint + unit tests (what CI runs)

lint: ## Ruff lint and format check
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .

fmt: ## Apply ruff fixes, then format
	$(UV_RUN) ruff check --fix .
	$(UV_RUN) ruff format .
