SHELL := /bin/bash

PYTHON ?= python3
FMLAB_CI_ROOT ?= /tmp/foundation-model-lab-ci
CPU_TORCH_INDEX ?= https://download.pytorch.org/whl/cpu

OFFLINE_ENV = env \
	CUDA_VISIBLE_DEVICES="" \
	FMLAB_DATA_ROOT="$(FMLAB_CI_ROOT)/data" \
	FMLAB_MODEL_ROOT="$(FMLAB_CI_ROOT)/empty-models" \
	HF_DATASETS_OFFLINE=1 \
	HF_HOME="$(FMLAB_CI_ROOT)/cache/huggingface" \
	HF_HUB_DISABLE_TELEMETRY=1 \
	HF_HUB_OFFLINE=1 \
	MPLBACKEND=Agg \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONHASHSEED=0 \
	TOKENIZERS_PARALLELISM=false \
	TORCH_HOME="$(FMLAB_CI_ROOT)/cache/torch" \
	TRANSFORMERS_OFFLINE=1

.PHONY: help bootstrap-cpu metadata lint format-check test evidence toy site-build site-check check ci

help:
	@echo "bootstrap-cpu  Install editable CPU/offline development dependencies"
	@echo "metadata        Validate workflows, issue forms, governance files, and links"
	@echo "lint           Run Ruff lint checks"
	@echo "format-check   Verify Ruff formatting"
	@echo "test           Run the CPU/offline test suite"
	@echo "evidence       Verify every sanitized public-evidence manifest and file hash"
	@echo "toy            Run the bounded download-free toy suite"
	@echo "site-check     Validate the static site/ Pages artifact"
	@echo "site-build     Rebuild social preview and deterministic Pages artifact"
	@echo "check          Run lint, format, evidence, and tests"
	@echo "ci             Run check plus the bounded toy suite"

bootstrap-cpu:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch --index-url "$(CPU_TORCH_INDEX)"
	$(PYTHON) -m pip install -e '.[dev,ml,vlm]'
	$(PYTHON) -m pip check

metadata:
	$(PYTHON) .github/scripts/validate_repository_metadata.py .

lint:
	$(PYTHON) -m ruff check src scripts tests .github/scripts

format-check:
	$(PYTHON) -m ruff format --check src scripts tests .github/scripts

test:
	mkdir -p "$(FMLAB_CI_ROOT)/data" "$(FMLAB_CI_ROOT)/empty-models" \
		"$(FMLAB_CI_ROOT)/cache/huggingface" "$(FMLAB_CI_ROOT)/cache/torch"
	$(OFFLINE_ENV) $(PYTHON) -m pytest -q -p no:cacheprovider

evidence:
	$(PYTHON) .github/scripts/verify_public_evidence.py public-evidence

toy:
	mkdir -p "$(FMLAB_CI_ROOT)/data" "$(FMLAB_CI_ROOT)/empty-models" \
		"$(FMLAB_CI_ROOT)/cache/huggingface" "$(FMLAB_CI_ROOT)/cache/torch"
	$(OFFLINE_ENV) $(PYTHON) scripts/run_toy_suite.py \
		--output-root "$(FMLAB_CI_ROOT)/toy-suite" \
		--device cpu \
		--llm-steps 1 \
		--tiny-steps 2 \
		--vlm-samples 1 \
		--fail-fast

site-build:
	$(PYTHON) scripts/build_social_preview.py
	$(PYTHON) scripts/build_portfolio_site.py

site-check:
	$(PYTHON) .github/scripts/validate_site.py site

check: metadata lint format-check evidence test

ci: check toy
