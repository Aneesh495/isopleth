PYTHON ?= $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PYTEST ?= $(shell if [ -f .venv/bin/pytest ]; then echo .venv/bin/pytest; else echo pytest; fi)

.PHONY: all bootstrap doctor build test data train evaluate ablate stress bench docs demo acceptance verify release numerics test-conservation test-gradients cross-grid inverse clean

all: test

bootstrap:
	uv venv --python 3.12 .venv || python3.12 -m venv .venv
	uv pip install --python $(PYTHON) -r requirements-dev.txt || $(PYTHON) -m pip install -r requirements-dev.txt
	uv pip install --python $(PYTHON) -e . || $(PYTHON) -m pip install -e .

doctor:
	$(PYTHON) -m isopleth.cli doctor

build:
	uv pip install --python $(PYTHON) -e . || $(PYTHON) -m pip install -e .

test:
	$(PYTEST) tests/

numerics:
	$(PYTEST) tests/test_burgers.py tests/test_shallow_water.py tests/test_mms.py tests/test_transforms.py tests/test_sources.py tests/test_limiters.py

test-conservation:
	$(PYTEST) tests/test_conservation.py

test-gradients:
	$(PYTEST) tests/test_gradients.py

cross-grid:
	$(PYTHON) -m isopleth.cli cross-grid --help

inverse:
	$(PYTHON) -m isopleth.cli inverse --help

data:
	$(PYTHON) -m isopleth.cli data generate --family burgers --count 100
	$(PYTHON) -m isopleth.cli data generate --family shallow_water --count 100

train:
	$(PYTHON) -m isopleth.cli train --config configs/train_burgers_flux.json

evaluate:
	$(PYTHON) -m isopleth.cli evaluate --run-dir runs/latest

ablate:
	$(PYTHON) -m isopleth.cli ablate --config configs/ablation_matrix.json

stress:
	$(PYTEST) tests/test_stress.py

bench:
	$(PYTHON) -m isopleth.cli bench

docs:
	$(PYTHON) scripts/render_diagrams.py
	$(PYTHON) scripts/check_docs.py

demo:
	$(PYTHON) -m isopleth.viewer.app --port 8501

acceptance:
	$(PYTHON) scripts/run_acceptance.py $(if $(RUN_ID),--run-id $(RUN_ID),)

verify:
	$(PYTHON) scripts/verify_evidence.py $(if $(EVIDENCE),--evidence $(EVIDENCE),)

release:
	$(PYTHON) scripts/package_release.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
