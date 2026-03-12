PYTHON ?= python
UVICORN ?= uvicorn
APP_MODULE ?= app.main:app
INDEX_DIR ?= index

.PHONY: install format lint fix test test-ingestion-e2e ingest clean-index rebuild-index run-api check

install:
	$(PYTHON) -m pip install -e .

format:
	$(PYTHON) -m ruff format .

lint:
	$(PYTHON) -m ruff check .

fix:
	$(PYTHON) -m ruff check . --fix

test:
	$(PYTHON) -m pytest tests

test-ingestion-e2e:
	RUN_INGESTION_E2E=1 $(PYTHON) -m pytest tests/test_ingestion.py -v

ingest:
	$(PYTHON) -m ingestion.build_index

clean-index:
	rm -rf $(INDEX_DIR)

rebuild-index: clean-index ingest

run-api:
	$(UVICORN) $(APP_MODULE) --reload --port 8001

check: lint test

