PYTHON ?= python3.12
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: setup lint format test check clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -e ".[dev]"

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

test:
	$(BIN)/pytest -q

check: lint test

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache
