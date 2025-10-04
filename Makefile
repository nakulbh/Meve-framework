.PHONY: help setup install run clean test lint format start

help:
	@echo "MeVe Framework Commands:"
	@echo ""
	@echo "  make setup     - Install uv and dependencies"
	@echo "  make install   - Install dependencies"
	@echo "  make run       - Run MeVe engine"
	@echo "  make clean     - Remove cache files"
	@echo "  make test      - Run tests"
	@echo "  make lint      - Run linting"
	@echo "  make format    - Format code"
	@echo "  make start     - Setup and run"

setup:
	@which uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
	@$(MAKE) install

install:
	uv sync

run:
	uv run meve_engine.py

clean:
	rm -rf __pycache__ services/__pycache__ .pytest_cache .mypy_cache
	rm -f *.pyc */*.pyc chroma-data/*.sqlite3-*

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .

format:
	uv run black .

start: setup run
