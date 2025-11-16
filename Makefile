.PHONY: help install install-dev run example test clean lint format download-data

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install        Install dependencies"
	@echo "  install-dev    Install with dev dependencies"
	@echo "  run            Run MeVe engine"
	@echo "  example        Run basic example"
	@echo "  test           Run all tests"
	@echo "  clean          Remove cache files"
	@echo "  lint           Run ruff linter"
	@echo "  format         Format code with black"
	@echo "  download-data  Download HotpotQA dataset"

install:
	uv sync

install-dev:
	uv sync --extra dev

run:
	uv run python main.py

example:
	uv run python examples/basic_usage.py

test:
	uv run pytest __tests__/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache

lint:
	@uv run ruff check . || echo "Run 'make install-dev' to install ruff"

format:
	@uv run black . || echo "Run 'make install-dev' to install black"

download-data:
	uv run python scripts/download_data.py
