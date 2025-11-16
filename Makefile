.PHONY: help install install-dev run example test clean lint format download-data script setup-chromadb compare basic-rag meve-rag simple-compare

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install        Install dependencies"
	@echo "  install-dev    Install with dev dependencies"
	@echo "  run            Run MeVe engine"
	@echo "  example        Run basic example"
	@echo "  compare        Compare Basic RAG vs MeVe RAG with detailed metrics"
	@echo "  simple-compare Simple side-by-side comparison (clean output)"
	@echo "  basic-rag      Run Basic RAG example"
	@echo "  meve-rag       Run MeVe RAG example"
	@echo "  test           Run all tests"
	@echo "  clean          Remove cache files"
	@echo "  lint           Run ruff linter"
	@echo "  format         Format code with black"
	@echo "  script name=<name>  Run script by name (e.g., make script name=download_data)"

install:
	uv sync

install-dev:
	uv sync --extra dev

run:
	uv run python main.py

example:
	uv run python examples/basic_usage.py

compare:
	uv run python examples/compare_rag.py

simple-compare:
	uv run python examples/simple_compare.py

basic-rag:
	uv run python examples/basic_rag.py

meve-rag:
	uv run python examples/meve_rag.py

test:
	uv run pytest __tests__/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache

lint:
	@uv run ruff check . || echo "Run 'make install-dev' to install ruff"

format:
	@uv run black . || echo "Run 'make install-dev' to install black"

# Run any script by name: make script name=<script_name>
script:
	@if [ -z "$(name)" ]; then \
		echo "Error: Please provide script name. Usage: make script name=<script_name>"; \
		echo "Available scripts:"; \
		ls scripts/*.py | xargs -n 1 basename -s .py; \
		exit 1; \
	fi; \
	if [ -f "scripts/$(name).py" ]; then \
		uv run python scripts/$(name).py; \
	else \
		echo "Error: Script 'scripts/$(name).py' not found"; \
		echo "Available scripts:"; \
		ls scripts/*.py | xargs -n 1 basename -s .py; \
		exit 1; \
	fi
