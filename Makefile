.PHONY: help setup install run clean test lint format

# Default target
help:
	@echo "MeVe Framework - Available commands:"
	@echo ""
	@echo "  make setup       - Initial project setup (install uv if needed)"
	@echo "  make install     - Install dependencies using uv"
	@echo "  make run         - Run the MeVe engine"
	@echo "  make clean       - Clean up cache and temporary files"
	@echo "  make test        - Run tests (if available)"
	@echo "  make lint        - Run linting checks"
	@echo "  make format      - Format code with black"
	@echo ""

# Setup: Install uv package manager if not present
setup:
	@echo "Setting up MeVe Framework..."
	@which uv > /dev/null || (echo "Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh)
	@echo "Running initial install..."
	@$(MAKE) install
	@echo "Setup complete!"

# Install dependencies
install:
	@echo "Installing dependencies with uv..."
	uv sync
	@echo "Dependencies installed!"

# Run the main MeVe engine
run:
	@echo "Running MeVe Framework..."
	uv run meve_engine.py

# Clean up cache and temporary files
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
	rm -rf services/__pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf *.pyc
	rm -rf */*.pyc
	rm -rf chroma-data/*.sqlite3-*
	@echo "Cleanup complete!"

# Run tests (placeholder for future test suite)
test:
	@echo "Running tests..."
	uv run pytest tests/ -v || echo "No tests found yet"

# Lint code
lint:
	@echo "Linting code..."
	uv run ruff check . || echo "Install ruff for linting: uv add --dev ruff"

# Format code
format:
	@echo "Formatting code..."
	uv run black . || echo "Install black for formatting: uv add --dev black"

# Quick start: setup and run
start: setup run
