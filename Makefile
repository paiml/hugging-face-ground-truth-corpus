.PHONY: setup lint format test test-fast test-unit test-doctest coverage coverage-check comply security check build clean

# Setup
setup:
	uv sync --all-extras
	uv run pre-commit install

# Linting
lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

# Testing
test:
	uv run pytest

test-fast:
	uv run pytest -x -q --no-cov tests/unit/

test-unit:
	uv run pytest tests/unit/ -v

test-doctest:
	uv run pytest --doctest-modules src/

# Coverage
coverage:
	uv run pytest --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

coverage-check:
	uv run pytest --cov-fail-under=95

# Security
security:
	uv run bandit -r src/ -ll

# Full quality check (Jidoka gates)
check: lint coverage-check security
	@echo "All quality gates passed!"

# Build
build:
	uv build

# Clean
clean:
	rm -rf dist/ build/ *.egg-info/ .coverage htmlcov/ .pytest_cache/ .ruff_cache/ __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
