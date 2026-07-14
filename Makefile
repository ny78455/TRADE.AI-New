.PHONY: help dev test lint format typecheck build clean train backtest

# Default target
help:
	@echo "TRADE.AI - Developer Commands"
	@echo "──────────────────────────────────────────"
	@echo "  make dev          Start FastAPI dev server with hot-reload"
	@echo "  make test         Run all tests with coverage"
	@echo "  make test-unit    Run unit tests only"
	@echo "  make test-perf    Run performance benchmarks"
	@echo "  make lint         Run ruff linter"
	@echo "  make format       Run black formatter"
	@echo "  make typecheck    Run mypy static type checker"
	@echo "  make install      Install all dependencies"
	@echo "  make install-dev  Install dev dependencies"
	@echo "  make train        Train the ML model"
	@echo "  make backtest     Run backtesting engine"
	@echo "  make docker-up    Start full stack with docker-compose"
	@echo "  make docker-down  Stop all docker services"
	@echo "  make clean        Remove generated files and caches"
	@echo "  make pre-commit   Install pre-commit hooks"

# ---- Development ----
dev:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

dev-engine:
	python -m engine.strategy_runner

# ---- Dependencies ----
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,monitoring]"
	playwright install chromium

# ---- Testing ----
test:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-perf:
	pytest tests/performance/ -v --benchmark-json=benchmark_results.json

# ---- Code Quality ----
lint:
	ruff check . --fix

format:
	black .

typecheck:
	mypy config/ core/ engine/ strategies/ indicators/ ml/ data/ api/

check: lint typecheck
	@echo "All checks passed!"

# ---- Pre-commit ----
pre-commit:
	pre-commit install
	pre-commit run --all-files

# ---- ML Pipeline ----
train:
	python scripts/train_model.py

backtest:
	python scripts/run_backtest.py

# ---- Docker ----
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

docker-logs:
	docker-compose logs -f api

# ---- Cleanup ----
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleaned up!"
