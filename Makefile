# =============================================================================
# Makefile for Ontology Reasoning System
# =============================================================================

.PHONY: help install dev test lint format typecheck clean docker-build docker-up docker-down docker-logs

# Default target
help:
	@echo "Ontology Reasoning System - Available Commands"
	@echo "==============================================="
	@echo ""
	@echo "Development:"
	@echo "  make install      Install production dependencies"
	@echo "  make dev          Install development dependencies"
	@echo "  make run          Run API server locally"
	@echo "  make run-dev      Run API server in development mode"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run all tests"
	@echo "  make test-unit    Run unit tests only"
	@echo "  make test-int     Run integration tests only"
	@echo "  make test-cov     Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         Run linter (ruff)"
	@echo "  make format       Format code (ruff)"
	@echo "  make typecheck    Run type checker (mypy)"
	@echo "  make check        Run all checks (lint + typecheck)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build Build Docker images"
	@echo "  make docker-up    Start all services"
	@echo "  make docker-down  Stop all services"
	@echo "  make docker-logs  View service logs"
	@echo "  make docker-clean Remove containers and volumes"
	@echo ""
	@echo "Database:"
	@echo "  make db-setup     Initialize Neo4j schema"
	@echo "  make db-reset     Reset Neo4j database"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        Clean build artifacts"
	@echo "  make clean-all    Clean everything including caches"

# =============================================================================
# Development
# =============================================================================

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

run-dev:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-int:
	pytest tests/integration/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -x --tb=short

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

format:
	ruff format src/ tests/

typecheck:
	mypy src/

check: lint typecheck
	@echo "All checks passed!"

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-up-logs:
	docker-compose up

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-logs-api:
	docker-compose logs -f api

docker-logs-neo4j:
	docker-compose logs -f neo4j

docker-clean:
	docker-compose down -v --rmi local

docker-restart:
	docker-compose restart

docker-shell-api:
	docker-compose exec api /bin/bash

docker-shell-neo4j:
	docker-compose exec neo4j /bin/bash

# =============================================================================
# Database
# =============================================================================

db-setup:
	python -c "import asyncio; from src.graph.neo4j_client import get_ontology_client; c = get_ontology_client(); asyncio.run(c.connect()); asyncio.run(c.setup_schema())"

db-reset:
	@echo "WARNING: This will delete all data!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	docker-compose exec neo4j cypher-shell -u neo4j -p password123 "MATCH (n) DETACH DELETE n"

db-stats:
	python -c "import asyncio; from src.graph.neo4j_client import get_ontology_client; c = get_ontology_client(); asyncio.run(c.connect()); print(asyncio.run(c.get_stats()))"

# =============================================================================
# Desktop App
# =============================================================================

desktop-install:
	cd desktop && npm install

desktop-dev:
	cd desktop && npm run dev

desktop-build:
	cd desktop && npm run build

desktop-package:
	cd desktop && npm run package

# =============================================================================
# Utilities
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	rm -rf node_modules/
	rm -rf desktop/node_modules/
	rm -rf desktop/dist/
	rm -rf desktop/release/

# Health check
health:
	curl -s http://localhost:8000/api/health | python -m json.tool

# Version info
version:
	@python -c "from src.config.settings import get_settings; s = get_settings(); print(f'{s.app_name} v{s.app_version}')"
