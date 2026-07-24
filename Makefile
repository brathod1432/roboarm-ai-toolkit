.PHONY: help install test lint clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install in dev mode with all extras
	pip install -e ".[all]"

test:  ## Run tests with coverage
	pytest tests/ -v --cov=roboarm --cov-report=term-missing

lint:  ## Run linter
	ruff check src/ tests/
	mypy src/roboarm/

format:  ## Auto-format code
	ruff format src/ tests/

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
