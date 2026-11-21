.PHONY: help install lint format typecheck test cov check demo serve bench bench-quality bench-big docker clean

PYTHON ?= python3

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and development tooling
	$(PYTHON) -m pip install -e ".[service,bench,dev]"

lint:  ## Run the linter
	ruff check .

format:  ## Format the code
	ruff format .

typecheck:  ## Run the type checker
	mypy

test:  ## Run the test suite
	pytest -q

cov:  ## Run tests with a coverage report
	pytest -q --cov=nanosearch --cov-report=term-missing

check: lint typecheck test  ## Everything CI runs

demo:  ## Build the demo index and run a few queries
	$(PYTHON) examples/demo.py

serve:  ## Serve the demo corpus at http://127.0.0.1:8000
	$(PYTHON) -m uvicorn nanosearch.service.app:app --reload

bench:  ## Recall/latency sweep at 100k vectors (about a minute)
	$(PYTHON) benchmarks/bench_sweep.py --n 100000 --dim 128 --nlist 1024 --m 16

bench-quality:  ## Retrieval quality of the text embedder
	$(PYTHON) benchmarks/bench_text_quality.py

bench-big:  ## The million-vector sweep quoted in the README (several minutes)
	$(PYTHON) benchmarks/bench_sweep.py --n 1000000 --dim 128 --nlist 1024 --m 32 \
		--queries 200 --latency-queries 100 --kmeans-iter 15 \
		--nprobe 4 8 16 --factor 16 32 64 --refine-nprobe 8

docker:  ## Build and run the container
	docker build -t nanosearch:local .
	docker run --rm -p 8000:8000 nanosearch:local

clean:  ## Remove build artefacts and caches
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
