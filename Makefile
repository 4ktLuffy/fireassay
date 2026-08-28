.PHONY: install test lint typecheck demo clean

DB := /tmp/fireassay-demo.db

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy src

demo:
	rm -f $(DB) $(DB)-wal $(DB)-shm
	python -m fireassay.cli init --db $(DB)
	python -m fireassay.cli questions import tests/fixtures/questions.jsonl --db $(DB)
	python -m fireassay.cli suite freeze --name demo --version 1.0.0 --db $(DB)
	python -m fireassay.cli run --matrix configs/matrix.example.yaml --corpus tests/fixtures/corpus.jsonl --db $(DB)
	python -m fireassay.cli compare --suite demo@1.0.0 --db $(DB)

clean:
	rm -f $(DB) $(DB)-wal $(DB)-shm
	find . -type d -name __pycache__ -exec rm -rf {} +
