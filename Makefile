.PHONY: install test lint typecheck demo mutants clean

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

# harness_mutation_score (M2-SPEC.md §8): mutates fireassay's own source
# (not the system under test) and reports whether our tests catch bugs in
# the harness itself. Slow — run on a schedule, not per-PR (see
# .github/workflows/mutants.yml). Requires `cosmic-ray` (dev extra).
mutants:
	mkdir -p artifacts
	cosmic-ray init cosmic-ray.toml artifacts/harness-mutation.sqlite
	cosmic-ray baseline cosmic-ray.toml
	cosmic-ray exec cosmic-ray.toml artifacts/harness-mutation.sqlite
	cr-report artifacts/harness-mutation.sqlite --show-pending > artifacts/harness-mutation.txt
	cr-report artifacts/harness-mutation.sqlite --json > artifacts/harness-mutation.json

clean:
	rm -f $(DB) $(DB)-wal $(DB)-shm
	find . -type d -name __pycache__ -exec rm -rf {} +
