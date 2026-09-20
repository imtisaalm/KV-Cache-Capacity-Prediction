.PHONY: test lint typecheck check

test:
	pytest -q

lint:
	ruff check src tests

typecheck:
	mypy src

check: lint typecheck test
