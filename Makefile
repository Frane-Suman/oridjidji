# Convenience targets (Linux, CI, Git Bash). On plain Windows use install.ps1 and
# start.ps1, or run the uv commands directly.

.PHONY: sync test lint type fmt check serve

sync:
	uv sync

test:
	uv run pytest -q -m "not slow"

lint:
	uv run ruff check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

type:
	uv run mypy src

check: lint type test

serve:
	uv run oridjidji serve --profile demo --candidate-root ./models --warmup
