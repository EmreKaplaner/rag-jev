.PHONY: setup setup-integrations check check-all demo serve schema

setup:
	uv sync --locked
	npm --prefix clients/typescript ci
	uv run playwright install chromium

setup-integrations:
	uv sync --locked --project integrations/dify --python 3.12

check-all: setup-integrations check

check:
	uv run ruff check src tests examples scripts integrations/dify benchmarks/public benchmarks/research benchmarks/robustness benchmarks/ecosystem benchmarks/fusion
	uv run ruff format --check src tests examples scripts integrations/dify benchmarks/public benchmarks/research benchmarks/robustness benchmarks/ecosystem benchmarks/fusion
	uv run mypy src/rag_jev
	npm --prefix clients/typescript test
	uv run --group benchmark pytest -q

schema:
	uv run python scripts/export_schema.py
	npm --prefix clients/typescript run generate

demo:
	uv run rag-jev demo

serve:
	uv run rag-jev serve
