# Contributing

Start with an issue describing the concrete pipeline, desired behavior, and a minimal
example using public or synthetic documents. Do not include keys or private documents.

```sh
uv sync --locked --group benchmark
make setup
make check-all
make schema
uv build
```

Python 3.11 and 3.13 are tested in CI. The Dify SDK check uses an isolated Python 3.12
environment. TypeScript requires Node 20+. Normal tests use controlled providers and
do not require paid keys. Live tests are explicit opt-in.

Keep source IDs, text and metadata intact. Add focused regression tests for changes to
selection, token budgets, failures, replay integrity and adapters. Regenerate OpenAPI and
TypeScript types after changing the wire contract. Document new configuration fields.

Benchmark changes must keep development and evaluation separate, preserve failures, and
report costs including scoring. Never tune against a held-out result and present it as
an independent improvement. Historical study bundles are immutable; use a new study ID.

Small pull requests with a problem statement, behavior change and actual validation are
welcome. Contributions are made under the project's MIT license; third-party content
retains its original terms. Be respectful, focus on evidence, and keep discussions technical.
