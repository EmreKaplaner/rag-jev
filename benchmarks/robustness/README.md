# Robustness and concurrency diagnostics

18 synthetic, hand-labeled cases across six categories. Both independent and contextual
Jev scoring were tested at client concurrency 1, 4 and 8 (108 calls total).
No answer generation was evaluated. These cases are diagnostic, not representative
benchmarks or proof of prompt-injection resistance.

| Strategy | Concurrency | p50 ms | p95 ms | Bypassed |
| --- | --- | --- | --- | --- |
| independent | 1 | 421 | 1192 | 0 |
| independent | 4 | 340 | 828 | 0 |
| independent | 8 | 667 | 847 | 0 |
| contextual | 1 | 331 | 518 | 0 |
| contextual | 4 | 379 | 1047 | 0 |
| contextual | 8 | 340 | 838 | 0 |

All labeled relevant passages were retained. All missing-evidence cases returned no
selection. Exact per-case selections, including embedded-instruction cases, are in
[results.json](results.json). Long passages contain approximately 250 repeated distractor
sentences and a relevant fact at the head or tail. Duplicates are preserved; this is not
a deduplication feature. Conversational cases use an explicitly supplied standalone query.

Per-call selection duration includes the selector semaphore queue; client queue+stage includes waiting for the workload concurrency slot. Cold first call included. p95 is nearest rank on 18 cases per condition; not a production SLA or hardware-neutral comparison.

Known scoring usage cost estimate: $0.00608996. No incomplete usage records.

## Reproduction

The original source snapshot and raw runs are in the release robustness archive. Extract
it into a new directory. Run `uv sync --locked` and then
`uv run python -m benchmarks.robustness.run report` to recompute without paid calls.
For fresh inference use the archived `run` stage; existing matching records are reused.
Start a new study directory for new attempts, retaining original failures. The workload
was frozen before live calls and was not used to tune the selector.
