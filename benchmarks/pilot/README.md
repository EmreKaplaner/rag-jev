# FastAPI documentation pilot

Ten fixed questions, eight public source files, and twelve actual SQLite FTS5/BM25
retrieval candidates per question. Prepared before receiving any generation results.
The purpose is to check the complete retrieval, selection, generation, and measurement
workflow. This small convenience sample does not establish usefulness across RAG stacks.

- `sources.json`: upstream commit, source URLs, and SHA-256 hashes.
- `corpus/`: original Markdown; MIT license retained in `FASTAPI_LICENSE`.
- `cases.json`: eight answerable and two unanswerable questions, with agent-authored
  expected points. These labels need independent review before reporting correctness.
- `candidates.json`: fixed selection requests from `examples.local_rag.retrieve`.
- `protocol.json`: retrieval, selection and generation settings, input hashes, limitations.

The live run is complete; see [results and limitations](RESULTS.md). The initial OpenAI
request failed with insufficient quota; the replacement key succeeded. To reproduce,
submit each candidate request
to `/v1/runs` as `{"request": <request>}`, then pass the returned `record` and `policy`
to `/v1/compare`. Save each resulting replay, including errors, under `artifacts/pilot/`.
Expected answer points must never be sent to Jev or the answer model.

Use the same Luna settings in both branches: `gpt-5.6-luna`, reasoning effort `low`,
`max_completion_tokens=2000`. Compare all twelve candidates with the fixed Jev filter
cutoff `0.2`. This cutoff is illustrative and uncalibrated; do not tune it after inspecting
these answers and then present those answers as held-out results.

Review all answer pairs for correctness, missing qualifications, abstention, and citation
support. First establish whether retrieval included sufficient evidence, then inspect
whether selection lost any. Source-file references alone are not passage relevance labels.
Measure actual prompt/output usage, total scoring plus generation cost, and stage latency.
Report application no-context fallbacks separately from model-generated answers. Include
failures and regressions in the report. A later representative held-out evaluation is
needed before using benchmark claims in a demo.
