# Try original retrieval, Jev reranking, and fusion

Fusion is available from this source checkout. Install with `uv sync --locked` or
`pip install .`; the previously published 0.2.0 packages do not contain this feature.

Supply the authorized chunks in your retriever's **best-first order**, before its final
top-N cut. A useful starting experiment is 20 candidates and 10 returned chunks.
Contextual scoring accepts up to 32 candidates. Fusion cannot recover missing evidence.

```python
from rag_jev import ContextSelector, Document, Jev

async with Jev() as provider:
    selector = ContextSelector(provider, on_error="passthrough")
    result = await selector.select(
        query=question,
        documents=[Document(id=c.id, text=c.text, metadata=c.metadata)
                   for c in retrieved_chunks],  # Keep the original ranking.
        mode="fusion",
        scoring_strategy="contextual",
        top_n=10,
        shadow=True,
    )
    # Shadow returns the originals. Inspect result.selected_ids before enabling.
    # With shadow=False, send result.documents to your answer model.
```

The HTTP request and generated TypeScript client use those same fields. Remote services
work identically; configure your reachable service URL and bearer token as described in
[deployment](DEPLOYMENT.md). LangChain accepts `Policy(mode="fusion", top_n=10, shadow=True)`.
In Dify choose **Ranking mode → Fusion**, ensure the incoming list is best-first, and
disable its score threshold. Dify receives raw RRF scores; they are not probabilities.

## Exact behavior

For each document, `score = 1/(60 + original_rank) + 1/(60 + jev_rank)`.
Both ranks start at one. Jev scores are sorted descending; equal scores preserve input
order. Equal fused totals preserve input order. Equal weights and k=60 are fixed in this
version. There is no training, query rewriting, extra retrieval, or extra Jev call.

The `min_relevance` argument is rejected in fusion mode. Apply `top_n` and/or a passage
token budget instead. `Decision.relevance` retains the original Jev usefulness estimate;
`original_rank`, `jev_rank`, and `fusion_score` expose the separate ranking signals.

Pins always survive; groups rank by their highest member score and stay in original order
within the group. The top-N target can be exceeded to preserve groups/pins; pinned groups
can exceed the token budget. These overrides mean final group order may differ from
individual score order. No deduplication or metadata rewriting occurs. On provider failure,
the default passthrough returns the complete original list, ignoring selection limits,
and marks `status="bypassed"`. Raise mode is also supported.

## Compare your production retrieval trace

1. Start `uv run rag-jev serve`, open the workbench, paste a real query and its retrieved
   chunks, and optionally label the relevant chunk IDs. Scores alone do not establish quality.
2. Set **Top N target** (for example, 10), choose a scoring strategy, and score once.
3. Click **Compare rankings**. It reuses those judgments for original order, Jev reranking,
   and fusion, all with the same top-N/token limits. No scoring calls are made.
4. With your generation endpoint configured, click **Generate three-way answers**. It uses
   the same question/model/prompt, up to three calls, and reuses an answer when contexts match
   exactly. Inspect correctness and supporting citations; these are not automatically graded.
5. **Export comparison** saves the three selections, diagnostics, and available answers.
   **Export replay** saves the scored source inputs, for further comparisons without Jev calls.

For a downloaded replay:

```sh
uv run rag-jev compare-rankings rag-jev-run.json --top-n 10 --output comparison.json
# Optional paid generation, using your .env endpoint and rate settings:
uv run rag-jev compare-rankings rag-jev-run.json --top-n 10 --generate --output answers.json
```

For an actual retrieval pipeline over your Markdown/text files, with the service running:

```sh
uv run python examples/local_rag.py --documents examples/knowledge \
  --query "How many days do I have to request a refund?" \
  --mode fusion --scoring-strategy contextual --top-k 20 --top-n 3 --shadow
uv run rag-jev compare-rankings artifacts/local-rag-run.json --top-n 3
```

Replace the document directory and `--base-url` with your own corpus and remote service.
SQLite FTS5/BM25 supplies the original ranking; no embedding credentials are needed.

For automated remote evaluation, score once with `POST /v1/runs` and pass its `record` to
`POST /v1/rankings` or `POST /v1/compare-rankings`:

```python
run = client.post("/v1/runs", json={"request": selection_request}).json()
report = client.post("/v1/rankings", json={
    "record": run["record"], "top_n": 10, "max_context_tokens": 3000,
}).json()
```

The three-way comparison intentionally uses input-order top-N as its baseline; a replay's
`baseline_ids` and relevance cutoff are not applied. Use the original two-way comparison
when you need that custom baseline. Pins/groups apply consistently across all three arms.

Set `RAG_JEV_SCORING_INPUT_PER_MILLION`, `RAG_JEV_SCORING_OUTPUT_PER_MILLION`,
`RAG_JEV_GENERATION_INPUT_PER_MILLION`, and `RAG_JEV_GENERATION_OUTPUT_PER_MILLION` to your
provider rates. Unknown rates or usage remain null. Reported deployed-arm API cost includes
Jev and generation; retrieval/local compute and cache discounts are excluded. Recorded
scoring time plus current generation time is not a fresh end-to-end latency measurement.

Evaluate representative queries including missing evidence and conflicting documents.
Use separate development and evaluation sets, a fixed generator, human/reference quality
checks, and paired comparisons. Count bypasses and failures separately. Choose a mode only
after comparing answer quality, coverage, latency, and **total** deployment cost.

## What our replay found

The [600-case replay and chart](../benchmarks/fusion/RESULTS.md) uses the original ecosystem
pilot's human dataset qrels and cached scores. Fusion exceeded original retrieval in all six
settings, but was below Jev-only reranking in five, with the remaining setting nearly tied.
It is optional; defaults remain unchanged. There is no demonstrated general fusion win,
generated-answer improvement, or total-cost saving from that replay.

Reproduce without keys or downloading the corpora:

```sh
uv run --group benchmark python -m benchmarks.fusion.evaluate
```

The independent [catalog fusion study](https://github.com/zhuyansen/jev-search-rerank-eval)
uses BGE-M3 embeddings and a four-level Jev rubric. Our replay uses custom BM25/E5-small
retrieval and contextual Noul judgments, so it tests a different setting. The
[Cohere comparison](https://github.com/anessbelbati/jev-rerank-bench) also illustrates why
aggregation, candidate recall, and scoring formulation must be reported explicitly.

## Live integration evidence

[The saved live smoke](../benchmarks/fusion/live-smoke.json) exercised real SQLite retrieval,
the HTTP service, Jev 1.13.0, and the configured Luna generation endpoint. Seven passages
were retrieved; all three policies selected the same two, so one generated answer was reused.
This verifies connectivity, selection, metadata and citation-ID mapping, not answer-quality
superiority. Recorded new API usage cost was $0.000125926, leaving $24.345274672 of the original
$25 pilot budget. Prices are the frozen pilot rates, not invoice verification.
