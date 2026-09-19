# Ecosystem pilot

This is a separate, exploratory campaign with a **$25 API ceiling**. The previous
research-v3 and retrieval-v1 experiments remain unchanged. Read the committed
[protocol](protocol.json) before interpreting any results.

## What is being tested

| Component | Pilot scope |
|---|---|
| BEIR | Complete FiQA (57,638), NFCorpus (3,633), SciFact (5,183) corpora; 100 official test questions and 16 development questions each |
| First-stage retrieval | Custom audited BM25 and pinned E5-small-v2, exact normalized dense search across each complete corpus |
| Selection | Original top10; BGE-v2-m3 top10; Jev rerank top10; Jev threshold .20 then top10; BGE top10 then fresh Jev filtering at .20 |
| FlashRAG | Actual upstream `SequentialPipeline`, CRAG Task1 official supplied pages, 40 public-test plus 16 validation questions, five answer arms |
| BERGEN | Actual upstream `Rerank.eval`, replaying paid contextual Jev scores and checking every ranking; **not the full BERGEN RAG benchmark** |
| RAGChecker | Native evaluator on eight deterministically sampled CRAG test questions, all five arms; same-model judge diagnostics |
| CRAG semantic diagnosis | Pinned upstream grading prompts, parser and 75-token trim; corrected loop and Luna judge, explicitly not official leaderboard scores |
| Human review | Blinded CRAG review packet; no human results claimed before independent review |

CRAG pages are cleaned and completely chunked into 220-word windows with stride
180, then BM25 retrieves 20 chunks. No gold evidence is inserted. This is CRAG's
supplied-search-page setting, not a fresh web search or Wikipedia corpus benchmark.
Generation uses the configured GPT-5.6 Luna endpoint, low reasoning, 2,000 output
tokens, one independent answer per arm. Filtering can return no context.

The .20 policy is fixed, not selected on these test outcomes. Development and test
IDs are sampled separately by SHA-256. SciFact's test set was evaluated earlier:
its results are replication, not an untouched holdout. All models may have training
overlap with public benchmarks. E5 truncates at 512 tokens and BGE at 1,024; report
the affected counts. BGE uses float16 on MPS, recorded in a pre-paid amendment
after dummy-input hardware timing, before any benchmark outcomes were inspected.

Retrieval uses graded `pytrec_eval` nDCG@10, Recall@10, MRR@10 and candidate recall@20.
Confidence intervals are descriptive paired bootstrap intervals; BEIR queries
sharing relevant documents are clustered. CRAG EM/F1 are short-answer diagnostics,
**not the official CRAG semantic-judge score**. RAGChecker uses the generator model
as judge, so it is not independent validation. No pilot SOTA or universal accuracy
claim follows from these measurements.

## Reproduce

Run from the repository root. The recorded hardware is macOS/Apple MPS; changing
device, model, sample, or policy creates a new experiment, not a byte-identical replay.
Environment manifests are pinned separately because RAGChecker requires an older
Accelerate version. `--no-deps` below installs the exact recorded environment;
optional upstream features outside this pilot are not installed.

```bash
uv venv artifacts/ecosystem-v1/venv --python 3.12
uv pip install --python artifacts/ecosystem-v1/venv/bin/python --no-deps \
  -r benchmarks/ecosystem/requirements.lock.txt
uv venv artifacts/ecosystem-v1/judge-venv --python 3.12
uv pip install --python artifacts/ecosystem-v1/judge-venv/bin/python --no-deps \
  -r benchmarks/ecosystem/judge-requirements.lock.txt

# Downloads public sources and pins upstream checkouts; no paid API calls.
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.bootstrap
artifacts/ecosystem-v1/venv/bin/python -m pytest -q tests/test_ecosystem.py

# Run each pair: fiqa/nfcorpus/scifact with bm25 and dense; crag with bm25.
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.retrieve fiqa bm25
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.retrieve fiqa dense
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.retrieve nfcorpus bm25
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.retrieve nfcorpus dense
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.retrieve scifact bm25
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.retrieve scifact dense
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.retrieve crag bm25
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.score bge

# Paid stages: .env provides the existing TypeSafe and generation credentials.
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.score jev --limit 1
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.score jev
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.answer --limit 1
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.answer
artifacts/ecosystem-v1/judge-venv/bin/python -m benchmarks.ecosystem.judge
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.crag_grade

# Offline verification and reporting.
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.bergen
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.verify
artifacts/ecosystem-v1/venv/bin/python -m benchmarks.ecosystem.report
```

`ledger.json` reserves an upper bound before each paid request under a process
lock. Returned usage replaces that reservation; unknown charges stay reserved.
An interrupted call is never silently retried. Keep the ledger and request
artifacts together. Matching successful artifacts resume without paying again.
The cap covers this campaign's API calls at the frozen public prices; it does not
control unrelated account activity or reconcile vendor invoices. Local retrieval
and reranker computation has **unknown monetary cost**, not zero cost. The report
charges the Jev selection stage to each applicable pipeline arm even though the
experiment reuses its scores across arms.

## Use the adapter in an existing FlashRAG pipeline

```python
from benchmarks.ecosystem.flashrag import JevFlashRAGRetriever
from rag_jev import ContextSelector, Jev

# `pipeline` is an already-configured native FlashRAG SequentialPipeline.
pipeline.retriever = JevFlashRAGRetriever(
    pipeline.retriever,
    ContextSelector(Jev(model="jev-1.13.0"), timeout_ms=30_000),
    scoring_strategy="contextual",
    mode="filter_and_rerank",
    min_relevance=0.20,
    top_n=10,
)
result = pipeline.run(dataset)
```

This adapter lives in the research harness, not the installed wheel API. The
retriever's `contents` field is scored, source records and metadata are preserved,
and duplicate upstream IDs are handled by position. Use `abatch_search` when
already inside an async event loop. Calibrate on your own development data before
production use; .20 is a benchmark policy, not a recommended universal threshold.

## Larger follow-up work

Canonical NQ, TriviaQA and HotpotQA full-Wikipedia runs require the matching corpus
and E5-base index: upstream's corpus ZIP is 5.13 GB and index ZIP is 37.28 GB. A
ZIP64 directory probe confirms the index alone expands to **64.56 GB**, before
adding the extracted corpus. This laptop did not have enough space for installation.
Do not substitute a gold-containing or reduced corpus and call it that benchmark.
Run on a machine sized for the extracted index, repeat generation across seeds,
freeze development choices separately, then use simultaneous intervals for any
superiority claim. Full BERGEN generation replication and independent blinded
human adjudication also remain separate follow-ups.

## Sources and licenses

- [BEIR](https://github.com/beir-cellar/beir), dataset downloads and checksums pinned
  in the protocol; source dataset terms still apply.
- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG), MIT.
- [BERGEN](https://github.com/naver/bergen), CC BY-NC-SA 4.0; kept as an external
  checkout, not vendored into this MIT package.
- [Meta CRAG](https://github.com/facebookresearch/CRAG); source archive is verified
  against the exact Git LFS SHA-256 at the pinned commit.
- [RAGChecker](https://github.com/amazon-science/RAGChecker), Apache-2.0.
- [E5-small-v2](https://huggingface.co/intfloat/e5-small-v2) and
  [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3), exact model
  revisions recorded in the protocol.

Raw downloaded corpora, HTML, model weights and third-party repositories stay in
ignored local artifacts. Reproduction downloads them from their original sources.
