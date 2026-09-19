# rag-jev

**Filter and rerank retrieved passages before they reach your answering model.**

```text
your retriever → rag-jev / Jev → your answering model
```

Keep your existing ingestion, vector store and answer model. Inspect each selection,
replay different cutoffs, preserve connected evidence, and measure the tradeoff on your data.
Python, HTTP, TypeScript, LangChain and Dify integrations are included.

**MIT-licensed integration.** Live scoring uses the external TypeSafe Jev API and requires
its key. Query and passage text leave your application during live scoring; arbitrary
metadata and evaluation labels stay local. Optional answer comparison uses your configured
model endpoint. See [data handling](SECURITY.md).

## Try it without an API key

```sh
python -m pip install rag-jev
rag-jev serve --replay-only
```

Open **http://127.0.0.1:8000**. Explore a recorded improvement and a recorded regression,
inspect dropped passages, and change selection settings without scoring again. Recorded
answers are clearly labeled. Live inference is disabled in replay mode, even with keys
in the environment. The separate hand-authored fixtures are labeled as fixtures.

Python 3.11+ is required. For source development, clone this repository and run
`uv sync --locked` using [uv](https://docs.astral.sh/uv/). Node 20+ is required for the
TypeScript client. Release wheels and npm tarballs are also available from
[GitHub Releases](https://github.com/EmreKaplaner/rag-jev/releases).

## Use live scoring

Get a key from [TypeSafe](https://console.typesafe.ai/) and export it as
`TYPESAFE_API_KEY`. From a source checkout, you can instead copy `.env.example`
to `.env` and set the key there. Restart without `--replay-only`. An answer-model
key is optional.

```sh
rag-jev serve
# From the source checkout:
rag-jev select examples/request.json
```

For your application:

```python
from rag_jev import ContextSelector, Document, Jev

async with Jev() as provider:
    selector = ContextSelector(provider)
    result = await selector.select(
        query=question,
        documents=[Document(id=c.id, text=c.text) for c in retrieved_chunks],
        scoring_strategy="contextual",
        mode="filter_and_rerank",
        min_relevance=0.20,  # Illustrative; calibrate on your data.
        max_context_tokens=3000,  # Passage-text estimate; leave room for prompts/output.
        shadow=True,  # Observe proposed selection while returning original context.
    )
```

[Real RAG integration guide](docs/REAL_RAG.md) · [Calibrate on your data](docs/CALIBRATION.md) ·
[Independent human review](docs/HUMAN_REVIEW.md) · [Contribute](CONTRIBUTING.md)

## What the measurements show

| Evaluation | Observed result | Scope |
| --- | --- | --- |
| SciFact retrieval | NDCG@10: **75.13 Jev / 72.11 Ettin / 66.47 BM25** | All 300 test queries; same BM25 top20 candidates |
| HotpotQA answers | F1 **76.70 → 77.20**; estimated API cost **−19.7%** | 200 isolated questions; three answers per arm/question |
| MuSiQue answers | F1 **70.31 → 72.80**; estimated API cost **−22.8%** | 200 isolated questions; three answers per arm/question |

Both answer-F1 difference intervals include zero. The answer-accuracy superiority gate
was not met. Costs use uncached normalization and include Jev; Ettin CPU costs are separate.
These are scoped results, not universal improvements. Human adjudication is pending.

[Controlled study and uncertainty](benchmarks/research/RESULTS.md) ·
[SciFact results](benchmarks/research/RETRIEVAL_RESULTS.md) ·
[Methods and reproduction](benchmarks/research/README.md) ·
[Robustness diagnostics](benchmarks/robustness/README.md) ·
[Demo script](docs/RESEARCH_DEMO.md)

The historical research bundle preserves original code and raw records. Its fingerprints
should be checked inside that bundle; the current product has evolved since the experiment.

## Python

Install with `pip install rag-jev`. The distribution is
[`rag-jev` on PyPI](https://pypi.org/project/rag-jev/), and the import is `rag_jev`.
For source development, install this checkout with `pip install .` or use `uv sync`.

```python
import asyncio
from rag_jev import ContextSelector, Document, Jev

async def main():
    async with Jev() as provider:  # reads TYPESAFE_API_KEY from the environment
        selector = ContextSelector(provider, timeout_ms=5000, max_concurrency=8)
        result = await selector.select(
            query="What is the refund window?",
            documents=[
                Document(id="shipping", text="Orders ship within 5 business days."),
                Document(id="refund", text="Request a refund within 30 days.",
                         metadata={"source": "refunds.md", "page": 2}),
            ],
            min_relevance=0.20,
            shadow=True,
        )
        # Feed result.documents to your existing generator.
        # In shadow mode these are the originals; selected_ids shows what would change.
        print(result.model_dump_json(indent=2))

asyncio.run(main())
```

The library does not implicitly read `.env`; export the key or call
`dotenv.load_dotenv()` yourself. See [`examples/python_rag.py`](examples/python_rag.py).
Reuse one provider and selector per application event loop, then close the provider
at shutdown. The concurrency bound is shared across all requests using that selector.

Any retriever can supply `{id, text, metadata}` objects. For frameworks, map their
documents into this shape and use returned IDs to recover the original objects.
Include relevant titles/headings in `text`: metadata is preserved but **not sent
to Jev**. The query, passage text, and explicit relevance guidance are sent to TypeSafe.

## TypeScript

Install the published client with `npm install rag-jev-client`.
For development from this checkout:

```sh
npm --prefix clients/typescript ci
npm --prefix clients/typescript run build
# With the service already running:
node examples/typescript-rag.mjs
```

The [client README](clients/typescript/README.md) covers local packaging and usage
in another project. The client uses generated OpenAPI types, validates responses,
supports caller cancellation, and has no runtime dependencies. Use it server-side.

## Selection contract

`POST /v1/select` accepts the same options as `selector.select(...)`:

| Argument | Behavior |
| --- | --- |
| `query` | Required nonblank text, up to 8,000 characters |
| `documents` | Up to 256 documents with unique nonblank string IDs |
| `mode` | `filter` (default), `rerank`, or `filter_and_rerank` |
| `min_relevance` | Required for filtering, in `[0,1]`; rejected in rerank-only mode |
| `top_n` | Optional positive target after thresholding; pins and whole groups may exceed it |
| `max_context_tokens` | Optional passage-text token budget; preserves pins and whole groups, reports pinned overruns |
| `scoring_strategy` | `independent` (default) or `contextual` shared-candidate scoring |
| `retrieval_query` | Optional standalone query for conversational retrieval |
| `shadow` | Default `false`; when true, return unchanged context and proposed decisions |
| `relevance_guidance` | Optional domain-specific criteria, up to 2,000 characters |

Filtering retains scores **greater than or equal to** the threshold. Reranking
sorts descending; ties preserve input order. Reranking uses probability of usefulness,
not a graded relevance rubric. Empty inputs make no provider calls. If every chunk
fails the threshold, a successful request returns an empty selection.

Independent scoring uses one Jev request per passage with a yes/no (`Noul`) question.
Contextual scoring judges the candidate passages together and reports shared usage. The rubric includes
partial evidence, useful context, and facts correcting a false premise. It does not
require one passage to independently answer the whole query. A Noul has no separate
confidence field; uncertainty is reflected in its probability.

The response includes:

- `documents`: the actual context to send downstream.
- `selected_ids`: the proposed ordered selection, or `null` when selection was bypassed.
- `decisions`: one row per original chunk, in original order, with probability,
  resolved model, observed tokens, selected/returned flags, and a policy reason.
- `status`: `applied`, `shadow`, or `bypassed`.
- `error_code`: a sanitized failure code on bypass.
- `models`, `prompt_version`, and `policy_version`: reproducibility information.
- `usage`, `elapsed_ms`, and original/selected/returned character counts.

Reasons are `retained`, `below_threshold`, `beyond_top_n`, `beyond_token_budget`, `pinned`, `group_retained`,
and `bypassed`. They are
policy explanations, not model-generated rationales. Shadow decisions distinguish
what **would be selected** from what **was actually returned**.

## Failures and limits

The total deadline includes queueing, network calls, and retries. The SDK may retry
once within that budget. On an upstream error, timeout, or invalid response, the
default library/service policy returns the **entire original context**, even when
`top_n` was requested. `status="bypassed"` makes that explicit. Partial rankings
are never applied. Outstanding tasks are canceled and awaited.

Use `ContextSelector(..., on_error="raise")` to raise `ProviderError` instead.
The service's equivalent is `RAG_JEV_ON_ERROR=raise`; failures become HTTP 502
(provider) or 504 (deadline). The CLI always raises on upstream failure and exits
nonzero, so evaluations cannot accidentally count passthrough as successful scoring.
Caller cancellation propagates rather than becoming passthrough.

On bypass, observed token usage can undercount provider billing: canceled or failed
calls may already have consumed tokens. `usage.complete=false` records this.
Core selection reports character counts. The workbench adds labeled token estimates and,
after optional generation, actual provider prompt/output usage plus configured-rate cost estimates.

Input limits: 60,000 characters per passage, 1,000,000 passage characters in total,
and 2 MB per HTTP body. These are local bounds, not promises about the provider's
token budget. Oversized or malformed input is rejected, never silently truncated.
Provider context-limit errors use normal failure handling.

`TYPESAFE_API_KEY` is required to start the live service. There is no silent demo
mode. `/healthz` checks service liveness, not provider availability.

| Environment variable | Default |
| --- | --- |
| `TYPESAFE_API_KEY` | Required for real calls |
| `RAG_JEV_MODEL` | `jev-latest`; pin a returned model version for repeatable evaluation |
| `RAG_JEV_TIMEOUT_MS` | `5000` |
| `RAG_JEV_MAX_CONCURRENCY` | `8` per selector / server process |
| `RAG_JEV_ON_ERROR` | `passthrough` for the service |
| `RAG_JEV_API_TOKEN` | Optional on localhost; required by CLI for external binds |

The service bearer token is separate from the provider key. When configured, send
`Authorization: Bearer <service-token>` to `/v1/select`. Keep the service behind
TLS/authenticated infrastructure if exposed beyond localhost. This MVP is a library
and single-service deployment, not a multitenant hosted SaaS. API access logs are
disabled by the CLI and input/response bodies are not logged by application code.

## Evaluate before enabling

Evaluation cases contain `id`, `query`, `documents`, `relevant_ids`, and an optional
`reference_answer`. Use representative positive, ambiguous, contradictory, and
unanswerable queries. Labels never enter the Jev request.

Offline demonstration:

```sh
uv run rag-jev eval examples/eval-cases.json \
  --replay examples/fixture-traces.json --min-relevance 0.20
```

Live scoring, then offline threshold adjustment:

```sh
mkdir -p artifacts
uv run rag-jev eval examples/eval-cases.json \
  --min-relevance 0.20 --save-traces artifacts/live-traces.json \
  --output artifacts/live-report.json
uv run rag-jev eval examples/eval-cases.json \
  --replay artifacts/live-traces.json --min-relevance 0.35 \
  --output artifacts/threshold-035.json
```

The report compares baseline retrieval, filter, rerank, and filter-and-rerank.
Jev scoring happens once per passage; policy comparisons reuse those judgments.
Saved traces include an input fingerprint and prompt version, and mismatches fail.
Traces contain judgments and input hashes, not passage text.

Metrics include precision, evidence recall within retrieved candidates, binary nDCG,
relevant chunks dropped, total loss of useful evidence, empty-result correctness,
context characters, actual scoring token usage, and observed scoring p95 latency.
Summary values are query-level means. Recall/nDCG exclude no-relevant-document
queries; empty-result correctness measures those separately. Precision is zero for
empty outputs. A shared `top_n` cap applies to every policy, including baseline;
nDCG uses that fixed cutoff (or candidate count), padding shorter outputs with zeros.

**These metrics do not establish answer correctness.** The report explicitly says
`answer_quality="not_evaluated"` unless you pass an application callback to the Python
`evaluate()` function:

```python
from rag_jev.evaluation import evaluate

async def generate_and_grade(case, documents):
    answer = await your_generator(case.query, documents)
    return await your_grader(case.query, answer, case.reference_answer)  # bool

report = await evaluate(cases, traces, min_relevance=0.20,
                        answer_evaluator=generate_and_grade)
```

The callback runs once per case per policy. Its inference cost/latency is not part
of the scoring metrics. Evaluate it separately with your existing generation stack.
Filtering cannot recover missing evidence, guarantee correct judgments, or act as
a prompt-injection security boundary.

## Development and verification

```sh
make setup
make check
uv build
```

`make check` runs lint/format checks, strict Python type checking, TypeScript build
and client tests, then Python tests including real local socket connections:

```text
TypeScript client → HTTP service → selector → official TypeSafe SDK → test HTTP endpoint
```

The controlled endpoint exercises provider responses, failures, and deadlines.
It does not verify live Jev quality. Real calls are explicitly opt-in:

```sh
RAG_JEV_LIVE=1 uv run pytest tests/test_live.py -q
```

Generate updated HTTP/client contracts with `make schema` after changing models.
`uv.lock` and the TypeScript `package-lock.json` are checked in. Build distributions
with `uv build`; build and `npm pack` the TypeScript client for another application.

The core is in [`src/rag_jev`](src/rag_jev), examples in [`examples`](examples), and
contract/failure/integration tests in [`tests`](tests). No account system, database,
semantic cache, or scorer-provider switching layer is required for this MVP. The workbench
is bundled with the service. Native LangChain and Dify integrations are documented in
[the real-RAG guide](docs/REAL_RAG.md).

Official references: [TypeSafe SDK](https://docs.typesafe.ai/sdk/python),
[Noul](https://docs.typesafe.ai/primitives/noul),
[RAG passage filtering](https://docs.typesafe.ai/cookbooks/classifying_rag_passages).
