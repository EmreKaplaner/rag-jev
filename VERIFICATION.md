# Release verification — September 19, 2026

Latest research checks: **93 Python tests passed, 1 paid opt-in test skipped, 6 TypeScript
tests passed**, plus Ruff, formatting, strict mypy, and package build. See
[the controlled-study results](benchmarks/research/RESULTS.md) and
[the separate corpus-retrieval result](benchmarks/research/RETRIEVAL_RESULTS.md).

## Version 0.2.0 release checks

- MIT license, contribution/security docs and private vulnerability reporting enabled.
- `make check-all`: 93 Python tests passed, one paid opt-in skipped; six TypeScript
  tests passed. Ruff, formatting, strict mypy and Dify SDK checks passed.
- Chromium tests exercise HTTP scoring, answer comparison, replay/import/export,
  regression recording, mobile token-budget selection, and human-review CSV export.
- Wheel installed outside the checkout: replay-only catalog, recordings, review page
  and token-budget model passed. npm tarball installed in a clean project: six client
  tests passed against the installed package.
- Distribution/research archive scan: 40,211 members checked for configured credentials
  and excluded local environment files. License files included in current packages.
- Contextual calibration smoke used recorded real Jev traces; development/held-out
  families were separated, winner frozen before held-out evaluation, reusable config
  defaults to shadow mode. Single-family intervals are suppressed.
- Robustness: 108 live selections over 18 synthetic cases × two strategies × three
  concurrency levels. Zero bypasses/errors/incomplete usage; all labeled relevant
  passages retained and all missing-evidence cases empty. These are diagnostics,
  not representative answer-quality or security guarantees.
- Robustness archive extracted into a clean directory: all frozen code hashes matched;
  freshly installed locked environment reproduced the report byte for byte without
  paid calls. Historical round-3 archive remains immutable.
- Independent human review remains **0/80 answers**, explicitly incomplete. The review
  interface and validation command are ready; automated tests are not human labels.
- Registry publishing remains pending account authentication. Downloadable packages
  are intended for the GitHub release. Remote CI status is recorded below once run.

## Controlled research round 3

- Frozen 96 development questions and 400 new evaluation questions, isolated from all
  640 previously exposed questions by supporting-title / single-hop-component keys.
  No shared evaluation isolation keys. Development: 11 arms × 2 answer samples;
  evaluation: 6 arms × 3 samples. Repetitions are averaged within each question for inference.
- All **9,312 answer records** matched benchmark-author EM/F1 functions; all **9,312
  actual HTTP request captures** matched the frozen prompts, model arguments, question,
  and selected passage text. Responses matched recorded answers and token usage.
  No answer errors, truncations, empty selections, or unknown-usage records.
- All **1,984** applicable Jev policy selections matched the production selector.
  The pinned Ettin baseline used CPU float32 and truncated no passage pairs. Its top-k
  was development-tuned. The optional research profile remains contextual, cutoff .20,
  filter and rerank, without a passage cap; production defaults were not promoted.
- Primary F1 differences versus all context: HotpotQA **+0.50 points**, simultaneous
  97.5% interval **[-2.72, +3.69]**; MuSiQue **+2.50**, **[-1.31, +6.37]**.
  Normalized API cost fell **19.7% / 22.8%**, including Jev. The frozen joint superiority
  gate **did not pass**; a nonsignificant result is not evidence of equivalence.
- Separate full **BEIR/SciFact test split**, 300 queries and 5,183 documents, BM25 top20
  then top10 reranking with no gold insertion: NDCG@10 **66.47 BM25 / 72.11 Ettin /
  75.13 Jev**. All **2,700** query/arm/metric values matched pytrec-eval-terrier 0.5.10
  exactly. Descriptive intervals resample 247 shared-paper query clusters.
- Main experiment estimated actual API spend: **$1.684983**; separate SciFact Jev scoring:
  **$0.131878**. These are returned-usage estimates with cache discounts, not invoices.
  Pipeline comparisons use uncached normalization; local neural compute is accounted
  separately, not assigned zero monetary cost.
- Blinded audit packet: 40 hash-selected questions, 80 anonymous answers, **unreviewed**.
  Known limitation: Ettin's published checkpoint-selection recipe includes NanoBEIR
  HotpotQA/SciFact tasks; disjointness is not established. Jev/Luna training overlap is unknown.
- Raw records, source/code hashes, manifests, environment, and credential-free captures:
  `artifacts/research-v3/`. Methods and offline replay commands:
  `benchmarks/research/README.md`. This is locally frozen, not externally preregistered
  or peer reviewed. Current release CI status is separate from these frozen results.
- Final `make check-all` passed. Both dashboards passed desktop/mobile layout, JSON
  download, and navigation checks; wheel and sdist contain the new research assets.
- The 14.7 MB reproduction bundle contains 39,795 files and passed a scan for configured
  credentials. Extracted into a separate directory, installed a new locked environment,
  downloaded the public datasets anew, and recomputed both studies without paid calls.
  Reports, rankings, per-query metrics, verification records, and the raw artifact manifest
  reproduced **byte for byte**. Evidence: `artifacts/research-v3/reproduction-check.json`.

## Contextual scoring and public benchmarks

- `make check-all`: **71 Python tests passed**, **1 opt-in paid test skipped**, **6 TypeScript
  tests passed**. Ruff, formatting, strict mypy, TypeScript compilation, three Chromium
  flows, and both scoring strategies through the real Dify SDK passed.
- Added optional contextual shared-state scoring, separate prompt version, bounded candidate
  and estimated token budgets, accurate shared usage, replay preservation, strategy-change
  invalidation, and Python/HTTP/TypeScript/LangChain/Dify controls. Independent mode remains
  the default. Older pilot replays retain their original independent-scoring fingerprints.
- A real Chromium session scored connected passages through Jev, generated both Luna answers,
  exported/imported the replay, verified invalidation and mobile layout, with no page errors.
  Replay: `artifacts/contextual-live-workbench.json`; screenshot: `artifacts/contextual-live-answers.png`.
- Frozen evaluation: **100 HotpotQA + 100 MuSiQue-Ans validation questions**, separate from
  **40 development questions**. **600 evaluation + 160 development generations** completed.
  Local EM/F1 matched the authors' metric implementations on all **760 predictions**.
- Observed F1: HotpotQA **77.97 → 80.08**; MuSiQue **64.30 → 64.76**. Estimated total API
  cost versus all context fell **26.7% / 23.1%**. **Both accuracy-difference confidence
  intervals include zero; the strong success gate was not met.** Wording differences explain
  some metric changes. No general factual-accuracy, latency, or query-expansion claim.
- The `/benchmarks` dashboard displays measured costs, baselines, and uncertainty. All 200
  comparison replays, including wins and regressions, were built and contract-validated.
  See [results](benchmarks/public/RESULTS.md), [reproduction](benchmarks/public/README.md),
  and [demo walkthrough](docs/BENCHMARK_DEMO.md).
- Python wheel/sdist and updated Dify plugin rebuilt successfully. Public raw datasets and
  credentials remain outside the distributable artifacts.

## Luna configuration follow-up

- **Resolved:** the replacement project key successfully generated a cited Luna smoke
  answer. The complete ten-case live pilot then produced 18 Luna answers and two
  application no-context fallbacks with no provider errors. All replay contracts validated.
- Pilot results: 60.1% fewer estimated passage tokens, but 49.7% higher estimated total
  stage cost including Jev. Review found a citation/context regression and retrieval gaps.
  See [the full pilot report](benchmarks/pilot/RESULTS.md); no general accuracy improvement
  is established. The original fixed inputs and cutoff were not changed.
- Real Chromium imported and rendered the live Luna replay with no page errors or mobile
  overflow. Service restarted with the funded key, low reasoning effort, a 2,000-token
  budget, and published-rate cost estimates.

- `make check` after generation changes: **63 Python tests passed**, **1 paid test skipped**,
  **5 TypeScript tests passed**; lint, format, mypy, compilation passed.
- Added endpoint-scoped `OPENAI_API_KEY` fallback, optional reasoning effort, recorded
  generation settings, safe quota diagnostics, and usage retention for empty paid responses.
  OpenAPI and TypeScript schemas regenerated.
- The configured `gpt-5.6-luna` request to OpenAI failed with HTTP **429**, upstream type
  **`insufficient_quota`**. No live answer was generated. The project-specific generation
  key remained empty; the attempted credential came from the environment fallback.
- Prepared ten questions and 120 BM25-retrieved candidates against eight MIT-licensed
  FastAPI documentation files, pinned to a Git commit and SHA-256 checksums in
  `benchmarks/pilot`. These are frozen inputs, not benchmark results. Source checksums and
  all ten selection request contracts were validated. Human review of labels is pending.
- Restarted the updated local service at `http://127.0.0.1:8765`.

## Final acceptance results

- `make check-all`: **58 Python tests passed**, **5 TypeScript client tests passed**.
  Ruff lint/format, strict mypy, TypeScript compilation, and official Dify SDK integration passed.
- The normal suite skips the opt-in paid test. Separately running
  `RAG_JEV_LIVE=1 uv run pytest tests/test_live.py -q`: **1 passed**.
- Real Chromium tests exercised authenticated scoring, comparison through a controlled HTTP
  generation endpoint, cached policy changes, answer invalidation, export/import, empty-context
  fallback, literal rendering of HTML-like source text, and mobile layout.
- A separate real-browser run called **Jev 1.13.0**, with no page errors or mobile overflow.
  Screenshots: `artifacts/workbench-live-desktop.png`, `artifacts/workbench-live-mobile.png`.
- Real local Markdown retrieval through SQLite FTS5/BM25 retrieved **5 passages** and Jev selected
  **1** at the illustrative 0.20 cutoff. Shadow mode preserved the original retrieval output.
  The recorded scoring time was **810.504 ms** for this single request, not a latency benchmark.
  Replay: `artifacts/local-rag-live.json`.
- LangChain sync/async/LCEL tests preserve original document objects, duplicate source IDs,
  non-JSON metadata, and explicit grouping. A separate **LangChain → service → real Jev**
  request returned the expected direct-answer source.
- Dify **0.7.4 SDK** discovered the native plugin and invoked its reranker through real HTTP,
  testing original indices, thresholds, empty results, and authentication failures.
  A separate **Dify SDK → service → real Jev** request returned the expected original index.
- Official Dify CLI **0.6.10** packaged `dist/rag-jev.difypkg` successfully. No full Dify
  application/server was deployed. The SDK emits a non-fatal gevent/logging cleanup warning
  when standalone check processes exit; invocation and packaging completed successfully.
- `uv build` succeeded. The wheel was installed in an isolated environment; its packaged
  playground, assets, fixture selection, and replay import passed smoke checks.
- Wheel, source archive, and Dify package were inspected: no `.env`, virtual environment,
  Python caches, or live artifacts were included. OpenAPI and TypeScript types were regenerated.

## Boundaries

The live model connection and ten-case pilot are complete. Generation requests, token
usage, citation mapping, empty-context fallback, cost arithmetic, failure handling, and
browser comparison were tested against controlled HTTP responses and the pilot used real
Luna answers. **General answer-quality improvement is not established.** The pilot has
agent review, not independent human grading. Citation IDs are checked; factual support is
not automatically graded.

The prior live receipt-supporting passage failure remains visible in the playground. Pins and
atomic groups can preserve caller-identified related evidence, but neither filtering nor
reranking has a demonstrated general quality advantage from these tiny tests.

CI is configured for Python 3.11/3.13, Chromium, and isolated Dify SDK checks, but has not run
remotely. Local main-project checks used Python 3.13; Dify checks used Python 3.12.

## Try it

The updated service is available at `http://127.0.0.1:8765`; use the root URL for the
playground and `/docs` for the API. Restart from the repository root with:

```sh
uv run rag-jev serve --port 8765
```

Follow [the real-RAG guide](docs/REAL_RAG.md) to test your own files, insert the LangChain
adapter, install the Dify plugin, or connect an answer model. The accepted implementation
scope is listed in [MVP_SCOPE.md](MVP_SCOPE.md).

---

# Earlier backend-only verification — September 18, 2026

## Checks completed

- `make check`: **45 Python tests passed**, **5 TypeScript client tests passed**,
  Ruff lint/format checks passed, strict mypy checks passed, TypeScript compilation passed.
- The regular suite skips the paid live test by design. Running it separately with
  `RAG_JEV_LIVE=1 uv run pytest tests/test_live.py -q`: **1 passed**.
- Network integration exercised a real local HTTP connection from the TypeScript
  client through the Python service and official TypeSafe SDK to a controlled
  upstream endpoint. It checked selection, shadow mode, empty results, authentication,
  invalid input, provider failure, and deadlines.
- A separate TypeScript request through the running service reached **real Jev**,
  resolved as **`jev-1.13.0`**, and returned a valid filtered result with source metadata.
- Python wheel installed in a clean isolated environment and ran the offline demo.
- npm tarball installed in a temporary project and imported successfully.
- Source archive checked to exclude credentials, local agent metadata, and artifacts.
- Exported OpenAPI checked against the application schema. Credentials and live
  artifacts are ignored by Git.

The only regular-suite warning is an upstream Starlette/AnyIO deprecation notice.
No application test failed. Local verification used Python 3.13; CI also targets Python 3.11.

## Real Jev smoke evaluation

Three tiny hand-labeled examples, eight passages total. These are integration
examples, **not a representative quality benchmark**. Live scoring consumed 3,292
input and 176 output tokens. The observed maximum/p95 across these three requests
was about 834 ms; this small sample does not establish production latency.

At the illustrative `0.20` cutoff:

- Direct refund evidence survived; shipping information was removed.
- Both passages correcting/explaining the subscription premise survived.
- The unanswerable Wi-Fi query returned no context.
- The receipt requirement, labeled as supporting evidence, scored `0.12` and was
  dropped. Query-averaged evidence recall on answerable cases was **0.75**.

The irrelevant Wi-Fi passage scored `0.15`, illustrating why lowering one global
threshold does not resolve every false negative without admitting noise. The
implementation exposes this tradeoff instead of presenting the score as a guarantee.
Final generative-answer quality was not evaluated; the Python evaluation API has
an application-provided generation/grading callback for that purpose.

Local evidence (ignored by Git):

- `artifacts/live-report.json`
- `artifacts/live-traces.json`
- `artifacts/live-threshold-sweep.json` — four cutoffs, replayed without new API calls
- `artifacts/live-typescript-response.json`

## Running locally

For local live testing use `http://127.0.0.1:8765`. The key-free replay server uses port 8766.
Docs: `http://127.0.0.1:8765/docs`. If the session process has ended, restart with:

```sh
uv run rag-jev serve --port 8765
```

The self-paced build/test loop is complete. No recurring scheduler was installed.
## Research round 2 verification

- Frozen 240 development / 400 new evaluation questions; normalized-question splits
  are disjoint. Protocol: `benchmarks/public/research-v2-protocol.json`.
- Tested two scoring prompts, six cutoffs each, and relevance ordering for the original
  prompt finalist. Frozen candidate: contextual, cutoff 0.20, filter and rerank, no cap.
- Completed 1,918 new generation calls and 640 new Jev scoring calls; two evaluation
  branches selected no context and correctly made no generation call. No errors,
  truncations, missing token usage, or missing cached-token usage in the final evaluation.
  Incremental API cost estimated from returned usage: $0.63843, excluding prior reused
  traces and charging each actual scoring call once. This is not an invoice.
- All 2,400 development/evaluation answer records match the authors' EM/F1 functions;
  all 1,520 contextual selections match production selected IDs and ordering. Evidence:
  `artifacts/research-v2/verification.json`.
- Published all 400 validated replay bundles, measured chart, JSON report, and dashboard
  at `/assets/research-v2.html`. The original benchmark report and defaults remain intact.
- The evidence-retention improvement is supported by positive paired bootstrap intervals
  versus the old profile on both datasets. Answer-F1 intervals include zero; no conclusive
  accuracy improvement or universal cost win is claimed. See
  [full results](benchmarks/public/RESEARCH_RESULTS.md) and [failure analysis](benchmarks/public/RESEARCH.md).
