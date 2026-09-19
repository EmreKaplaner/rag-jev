# Live Luna + Jev pilot — September 18, 2026

The full pipeline works with the funded project key: fixed BM25 retrieval candidates →
real Jev scoring → real GPT-5.6 Luna answers → importable comparison replays.
All ten requests completed without provider errors: 18 model answers and two explicit
application no-context fallbacks. No threshold or question was changed after seeing results.

This run demonstrates context reduction, **not an accuracy or cost win**. The filtered
pipeline cost more overall, and review found a factual/citation regression.

## Measured results

| Metric, across ten questions | Original candidates | Jev filter at 0.20 |
| --- | ---: | ---: |
| Passages sent to generation | 120 | 43 |
| Estimated passage tokens (`cl100k_base`) | 4,218 | 1,682 |
| Actual generation prompt tokens | 6,599 | 2,932 |
| Actual generation output tokens | 1,126 | 963 |
| Estimated total stage cost | $0.002671 | $0.003997736 |
| Median stage latency | 2.121 s | 2.551 s |
| Model-generated answers | 10 | 8 |
| Application no-context fallbacks | 0 | 2 |
| Unrecognized citation IDs | 0 | 0 |

Passage tokens fell **60.1%**. Generation prompt tokens fell **55.6%**, including two
skipped generation calls. Estimated stage cost rose **49.7%** after including Jev's
$0.002255736 scoring cost. All ten selected branches were individually more expensive.
These are tiny contexts and a low-cost generator; scoring overhead outweighed savings.
Zero unknown citation IDs does not mean every claim is supported by its cited passage.

Cost uses reported usage at published per-million rates checked September 18, 2026:
Luna input $0.20 / output $1.20 ([OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna));
Jev input $0.042 / output $0 ([TypeSafe pricing](https://typesafe.ai/blog/introducing-system-one-models-and-jev)).
These are estimates, not invoice totals; cache discounts and retrieval are excluded.
Connection smoke-test usage is excluded. Median Jev scoring time was 0.776 s.
There was one observation per branch, always baseline first; timings are exploratory,
exclude retrieval, and do not establish a reliable latency advantage or disadvantage.

## Answer review

The following is agent review against the fixed expected points and actual supplied
passages, not independent human grading or a validated accuracy score.

| Case | Review of both branches and selected evidence |
| --- | --- |
| CORS | Both correctly state the explicit-origin restriction from the docs. Selected answer retains the key passage. |
| Background tasks | Both recommend Celery for distributed heavy work. Selected answer additionally claims FastAPI runs async methods in a threadpool, citing an unrelated **file-upload** passage (score 0.50). This is a context/citation regression. |
| Upload | Both explain memory-to-disk spooling; the selected branch keeps the key evidence. |
| HTTP 204 | Both give the correct no-body rule, but retrieval missed the paragraph explicitly naming 204. The remaining generic no-body statements do not substantiate that specific claim. This is a retrieval/grounding failure in both branches. |
| Headers | Both explain underscore-to-hyphen conversion. Neither explicitly states header case insensitivity; that supporting sentence was absent from the retrieved candidates. |
| Optional query | Both give `None` as the default; selection retains explicit evidence. |
| Form and JSON | Both explain the encoding restriction. Selected answer adds a JSON-in-a-form-string workaround not established by supplied evidence. |
| Dependencies | Both explain per-request caching and `use_cache=False`; selection retains explicit supporting evidence. |
| Unknown company origin | Original model appropriately abstains. Jev retains zero passages and the application asks for more evidence. This is not a corrected baseline hallucination. |
| Unknown company upload limit | Same outcome: original model abstains; selected branch makes no model call. |

Full passage relevance labels were not created; **evidence recall remains unmeasured**.
The corpus contains answers that this simple paragraph/BM25 retriever sometimes misses.
Jev also accepts tangential or context-dependent snippets. Correct-looking answers can
therefore reflect model knowledge rather than retrieved support.

## Inspect it and demonstrate it

Open `http://127.0.0.1:8765` and import `artifacts/pilot/upload.json` for a concise example:
twelve passages become two, estimated passage tokens fall from 397 to 192, and both
answers describe spooling. Keep the cost visible: $0.0002214 becomes $0.000386934.
Then import `unknown-origin.json` to show the explicit no-context path, and
`background.json` to show that the inspector exposes a real failure.

Chromium verified real Luna replay rendering, citation links' presence through the answer
panels, four kept CORS passages, and mobile layout without overflow or page errors.
Screenshot: `artifacts/pilot-cors-answers.png`. This replay check made no additional
generation calls. It does not independently verify factual citation support.

The honest demo claim is: **inspect and reduce retrieved context, then measure whether
it actually helps your pipeline**. This pilot does not justify “solved RAG precision,”
“cheaper RAG,” or a general quality improvement. Next evaluate larger, representative
production contexts against the existing reranker, with source-aware chunks and independently
reviewed answer/citation labels. Freeze a held-out set before changing the policy.

## Artifacts and reproducibility

`protocol.json`, `cases.json`, `candidates.json`, and source hashes remain unchanged.
Each `artifacts/pilot/<case-id>.json` contains the original documents, Jev judgments,
policy, usage, timing, both answers, citations, and requested generation settings.
`artifacts/pilot/summary.json` contains all per-case metrics. All ten replay contracts and
input/source checksums passed validation. Local artifacts are Git-ignored.

Generator settings: `gpt-5.6-luna`, `reasoning_effort=low`,
`max_completion_tokens=2000`, identical system prompt in both branches. No answer judge
model was called. The two zero-context branches use application text, not Luna output.
