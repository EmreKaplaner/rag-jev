# Public benchmark results — September 18, 2026

Contextual Jev filtering produced higher observed answer scores and lower estimated API cost than sending all context on both fixed evaluation subsets. **The accuracy confidence intervals include zero: the predeclared strong success gate is not met.**

| Dataset (100 questions) | Pipeline | Answer EM | Answer F1 | API cost / 100 | Support recall |
| --- | --- | --- | --- | --- | --- |
| HotpotQA | All context | 65.00 | 77.97 | $0.034281 | 100.00% |
| HotpotQA | BM25 top 5 | 51.00 | 59.88 | $0.020512 | 78.50% |
| HotpotQA | Jev contextual | 69.00 | 80.08 | $0.025130 | 97.00% |
| MuSiQue-Ans | All context | 52.00 | 64.30 | $0.066016 | 100.00% |
| MuSiQue-Ans | BM25 top 5 | 26.00 | 34.46 | $0.027427 | 58.42% |
| MuSiQue-Ans | Jev contextual | 58.00 | 64.76 | $0.050748 | 87.67% |

HotpotQA: F1 difference **+2.11 points**, 95% paired bootstrap interval **[-2.08, +6.68]**; estimated API cost **26.7% lower**.
MuSiQue-Ans: F1 difference **+0.46 points**, 95% paired bootstrap interval **[-5.52, +6.63]**; estimated API cost **23.1% lower**.

Against BM25 top-five, Jev improved F1 by 20.20 points on HotpotQA and 30.30 on MuSiQue, but cost 22.5% and 85.0% more respectively. No single pipeline dominates every baseline.

All 600 evaluation generations completed without errors or truncation; token and cache usage were available for every call. Another 160 generations were used only for development. Official author metric functions matched all 760 locally scored predictions.

## What changed in the product

The optional `scoring_strategy="contextual"` gives Jev the candidate set as shared state, then asks a separate evidence-relevance question for each passage. This lets it consider connections and avoids repeatedly sending the query/state across independent requests. The tested cutoff was 0.35, chosen on 40 development questions and frozen before the 200-question evaluation. Independent scoring remains available and is still the default.

This measures filtering, not free-form query expansion. Reranking modes remain available, but the frozen evaluation policy preserves original passage order after filtering.

## Limits that matter

- Manual inspection found wording-driven metric changes: “England national team” →
  “England” earns more F1, while “television series” → “TV series” loses F1 despite similar
  meaning. There are factual wins and losses too. Do not equate these small aggregate F1
  gains with a demonstrated reduction in factual errors.
- These are deterministic public validation-set subsets, not full benchmark runs or official test-set leaderboard results. Benchmark contamination in either model is unknown.
- The input is each benchmark's supplied candidate set. No full-corpus retrieval or query expansion was measured. BM25 ranks within that set; it is not a modern neural reranker.
- Answer F1 is normalized token overlap; EM is normalized exact match. These do not establish factual citation support. The benchmark uses a short-answer prompt, unlike the normal UI.
- MuSiQue supporting-passage recall fell to 87.67%; all supporting passages survived only 73% of questions. Some correct answers may rely on model knowledge rather than retained evidence.
- One response per branch. Latencies are exploratory: median stage time rose from 1.134 to 1.376 seconds on HotpotQA and 1.459 to 1.823 seconds on MuSiQue. No speedup claim.
- API cost includes Jev scoring and Luna generation, with reported cached-input discounts. It excludes common upstream retrieval, hosting and networking, and is not an invoice.

Pricing: [OpenAI Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna), [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). Rates and settings are frozen in `protocol.json`.

## Demo claim and inspection

> On fixed 100-question samples of HotpotQA and MuSiQue, contextual Jev filtering used 23–27% less estimated API cost than passing every candidate to GPT-5.6 Luna, with higher observed answer scores. Accuracy uncertainty remains; results depend on the pipeline.

Do not present this as statistically proven accuracy improvement, a full-benchmark result, or a win against all rerankers. Keep the sample size and comparison baseline on screen.

Open `/benchmarks` for the measured dashboard. Import any file in `artifacts/public-benchmark/replays/<dataset>/<case-id>.json` into the main workbench to inspect raw scores, support labels, both answers, and costs. Benchmark answers use the frozen short-answer prompt; generating again in the UI uses its citation prompt and creates a separate demonstration, not a replacement benchmark observation.

[Reproduction, protocol, dataset attribution and limitations](README.md). Raw summaries are in `artifacts/public-benchmark/evaluation-report.json`; all per-case records remain under `artifacts/public-benchmark/evaluation/`.
