# Public benchmark evaluation

This evaluates context selection on **HotpotQA distractor** and **MuSiQue-Ans** using
their supplied passages and reference answers. It does not evaluate search over the full
Wikipedia corpus or query expansion. These are fixed subsets of public validation sets,
not official leaderboard submissions. See [results](RESULTS.md).

## Reproduce

From the repository root:

```sh
uv sync --locked --group benchmark
uv run python -m benchmarks.public.download
uv run --group benchmark python -m benchmarks.public.prepare
# Real paid Jev and Luna calls; credentials come from .env.
uv run python -m benchmarks.public.run development
uv run python -m benchmarks.public.report development --freeze
uv run python -m benchmarks.public.run evaluation
uv run python -m benchmarks.public.report evaluation
uv run --group benchmark python -m benchmarks.public.publish_report
```

Already prepared inputs cannot be overwritten by `prepare`. Paid jobs resume from validated
cached records; they do not silently replace failed answers. To reproduce independently,
use a fresh checkout/output directory. Do not selectively delete losses and rerun them.
The original raw observations remain in `artifacts/public-benchmark/` (Git-ignored).

The checked-in `protocol.json` records source URLs, immutable Hugging Face revisions,
SHA-256 checksums, deterministic sampling seed, model settings, and evaluation criteria.
`frozen-policy.json` records the policy selected before evaluation and development results.
Downloads are approximately 58 MB. Raw benchmark data is not redistributed in the package.

## Method

- SHA-256 ordering picks 20 development and 100 disjoint evaluation questions per dataset.
  Normalized duplicate questions are excluded across selected questions. Benchmark and
  generator training-data overlap is unknown.
- Each paragraph retains its article title. Gold answers, supporting-fact labels, and
  MuSiQue decompositions never enter scoring or generation prompts.
- Development scores compare independent and contextual Jev, thresholds 0.1/0.2/0.35/0.5,
  and rerank top-3/5/8. For each scorer, shortlist the least passage tokens at >=95% mean
  support recall (or highest recall if none qualify). Generate answers for both shortlisted
  policies, all context, and BM25 top-five. Choose highest development F1 among candidates
  cheaper than all context; freeze the choice before reading evaluation results.
- All branches use GPT-5.6 Luna, low reasoning effort, 2,000 completion tokens, and the
  same short-answer-only system prompt. This differs from the workbench's citation prompt.
  Rotate branch order deterministically. One generation per branch; no answer-judge LLM.
- Use official normalized answer EM/token F1, maximum over MuSiQue aliases, including
  Hotpot's yes/no/noanswer rule. The local scorer was checked against the authors' metric
  functions on all 760 development/evaluation predictions.
- Report paragraph-support recall separately. This is not Hotpot's sentence-support F1
  or joint leaderboard metric. Errors count as zero answer scores and remain in reports;
  missing usage prevents a complete cost/success claim.
- Costs include actual reported Jev plus generation usage. Apply cached-input discounts
  when reported. Exclude common upstream retrieval, hosting, and networking; do not call
  these provider invoices or complete infrastructure costs. Experimental reuse of a trace
  does not erase the scoring cost assigned to a deployed branch.
- Use 2,000 paired bootstrap resamples for F1 differences. The predeclared strong success
  gate requires positive lower 95% bounds and lower cost on each dataset versus all context.
  Report BM25 comparisons too; no claim against modern neural rerankers is supported.

## Dataset attribution

[HotpotQA](https://github.com/hotpotqa/hotpot): Yang et al., *HotpotQA: A Dataset for Diverse,
Explainable Multi-hop Question Answering* (EMNLP 2018), via the pinned
[hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa) dataset distribution.

[MuSiQue](https://github.com/StonyBrookNLP/musique): Trivedi et al., *MuSiQue: Multihop
Questions via Single-hop Question Composition* (TACL 2022), CC BY 4.0. Downloaded from a
pinned [dgslibisey/MuSiQue mirror](https://huggingface.co/datasets/dgslibisey/MuSiQue), not an
author-owned Hugging Face repository. The original authors distribute it via Google Drive.
Mirror checksum and schema are verified; byte identity to that original ZIP was not checked.

Jev's documented [shared-state API](https://docs.typesafe.ai/concepts/state) motivates the
contextual variant; it evaluates a separate question for each supplied passage.
That does not provide free-form query generation or guarantee correct relevance judgments.
## Research continuation

The later [research round](RESEARCH.md) uses all 240 round-1 questions as development
and evaluates its frozen candidate on 400 new questions. Its [results](RESEARCH_RESULTS.md)
are mixed: better supporting-evidence retention, inconclusive answer improvements,
and a cost tradeoff. Round-1 observations below remain unchanged.
