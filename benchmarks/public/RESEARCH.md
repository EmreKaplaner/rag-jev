# Research round 2: evidence retention versus cost

This round tests three changes: a lower relevance cutoff, a shorter evidence-chain
scoring prompt, and relevance ordering after filtering. It does not change the answer
model, its prompt, reasoning effort, or output budget between branches.

The 240 questions examined in round 1 are **development data for round 2**, including
the old evaluation questions. Round 1's published results remain historical observations
of its frozen policy. They are not a held-out test of this round's improvements.

Before running new scoring experiments, `research prepare` freezes 200 new questions
per dataset. Selection uses SHA-256 ordering with a new fixed seed. Normalized question
duplicates are excluded across all old cases and both new evaluation datasets. The
source checksums are unchanged from round 1. These are public validation subsets with
supplied candidate passages, not full-corpus retrieval or official test leaderboard runs.

## Predeclared experiment

1. Score all development passages using the original contextual prompt and the shorter
   prompt. Original scoring traces are reused with source hashes.
2. Sweep cutoffs 0.05, 0.10, 0.15, 0.20, 0.25, and 0.35. For each scorer, shortlist the
   setting using the fewest passage tokens at at least 97% macro support recall.
3. Generate answers for those two finalists and an additional relevance-ordered version
   of the original-prompt finalist. Compare against all context and the old 0.35 profile.
   Development baseline answers are reused, with their original usage and source hashes.
4. Among candidates costing less than all context **on each dataset**, freeze the highest
   macro development answer F1. Break ties by total API cost. No eligible candidate means
   no promotion.
5. Evaluate that one frozen candidate, the old profile, and all context on all 400 new
   questions. Every evaluation answer is newly generated. Rotate branch order by case.
   Preserve errors, empty selections, and truncations; no selective answer retries.

The new scorer and selector see only the question and passage text, never supporting
labels, reference answers, question decomposition, or dataset identity. Selection has
no access to labels. Labels are used only for development calibration and evaluation.

Reported API cost charges each branch its entire scoring call plus generation usage,
even when experiments share a cached trace. Cached generation input gets the provider's
reported discount. Rates are frozen in the protocol. Common retrieval, hosting, network
costs, and invoice adjustments are excluded. Scoring transport may retry once under the
existing SDK policy; usage for an unreported failed attempt cannot be recovered.

Answer F1 uses the authors' normalization and token overlap rules, with aliases for
MuSiQue. Paired bootstrap intervals use 2,000 resamples. Supporting-passage recall and
the fraction of questions retaining every annotated supporting passage are secondary
diagnostics, not substitute answer-accuracy measures. Annotations can be redundant;
answer F1 can change with wording. Model training overlap is unknown.

## Reproduce

Prepare the original downloaded sources first, using the source/checksum instructions
in [README.md](README.md). This round also requires round 1's raw development/evaluation
traces for its declared reuse; those private local artifacts are not packaged in git.
Configure the Jev and generation credentials in `.env`; never put keys in commands.

```sh
uv run --group benchmark python -m benchmarks.public.research prepare
uv run --group benchmark python -m benchmarks.public.research score
uv run --group benchmark python -m benchmarks.public.research shortlist
uv run --group benchmark python -m benchmarks.public.research answer
uv run --group benchmark python -m benchmarks.public.research freeze
uv run --group benchmark python -m benchmarks.public.research score --split evaluation
uv run --group benchmark python -m benchmarks.public.research answer --split evaluation
uv run --group benchmark python -m benchmarks.public.research report --split evaluation
uv run --group benchmark python -m benchmarks.public.research_verify
uv run --group benchmark python -m benchmarks.public.research_publish
```

Each paid stage resumes completed, fingerprint-matching cases. The prepare and freeze
stages refuse to overwrite existing frozen records. `research-v2-protocol.json` and
`research-v2-frozen.json` provide the checked-in protocol and selected policy. Raw cases,
scores, answers, usage, and reports live in `artifacts/research-v2/`.

The verification command also requires the author metric snapshots used in round 1:
`artifacts/benchmark-research/hotpotqa-official-metrics.py` from
<https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py> and
`musique-official-metrics.py` from
<https://raw.githubusercontent.com/StonyBrookNLP/musique/main/metrics/answer.py>.
Their SHA-256 hashes are recorded in `artifacts/research-v2/verification.json`. Verification
checks every answer against these functions and compares selected passage IDs/order with
the production policy implementation. Publication exports all 400 replay files.

## What this round established

[The complete results](RESEARCH_RESULTS.md) show significantly better supporting-passage
retention versus the old filter, but no conclusive answer-accuracy improvement. The new
profile costs more than the old filter and less than all context. It is optional; the
default and the original scorer prompt are unchanged.

On the new questions, the profile had 9 F1 wins / 11 losses / 180 ties against the old
profile on HotpotQA, and 21 wins / 14 losses / 165 ties on MuSiQue. There were also 8
HotpotQA and 20 MuSiQue questions where it scored below all context **despite retaining
all annotated supporting passages**. Evidence recall alone does not solve answer quality.

Examples retained for diagnosis, not representative showcase claims:

- MuSiQue `4hop1__749065_698949_157828_239539`: old profile lost support and answered
  `unanswerable`; the new profile retained all support and answered `Pristina` correctly.
- HotpotQA `5a80abd15542996402f6a5e0`: both retained all annotated support, but the old
  answer `Ding Sheng` was correct and the new answer `Zhang Yimou` was wrong.
- MuSiQue `2hop__68162_482364`: the new profile restored all annotated support but changed
  the correct answer `Aeson` to `unanswerable`.

The next useful research question is how evidence order and extra distractors affect
generation when support is already present. This round does not isolate those effects:
the held-out candidate changes both cutoff and ordering. A future ablation needs a fresh
evaluation set or must explicitly treat these 400 now-observed questions as development.
Do not tune on these results and relabel them as another held-out win.
