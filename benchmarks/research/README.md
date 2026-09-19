# Controlled context-selection study, round 3

This is a reproducible, locally predeclared experiment—not an external preregistration,
peer-reviewed result, or proof that the product improves every RAG pipeline.

## Research questions

1. Does the development-selected Jev policy improve answer F1 **and** lower normalized
   API cost versus all context on both isolated evaluation subsets?
2. Separately, what are the effects of lowering the cutoff and changing passage order?
3. How does Jev compare with a pinned neural cross-encoder, after tuning that baseline
   on the same development questions?
4. How much answer variability remains when context selection is held fixed?

The design follows the motivation of [Lost in the Middle](https://arxiv.org/abs/2307.03172)
for position sensitivity; that paper does not establish an effect for our model or inputs.
The stronger baseline is [Ettin-32M](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1),
revision `b33e5ceb5110773ea9cf5e00c9bedc83a8c2afdd`, using its documented Sentence
Transformers interface. The [NLP significance-testing reference](https://aclanthology.org/P18-1128/)
informs paired comparisons. Our precise decision rules are in `protocol.json`.

## Controls and isolation

- All 640 previously observed questions are marked exposed in `prior-exposure.json`.
  Development uses 48 per dataset, deterministically chosen from those exposed IDs.
- Evaluation uses 200 new questions per dataset. MuSiQue questions share **no single-hop
  component IDs** with previous cases or another evaluation question. HotpotQA questions
  share **no supporting titles** with previous cases or another evaluation question.
  Normalized questions are also deduplicated. This stricter sample is a different
  population; do not pool it with earlier rounds or compare headline scores across rounds.
- MuSiQue's author archive was downloaded and its validation file matched the mirror
  byte for byte. Source hashes and author URL are in `source-verification.json`.
- Every answer is newly generated. No historical baseline-answer reuse. Development has
  two samples per pipeline/question; evaluation has three. The API's sampling defaults
  apply; no seed or temperature is sent. Model-request names and actual returned model
  names are recorded. Proprietary weights and pretraining overlap cannot be independently
  verified. A mutable API alias is not an immutable model snapshot.
- The question and whole passage text are identical across arms before selection.
  No answer references, decomposition, labels, or dataset names enter the real selectors.
  The oracle arm intentionally uses labels, is marked diagnostic, appears only in
  development, and cannot be selected for evaluation or production.
- Jev scores are sampled once per question and shared across arms/repetitions. Repeated
  answers measure generation variance conditional on those scores, not all sources of
  end-to-end pipeline variance. Neural inference is pinned, CPU float32, no truncation.

## Arms and selection

The development candidates cross cutoffs 0.20/0.35 with original, descending-relevance,
and bookend order. Bookending puts the highest-ranked passage first and the second highest
last; it changes no text and retains exactly the same passages as descending order.

The highest macro answer-F1 Jev candidate that is cheaper than all context on each dataset
at uncached token rates is frozen. Ties favor lower cost, then name. If none qualifies,
the best F1 candidate is evaluated with the failed eligibility flag; no promotion is allowed.
Ettin's top-k is independently selected for highest development F1 among 3, 5, and 8,
with smaller k breaking ties. It is not deliberately held to an unfavorable fixed k.

Evaluation includes all context, the complete cutoff/order 2×2 factorial, the frozen Jev
candidate if distinct, and the selected Ettin arm. The bookend variants cannot be chosen
after seeing evaluation results. Jobs are deterministically shuffled across questions,
arms, and repetitions to reduce timing and cache-order confounding.

## Statistical analysis

The independent unit is the **question**, not a generation. Each question's three F1
values are averaged before paired inference. We report within-question F1 standard
deviation and normalized-answer disagreement separately.

The two primary endpoints are frozen-candidate-minus-all-context F1, one per dataset.
We use 10,000 paired question bootstrap draws and 20,000 sign-flip randomization draws
with a +1 Monte Carlo correction. Primary intervals are two-sided 97.5% intervals
(Bonferroni simultaneous coverage across the two endpoints); one-sided p-values receive
Holm correction. A superiority claim requires both lower bounds positive, both adjusted
p-values below .05, and both normalized-cost-difference upper bounds negative, with
complete usage and no errors/truncations. Positive point estimates alone do not pass.

Factorial main effects/interaction, neural comparisons, exact match, evidence metrics,
NDCG, and subgroup results are descriptive secondary endpoints. Their 95% intervals
do not authorize confirmatory claims. Small subgroups remain underpowered. The freeze
record includes an approximate 80%-power minimum detectable F1 difference estimated
from development paired variance. Failure to detect a difference is not equivalence.

Answer F1/EM use the benchmark authors' metrics. They measure token overlap/exact match,
not semantic correctness or citation faithfulness. A blinded review packet is prepared
for independent adjudication; unfilled human labels are never reported as completed work.

## Cost, latency and failure accounting

We report returned-usage API estimates with cache discounts and normalized estimates at
uncached rates. The latter prevent cache warmed by repeated evaluation from being called
a selection optimization. Each hypothetical pipeline pays its full Jev scoring charge
even though the experiment shares scores. Actual experiment spend charges shared scores
only once. Unknown usage is flagged; any subtotal excludes unknown charges and cannot
support a savings claim. Prices are frozen, not invoice figures.

Ettin's local CPU inference is **not free**. Its generation API cost, CPU seconds, wall
time, and break-even CPU-hour price versus Jev are shown separately; total local monetary
cost is unknown. Break-even excludes idle capacity and other hosting costs. Neither
CPU wall time versus a remote API nor concurrent-run latency establishes a hardware-neutral
speed advantage or a production SLA. Common retrieval/network/application costs are excluded.

No provider retries are configured in this round. Errors/empty contexts/truncations remain
in the denominator. Empty selections make no generation request. Started-attempt records
and allowlisted response captures prevent silent duplicate calls after interruption.
An orphaned attempt blocks automatic resume. A $10 returned-usage stop threshold bounds
normal running; in-flight requests and unreported charges are not an invoice hard cap.

## Reproduction

Run from the repository root. Keys belong in `.env`, never in commands or reports.

```sh
uv sync --locked --group benchmark --group research
uv run --no-sync python -m benchmarks.research.fetch_sources
uv run --no-sync python -m benchmarks.research.design
uv run --no-sync python -m benchmarks.research.runner score development
uv run --no-sync python -m benchmarks.research.neural development
uv run --no-sync python -m benchmarks.research.runner answer development
uv run --no-sync python -m benchmarks.research.statistics development --freeze
uv run --no-sync python -m benchmarks.research.runner score evaluation
uv run --no-sync python -m benchmarks.research.neural evaluation
uv run --no-sync python -m benchmarks.research.runner answer evaluation
uv run --no-sync python -m benchmarks.research.statistics evaluation
uv run --no-sync python -m benchmarks.research.verify
uv run --no-sync python -m benchmarks.research.publish
```

Preparation reconstructs the sample from pinned source datasets and the checked-in
exposure registry. It needs no private previous-run predictions. The machine-local
`artifacts/research-v3/reranker-source.json` contains `model`, `revision`, and `card_url`
matching the protocol's neural source; it can be copied from the checked-in source file.
Frozen protocols and policies cannot be overwritten; fingerprint-matching inference
artifacts can be resumed. Core code and lockfile hashes are checked before paid stages.

## Remaining validity limits

This is context selection over benchmark-provided candidates, not a corpus-search test.
Gold evidence is present before selection, unlike many real retrieval failures. Only two
English multi-hop datasets and one answer model are tested. Single-hop IDs/titles reduce
local leakage but cannot detect semantic near-duplicates, shared Wikipedia facts across
datasets, or model-training contamination. Prompts and thresholds may not transfer to
private enterprise documents. There is no query-expansion result in this experiment.

The [Ettin authors' training recipe](https://huggingface.co/blog/ettin-reranker) selects
checkpoints using NanoBEIR, including HotpotQA and SciFact tasks. We have not established
disjointness from those model-selection examples. Thus the baseline is not demonstrably
benchmark-naive. Jev/Luna training exposure remains unknown. Local experiment isolation
does not remove these upstream model-selection and pretraining limitations.

## Separate corpus-retrieval check

`retrieval.py` freezes a separate, exploratory evaluation on the entire **BEIR/SciFact
test split: 300 queries, 5,183 corpus documents**. A custom indexed BM25 implementation
retrieves 20 candidates from the full corpus; the exact same candidates go to Jev and
Ettin, and each returns 10. The BM25 formula/tokenizer is unit-checked against the earlier
baseline. There is no gold insertion, cutoff tuning, answer generation, or opportunity
to alter the main study's frozen candidate. Source ZIP MD5 matches BEIR's published value;
SHA-256 is also frozen.

NDCG@10, Recall@10 and MRR@10 are verified against `pytrec_eval`. Relevant passages
missing from the candidate pool remain in the denominator. Paired descriptive intervals
resample connected components of queries sharing judged-relevant papers, preserving the
query-weighted mean. This is an external-validity check for retrieval, not an answer-
accuracy result, a full BEIR-suite result, or a new primary endpoint.

```sh
uv run --no-sync python -m benchmarks.research.retrieval prepare
uv run --no-sync python -m benchmarks.research.retrieval jev
uv run --no-sync python -m benchmarks.research.retrieval neural
uv run --no-sync python -m benchmarks.research.retrieval report
uvx --from pytrec-eval-terrier==0.5.10 python benchmarks/research/verify_retrieval.py
uv run --no-sync python -m benchmarks.research.publish_retrieval
```

`fetch_sources` downloads the checksum-pinned `scifact.zip` and its URL/SHA-256 source record
under `artifacts/retrieval-v1/`. Raw rankings, qrels, per-query metrics and the separate report
stay in that directory. `uvx` runs the independent metric implementation in an isolated
environment without changing the frozen project lockfile.

## Audit existing results without paid calls

The bundle includes the actual requests/responses (without credentials), all arm records,
frozen inputs and code, lockfile, source manifests, and an unreviewed blinded audit packet.
It excludes large source datasets and model weights; `fetch_sources` retrieves and verifies
the public data. Extract into a new directory, never over a working project:

```sh
shasum -a 256 -c reproduction.sha256
mkdir research-audit
tar -xzf reproduction.tar.gz -C research-audit
cd research-audit
uv sync --locked --group benchmark
uv run --no-sync python -m benchmarks.research.fetch_sources
uv run --no-sync python -m benchmarks.research.statistics evaluation
uv run --no-sync python -m benchmarks.research.verify
uv run --no-sync python -m benchmarks.research.retrieval report
uvx --from pytrec-eval-terrier==0.5.10 python benchmarks/research/verify_retrieval.py
```

These commands analyze saved predictions and make no paid inference calls. For a new
stochastic replication, use a fresh checkout/directory and the full reproduction sequence
above; do not remove failed attempts or overwrite the original study. Scores may differ
because API model aliases and generation sampling are not immutable.

The blind packet contains 40 deterministically sampled evaluation questions with two
anonymized answers each. Give reviewers `blind-review.json` and `review-labels.csv`, keeping
`review-key.json` hidden. Labels are currently blank. The packet assesses semantic answer
correctness and question ambiguity, not citation or selected-context faithfulness.
