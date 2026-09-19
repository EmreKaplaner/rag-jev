# Controlled research results — round 3

The predeclared accuracy-and-cost superiority gate is not met.

| Dataset | Pipeline | Answer F1 | EM | All support retained | Uncached API $ / 1,000 | CPU seconds / question | Answer disagreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HotpotQA | all_context | 76.70 | 62.00 | 100.0% | 0.3472 | 0.000 | 16.5% |
| HotpotQA | jev_0.2_original | 78.53 | 64.83 | 99.0% | 0.2797 | 0.000 | 13.5% |
| HotpotQA | jev_0.2_rank (frozen candidate) | 77.20 | 63.00 | 99.0% | 0.2788 | 0.000 | 14.0% |
| HotpotQA | jev_0.35_original | 78.88 | 64.83 | 96.5% | 0.2526 | 0.000 | 13.5% |
| HotpotQA | jev_0.35_rank | 76.33 | 61.33 | 96.5% | 0.2516 | 0.000 | 12.5% |
| HotpotQA | ettin_top8 | 76.24 | 62.83 | 95.0% | 0.2810 | 0.409 | 12.5% |
| MuSiQue-Ans | all_context | 70.31 | 60.83 | 100.0% | 0.6197 | 0.000 | 20.0% |
| MuSiQue-Ans | jev_0.2_original | 68.79 | 59.50 | 95.5% | 0.4798 | 0.000 | 22.5% |
| MuSiQue-Ans | jev_0.2_rank (frozen candidate) | 72.80 | 63.33 | 95.5% | 0.4783 | 0.000 | 21.0% |
| MuSiQue-Ans | jev_0.35_original | 67.29 | 59.17 | 85.5% | 0.4227 | 0.000 | 21.5% |
| MuSiQue-Ans | jev_0.35_rank | 67.53 | 59.83 | 85.5% | 0.4244 | 0.000 | 21.0% |
| MuSiQue-Ans | ettin_top8 | 54.32 | 46.33 | 63.5% | 0.3262 | 0.558 | 20.5% |

400 isolated evaluation questions, three samples per pipeline/question, 7,200 evaluation branches. The independent sample size is 200 questions per dataset, not the number of generations. All-context, a full cutoff/order factorial, and development-tuned Ettin are evaluated on the same questions. Errors: 0; truncations: 0; unknown-usage records: 0.

HotpotQA: frozen candidate versus all context, F1 difference +0.50 points, simultaneous 97.5% interval [-2.72, +3.69]; Holm-adjusted one-sided p=0.3595. Normalized API cost 19.7% lower.

HotpotQA: candidate versus Ettin, descriptive F1 difference +0.96 points, unadjusted 95% interval [-1.31, +3.35]. CPU-cost break-even: $-0.0195 per CPU-hour, excluding other hosting costs. A negative break-even means Ettin generation alone already costs more than the candidate's API bill.

HotpotQA: lowering the cutoff, averaged over both orders, descriptive F1 difference +0.26 points; unadjusted 95% interval [-1.19, +1.65].

HotpotQA: relevance ordering, averaged over both cutoffs, descriptive F1 difference -1.94 points; unadjusted 95% interval [-3.91, -0.15].

HotpotQA evaluation composition — bridge: 166, comparison: 34.

MuSiQue-Ans: frozen candidate versus all context, F1 difference +2.50 points, simultaneous 97.5% interval [-1.31, +6.37]; Holm-adjusted one-sided p=0.1531. Normalized API cost 22.8% lower.

MuSiQue-Ans: candidate versus Ettin, descriptive F1 difference +18.48 points, unadjusted 95% interval [+13.24, +23.92]. CPU-cost break-even: $0.9808 per CPU-hour, excluding other hosting costs. A negative break-even means Ettin generation alone already costs more than the candidate's API bill.

MuSiQue-Ans: lowering the cutoff, averaged over both orders, descriptive F1 difference +3.39 points; unadjusted 95% interval [+1.22, +5.78].

MuSiQue-Ans: relevance ordering, averaged over both cutoffs, descriptive F1 difference +2.13 points; unadjusted 95% interval [+0.27, +4.17].

MuSiQue-Ans evaluation composition — 2hop: 170, 3hop1: 23, 3hop2: 2, 4hop1: 3, 4hop3: 2.

Primary intervals use question-paired bootstrap (10,000 draws) with simultaneous Bonferroni coverage for two endpoints. One-sided sign-flip tests use 20,000 draws and Holm correction. Secondary factorial, neural and subgroup intervals are descriptive. A nonsignificant difference does not establish equivalence.

Costs are returned-usage estimates at uncached token rates, including Jev. Ettin's listed dollars cover GENERATION ONLY; CPU cost is separate and its total monetary cost is unknown. Observed cache-discounted API costs are in the JSON report. Common retrieval, application hosting and network costs are excluded.

Strict source-component isolation changes the validation population. These are not full benchmark or corpus-retrieval results. Answer F1 is token overlap, not semantic or citation correctness. Human adjudication has not been completed. Proprietary-model training contamination and weights cannot be independently verified. Ettin's published checkpoint-selection procedure uses NanoBEIR, including HotpotQA and SciFact; disjointness from those selection examples is not established. Scoring is fixed per question; repeats measure conditional generation variance. No query expansion claim. Secondary findings cannot be used to replace the frozen candidate and claim a new held-out win. In particular, the small higher-hop MuSiQue subgroups cannot support precise higher-hop accuracy claims.

[Design, sources and reproduction](README.md). Raw records, audit packets and integrity manifest: `artifacts/research-v3/`.
