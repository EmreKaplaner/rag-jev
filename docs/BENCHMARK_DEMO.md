# Initial subset study: a 45-second demo

This script preserves the initial 100-question-per-dataset study. Use
[the current research demo](RESEARCH_DEMO.md) for the release and newer results.

Use the actual local workbench and saved API responses. Record the browser; an AI voice
can narrate these measured results. Generated animations should illustrate the flow, not
stand in for real model outputs or dashboards.

| Time | Screen | Narration |
| --- | --- | --- |
| 0–6 s | `/benchmarks`, full table with sample-size note visible | “Does cutting RAG context actually help? We tested it.” |
| 6–20 s | Main workbench: import the MuSiQue example below, show retained/dropped passages, then both answers | “Here, the full context produces the city. Jev keeps connected evidence, and Luna returns the person the question asks for.” |
| 20–32 s | Benchmark chart; show both datasets and the cost axis | “Across two fixed 100-question samples, estimated API cost fell 23 to 27 percent. Observed answer scores rose slightly.” |
| 32–40 s | Confidence intervals, then the regression replay below | “That isn't proof of better accuracy. There are regressions, and keyword filtering is cheaper. Every result is inspectable.” |
| 40–45 s | Scoring strategy control and API request example | “Drop it after retrieval, measure it on your data, and keep the evidence that matters.” |

Successful illustrative case (chosen after evaluation, not representative proof):

`artifacts/public-benchmark/replays/musique/2hop__747894_135844.json`

Question: “Whom is the birth place of Ravil Aryapov named after?”
Original answer: “Tolyatti.” Selected answer/reference: “Palmiro Togliatti.”
Estimated stage cost: $0.0005536 → $0.000481824, including Jev.

Regression to inspect:

`artifacts/public-benchmark/replays/musique/3hop2__57238_1926_54362.json`

Original answer: “John Bell Hood.” Selected answer: “William Sherman.”
The reference identifies John Bell Hood. Show this as a failure, not an improvement.

Benchmark answers use a short-answer prompt, so recorded citations were not requested.
Do not click “Generate both answers” and present that fresh citation-prompt run as the
original benchmark result. Imported replays are explicitly labeled in the product.

Keep this caption visible with aggregate results:

> 100 questions per dataset · fixed candidate sets · GPT-5.6 Luna · API cost includes Jev ·
> accuracy-difference 95% intervals include zero

Use the repository's results report and protocol as the linked evidence for an X post.
Do not claim “solved RAG,” “beats every reranker,” or query-expansion gains. Query expansion
is not implemented or evaluated here. A full published benchmark run and stronger neural
reranker baselines are further work; this demo shows measured subset results.
