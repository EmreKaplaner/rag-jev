# A 50-second demo backed by the controlled study

Record the actual local pages. An AI voice or editor can provide narration and captions;
keep benchmark tables and generated answers as captured evidence. Do not substitute
synthetic screenshots or fabricated model output.

| Time | Screen | Narration |
| --- | --- | --- |
| 0–8 s | `/assets/retrieval-v1.html`, all three rows visible | “Same retriever. Same twenty candidates. Can a better reranker find the evidence?” |
| 8–18 s | SciFact scores and scope note | “Across all 300 SciFact test queries, Jev scored 75.13 NDCG at ten, versus 72.11 for Ettin and 66.47 for BM25.” |
| 18–30 s | `/assets/research-v3.html`, primary intervals and cost table | “We also ran 7,200 answer evaluations on 400 isolated questions. Estimated API cost fell about twenty to twenty-three percent. Answer scores rose, but the uncertainty still includes no improvement.” |
| 30–40 s | Workbench: strategy, filter/rerank control, threshold | “Reranking helped one dataset and hurt another. That's why filtering and ordering are configurable.” |
| 40–50 s | Downloadable results/protocol links, then API example | “Drop it after retrieval. Test on your own data. The results, baselines, and raw evidence are inspectable.” |

Keep these captions visible on the relevant screens:

- Retrieval: **BEIR/SciFact test · 300 queries · BM25 top20 → top10 · exploratory · no answer generation**.
- Answers: **200 questions per dataset · 3 answer samples per arm/question · estimated uncached API cost includes Jev · accuracy superiority not established**.

The SciFact Jev-versus-Ettin NDCG difference is +3.01 points, with a descriptive 95%
cluster-bootstrap interval of [+0.61, +5.41]. This is not a full BEIR-suite or state-of-the-art
claim. The neural baseline has potential benchmark-selection exposure, and proprietary
training exposure is unknown. Link the methods alongside the video.

Do not claim lower cost than Ettin: its generation API usage and measured CPU time are
reported separately, and its total hosting cost is unknown. The answer-study cost savings
are versus passing all candidate passages to Luna. Do not describe this as proven semantic
accuracy or faithfulness; independent human adjudication is pending.

The report screenshot and plot are ready to use under `artifacts/research-v3/`; the retrieval
screenshots are under `artifacts/retrieval-v1/`. For a visual walkthrough of individual
answers, earlier real workbench replays remain available, but label them as illustrative
examples from the earlier study. Do not present them as this round's aggregate evidence.
