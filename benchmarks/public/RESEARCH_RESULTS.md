# Research round 2 results — September 18, 2026

The predeclared accuracy-and-cost gate versus all context is not met.

| Dataset (200 questions) | Pipeline | Answer EM | Answer F1 | API cost / 100 | Support recall | All support retained |
| --- | --- | --- | --- | --- | --- | --- |
| HotpotQA | All context | 58.00 | 75.08 | $0.033994 | 100.00% | 100.00% |
| HotpotQA | Old Jev profile | 62.00 | 77.57 | $0.025022 | 97.75% | 95.50% |
| HotpotQA | Research v2 | 59.50 | 76.87 | $0.027746 | 99.25% | 98.50% |
| MuSiQue-Ans | All context | 55.00 | 66.52 | $0.064869 | 100.00% | 100.00% |
| MuSiQue-Ans | Old Jev profile | 50.00 | 62.24 | $0.047798 | 91.79% | 80.00% |
| MuSiQue-Ans | Research v2 | 54.00 | 63.92 | $0.053823 | 96.50% | 91.00% |

HotpotQA, research v2 versus all context: F1 +1.80 points; 95% paired interval [-1.34, +4.78]. Estimated API cost 18.4% lower.

HotpotQA, research v2 versus old jev profile: F1 -0.70 points; 95% paired interval [-3.67, +2.29]. Estimated API cost 10.9% higher.

MuSiQue-Ans, research v2 versus all context: F1 -2.60 points; 95% paired interval [-7.36, +2.42]. Estimated API cost 17.0% lower.

MuSiQue-Ans, research v2 versus old jev profile: F1 +1.68 points; 95% paired interval [-2.54, +6.08]. Estimated API cost 12.6% higher.

400 fresh validation questions, 200 per dataset, disjoint from all 240 development questions. Supplied candidate passages; Jev 1.13.0 and GPT-5.6 Luna, low reasoning. At most one generation per branch; 1,200 evaluation branches. Costs include Jev and generation with reported cache discounts; common retrieval, hosting and networking excluded. These are subsets, not full benchmark or leaderboard results.

0 errors; 0 unknown costs; 0 unknown cache-usage records; 0 truncations.

Answer F1 measures normalized token overlap, so wording changes can affect scores. Supporting-passage retention is a separate diagnostic. Model training overlap is unknown. No query expansion, full-corpus retrieval, or modern neural reranker comparison was tested. MuSiQue uses the pinned community mirror described in the protocol.

[Protocol and reproduction](RESEARCH.md). Raw records: `artifacts/research-v2/`.
