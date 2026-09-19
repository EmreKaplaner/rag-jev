# Exploratory BEIR/SciFact corpus retrieval

| Pipeline | NDCG@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- |
| bm25 | 66.47 | 78.49 | 63.28 |
| jev | 75.13 | 82.42 | 73.64 |
| ettin | 72.11 | 80.26 | 70.64 |

Exploratory retrieval-only result: all 300 BEIR/SciFact test queries and all 5,183 corpus documents. Custom BM25 retrieves 20 candidates; BM25, Jev and Ettin each return 10 from that identical pool. No gold passages are inserted. All table values are percentages.

First-stage Recall@20: 82.59%. Queries with missing relevant papers remain in every denominator.

jev vs bm25: NDCG@10 difference +8.66 points; descriptive 95% interval [+5.85, +11.59]. Bootstrap units: 247 connected groups of queries sharing relevant papers; the estimate remains query-weighted.

jev vs ettin: NDCG@10 difference +3.01 points; descriptive 95% interval [+0.61, +5.41]. Bootstrap units: 247 connected groups of queries sharing relevant papers; the estimate remains query-weighted.

Jev scoring API usage estimate: $0.13188 for 300 queries ($0.4396 per 1,000). Ettin inference: 780.1 process CPU seconds; its hosting cost is unknown. Common retrieval cost is excluded; these figures do not establish a total-cost advantage over Ettin.

All 2,700 per-query metric values match pytrec-eval-terrier 0.5.10 within 1e-12. The exported run contains only ten results per query, so reciprocal rank is truncated at ten.

This is one retrieval dataset with a custom BM25 first stage, not the full BEIR suite, an answer-accuracy result, or a state-of-the-art claim. The separate HotpotQA/MuSiQue answer study keeps its original primary endpoints.

Ettin's published training recipe selects checkpoints on NanoBEIR, including SciFact and HotpotQA; overlap with those selection examples has not been ruled out. Jev pretraining exposure is unknown. These are not demonstrably benchmark-naive models. No dense retriever or query expansion is evaluated.

Sources: [BEIR data](https://github.com/beir-cellar/beir), [Ettin training recipe](https://huggingface.co/blog/ettin-reranker), [trec_eval wrapper](https://github.com/cvangysel/pytrec_eval). [Methods and reproduction](README.md).
