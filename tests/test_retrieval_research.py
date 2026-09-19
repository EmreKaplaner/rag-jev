import pytest

from benchmarks.public.common import bm25_order
from benchmarks.research.retrieval import BM25, cluster_interval, clusters, metrics


def test_indexed_bm25_preserves_the_declared_formula_and_ties():
    docs = [
        {"text": "alpha alpha beta"},
        {"text": "beta gamma"},
        {"text": "alpha"},
        {"text": "unrelated"},
        {"text": "unrelated"},
    ]
    index = BM25(docs)
    for query in ["alpha beta", "BETA", "missing", "alpha alpha gamma"]:
        order, _ = index.rank(query)
        assert order == bm25_order(query, docs)


def test_missing_retrieved_gold_stays_in_metric_denominator():
    result = metrics(["a", "distractor"], ["a", "missing"])
    assert result["recall10"] == 0.5
    assert 0 < result["ndcg10"] < 1
    assert result["mrr10"] == 1
    assert metrics(["none"], ["a"])["ndcg10"] == 0


def test_paper_clusters_are_transitive_and_keep_query_weighted_estimate():
    cases = [
        {"relevant_ids": ["a"]},
        {"relevant_ids": ["a", "b"]},
        {"relevant_ids": ["b"]},
        {"relevant_ids": ["c"]},
    ]
    groups = clusters(cases)
    assert sorted(groups) == [[0, 1, 2], [3]]
    result = cluster_interval([0, 0, 0, 0], [1, 1, 1, 0], groups)
    assert result["delta"] == pytest.approx(0.75)
    assert result["query_clusters"] == 2
    assert result["ci95"] == [0, 1]
