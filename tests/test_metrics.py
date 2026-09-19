import math

from rag_jev.demo import CASES
from rag_jev.evaluation import measure


def test_ndcg_does_not_reward_dropping_other_relevant_evidence():
    case = CASES[0]
    only_direct_answer = [case.documents[1]]
    metrics = measure(case, only_direct_answer)
    assert metrics.precision == 1
    assert metrics.evidence_recall == 0.5
    assert metrics.ndcg == 1 / (1 + 1 / math.log2(3))
    # With a shared top-1 evaluation budget, that same answer is optimal.
    assert measure(case, only_direct_answer, cutoff=1).ndcg == 1


def test_unanswerable_query_is_reported_separately():
    metrics = measure(CASES[2], [])
    assert metrics.evidence_recall is None and metrics.ndcg is None
    assert metrics.correctly_empty is True
    assert metrics.all_evidence_lost is False
