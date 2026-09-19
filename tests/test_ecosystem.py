import pytest

from benchmarks.ecosystem.common import Ledger, sample, select_indices
from benchmarks.ecosystem.flashrag import JevFlashRAGRetriever, run_native
from rag_jev.models import Judgment
from rag_jev.selector import ContextSelector


def test_managed_flashrag_reuses_loop_under_semaphore_contention_and_closes_provider():
    import asyncio

    from benchmarks.ecosystem.managed_flashrag import ManagedJevFlashRAGRetriever

    class SlowScorer:
        closed = False
        loops = set()

        async def score(self, query, document, guidance):
            self.loops.add(asyncio.get_running_loop())
            await asyncio.sleep(0.001)
            return Judgment(relevance=1, model="fixture")

        async def aclose(self):
            assert asyncio.get_running_loop() in self.loops
            self.closed = True

    class ManyDocuments:
        def batch_search(self, queries):
            return [[{"id": str(i), "contents": "evidence"} for i in range(12)] for _ in queries]

    provider = SlowScorer()
    selector = ContextSelector(provider, max_concurrency=2, on_error="raise")
    with ManagedJevFlashRAGRetriever(
        ManyDocuments(), selector, close_provider=True, min_relevance=0.2
    ) as retriever:
        assert len(retriever.batch_search(["q"])[0]) == 12
        assert len(retriever.batch_search(["q2"])[0]) == 12
    assert provider.closed and len(provider.loops) == 1
    with pytest.raises(RuntimeError, match="closed"):
        retriever.batch_search(["q"])


def test_ledger_reserves_across_instances_and_never_retries_unknown(tmp_path):
    first, second = Ledger(tmp_path, 1), Ledger(tmp_path, 1)
    first.reserve("a", "fp", 0.8)
    with pytest.raises(ValueError, match="exhausted"):
        second.reserve("b", "fp", 0.3)
    with pytest.raises(ValueError, match="already attempted"):
        second.reserve("a", "fp", 0.1)
    first.finish("a", None, "error")
    assert second.summary()["reserved_unknown_usd"] == 0.8
    second.reserve("b", "fp", 0.1)
    second.finish("b", 0.02)
    assert second.summary()["remaining_after_reservations_usd"] == pytest.approx(0.18)


def test_reservation_overrun_stops_future_calls(tmp_path):
    ledger = Ledger(tmp_path, 25)
    ledger.reserve("a", "fp", 0.01)
    ledger.finish("a", 0.02)
    with pytest.raises(ValueError, match="reconcile"):
        ledger.reserve("b", "fp", 0.01)


def test_filter_after_reranker_uses_fresh_context_aligned_scores():
    assert select_indices("bge_jev_filter", 3, bge=[0.1, 0.9, 0.8], post=[0.1, 0.7], top_k=2) == [2]
    with pytest.raises(ValueError, match="Fresh scores"):
        select_indices("bge_jev_filter", 3, bge=[0.1, 0.9, 0.8], post=[0.1, 0.7, 1], top_k=2)
    assert select_indices("jev_filter", 3, jev=[0.1, 0.05, 0.15]) == []
    assert select_indices("jev_rerank", 3, jev=[0.1, 0.05, 0.15]) == [2, 0, 1]


def test_hash_sampling_is_order_independent():
    assert sample(["a", "b", "c"], 2) == sample(["c", "a", "b"], 2)


@pytest.mark.parametrize(
    "arm,mode", [("jev_rerank", "rerank"), ("jev_filter", "filter_and_rerank")]
)
def test_pilot_selection_matches_product_policy_including_ties(arm, mode):
    from rag_jev.models import Document, SelectRequest
    from rag_jev.selector import apply_policy

    scores = [0.2, 0.19, 0.8, 0.8, 0.0, 1.0]
    request = SelectRequest(
        query="q",
        documents=[Document(id=str(i), text="passage") for i in range(6)],
        mode=mode,
        min_relevance=None if mode == "rerank" else 0.2,
        top_n=3,
    )
    actual = apply_policy(request, [Judgment(relevance=v, model="fixture") for v in scores])
    expected = select_indices(arm, 6, jev=scores, top_k=3)
    assert actual.selected_ids == [str(i) for i in expected]


def test_graded_retrieval_and_empty_run():
    pytest.importorskip("pytrec_eval")
    from benchmarks.ecosystem.report import retrieval_metrics

    qrels = {"high": 2, "low": 1, "irrelevant": 0}
    assert retrieval_metrics(["high", "low"], qrels)["ndcg10"] == 1
    assert retrieval_metrics(["low", "high"], qrels)["ndcg10"] < 1
    assert retrieval_metrics([], qrels) == {"ndcg10": 0, "recall10": 0, "mrr10": 0}
    assert retrieval_metrics(["high"], qrels)["recall10"] == 0.5


def test_empty_extraction_bridge_never_accepts_empty_checker_or_incomplete_response():
    from benchmarks.ecosystem.ragchecker_resume import completed_empty_extraction

    record = {
        "http_status": 200,
        "cost_usd": 0.0001,
        "choice": {"finish_reason": "stop", "message": {"content": ""}},
    }
    assert completed_empty_extraction("Extract claims: unanswerable", record, "Extract claims:")
    assert not completed_empty_extraction("Check entailment", record, "Extract claims:")
    record["choice"]["finish_reason"] = "length"
    assert not completed_empty_extraction("Extract claims: q", record, "Extract claims:")
    record["choice"]["finish_reason"] = "stop"
    record["cost_usd"] = None
    assert not completed_empty_extraction("Extract claims: q", record, "Extract claims:")


def test_crag_numeric_reference_amendment_is_narrow_and_preserves_aliases():
    from benchmarks.ecosystem.report_resume import scalar_reference_metrics

    assert scalar_reference_metrics("0", ["invalid question", 0], "crag") == {"em": 1, "f1": 1}
    assert scalar_reference_metrics("invalid question", ["invalid question", 0], "crag") == {
        "em": 1,
        "f1": 1,
    }
    for invalid in [None, True, float("nan"), {}, []]:
        with pytest.raises(TypeError):
            scalar_reference_metrics("0", [invalid], "crag")
    with pytest.raises(TypeError):
        scalar_reference_metrics("0", [0], "hotpotqa")


class Retriever:
    def batch_search(self, queries):
        return [
            [
                {"id": "duplicate", "contents": "noise", "metadata": {"url": "a"}},
                {"id": "duplicate", "contents": "evidence", "metadata": {"url": "b"}},
            ]
            for _ in queries
        ]


class Scorer:
    async def score(self, query, document, guidance):
        return Judgment(relevance=float(document.text == "evidence"), model="fixture")


def test_flashrag_wrapper_preserves_metadata_and_handles_duplicate_source_ids():
    retriever = JevFlashRAGRetriever(Retriever(), ContextSelector(Scorer()), min_relevance=0.2)
    result = retriever.batch_search(["q"])
    assert result == [[{"id": "duplicate", "contents": "evidence", "metadata": {"url": "b"}}]]


@pytest.mark.asyncio
async def test_flashrag_sync_bridge_requires_async_entrypoint_in_loop():
    retriever = JevFlashRAGRetriever(Retriever(), ContextSelector(Scorer()), min_relevance=0.2)
    with pytest.raises(RuntimeError, match="abatch_search"):
        retriever.batch_search(["q"])
    assert len((await retriever.abatch_search(["q"]))[0]) == 1


def test_native_flashrag_pipeline_consumes_selected_context():
    pytest.importorskip("flashrag")

    class Generator:
        def generate(self, prompts):
            assert prompts == [
                {"query": "q", "documents": [{"id": "duplicate", "text": "evidence"}]}
            ]
            return ["fixture answer"]

    retriever = JevFlashRAGRetriever(Retriever(), ContextSelector(Scorer()), min_relevance=0.2)
    result = run_native([{"id": "case", "query": "q"}], retriever, Generator())
    assert result.pred == ["fixture answer"]
    assert len(result.retrieval_result[0]) == 1
