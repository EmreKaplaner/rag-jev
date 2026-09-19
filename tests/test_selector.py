import asyncio

import pytest
from pydantic import ValidationError

from rag_jev import ContextSelector, Document, Judgment, ProviderError, SelectRequest
from rag_jev.selector import apply_policy

DOCS = [Document(id=str(i), text=f"Source passage {i}", metadata={"page": i}) for i in range(4)]


class Scorer:
    def __init__(self, values=(0.1, 0.8, 0.9, 0.8)):
        self.values = values
        self.active = self.peak = self.calls = self.canceled = 0

    async def score(self, query, document, guidance):
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.002)
            return Judgment(
                relevance=self.values[int(document.id)], model="test-v1", input_tokens=10
            )
        except asyncio.CancelledError:
            self.canceled += 1
            raise
        finally:
            self.active -= 1


@pytest.mark.parametrize(
    "mode,threshold,expected",
    [
        ("filter", 0.2, ["1", "2", "3"]),
        ("rerank", None, ["2", "1", "3", "0"]),
        ("filter_and_rerank", 0.8, ["2", "1", "3"]),
    ],
)
async def test_selection_modes_preserve_content(mode, threshold, expected):
    result = await ContextSelector(Scorer()).select(
        query="question", documents=DOCS, mode=mode, min_relevance=threshold
    )
    assert [d.id for d in result.documents] == expected
    assert result.selected_ids == expected
    assert result.status == "applied"
    assert all(d == DOCS[int(d.id)] for d in result.documents)
    assert result.usage.input_tokens == 40
    assert result.models == ["test-v1"]


async def test_cap_and_threshold_reasons_are_distinct():
    result = await ContextSelector(Scorer()).select(
        query="q", documents=DOCS, mode="filter_and_rerank", min_relevance=0.2, top_n=2
    )
    assert result.selected_ids == ["2", "1"]
    assert [d.reason for d in result.decisions] == [
        "below_threshold",
        "retained",
        "retained",
        "beyond_top_n",
    ]


async def test_shadow_returns_original_and_reports_counterfactual():
    result = await ContextSelector(Scorer()).select(
        query="q", documents=DOCS, min_relevance=0.2, top_n=1, shadow=True
    )
    assert result.documents == DOCS
    assert result.selected_ids == ["1"]
    assert result.status == "shadow"
    assert all(d.returned for d in result.decisions)
    assert sum(d.selected for d in result.decisions) == 1
    assert result.returned_characters == result.input_characters > result.selected_characters


async def test_successful_empty_selection_is_not_a_failure():
    result = await ContextSelector(Scorer()).select(query="q", documents=DOCS, min_relevance=0.99)
    assert result.documents == []
    assert result.selected_ids == []
    assert result.status == "applied"


async def test_empty_input_does_not_call_provider():
    scorer = Scorer()
    result = await ContextSelector(scorer).select(query="q", documents=[], min_relevance=0.2)
    assert not result.documents and not scorer.calls
    assert result.usage.completed_documents == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"min_relevance": None},
        {"min_relevance": float("nan")},
        {"min_relevance": 1.1},
        {"mode": "rerank"},
        {"top_n": 0},
        {"top_n": True},
        {"query": " "},
        {"documents": [DOCS[0], DOCS[0]]},
        {"unknown": True},
    ],
)
async def test_invalid_input_never_calls_provider(changes):
    scorer = Scorer()
    args = {"query": "q", "documents": DOCS, "min_relevance": 0.2, **changes}
    with pytest.raises(ValidationError):
        await ContextSelector(scorer).select(**args)
    assert scorer.calls == 0


async def test_failure_abandons_partial_selection_and_ignores_cap():
    class Failing:
        async def score(self, query, document, guidance):
            if document.id == "0":
                return Judgment(relevance=0.0, model="test", input_tokens=7)
            await asyncio.sleep(0.01)
            raise ProviderError("provider_rate_limit")

    result = await ContextSelector(Failing()).select(
        query="q", documents=DOCS, min_relevance=0.9, top_n=1
    )
    assert result.documents == DOCS
    assert result.status == "bypassed"
    assert result.selected_ids is None
    assert all(d.selected is None and d.returned for d in result.decisions)
    assert result.usage.input_tokens == 7
    assert not result.usage.complete
    assert result.error_code == "provider_rate_limit"


async def test_raise_policy_exposes_only_error_code():
    class Failing:
        async def score(self, *args):
            raise ProviderError("provider_auth")

    with pytest.raises(ProviderError, match="provider_auth"):
        await ContextSelector(Failing(), on_error="raise").select(
            query="q", documents=DOCS, min_relevance=0.2
        )


async def test_total_deadline_includes_queue_and_cancels_work():
    class Slow(Scorer):
        async def score(self, *args):
            self.active += 1
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.canceled += 1
                raise
            finally:
                self.active -= 1

    scorer = Slow()
    result = await ContextSelector(scorer, timeout_ms=20, max_concurrency=1).select(
        query="q", documents=DOCS, min_relevance=0.2
    )
    assert result.error_code == "deadline_exceeded"
    assert result.elapsed_ms < 500
    assert scorer.active == 0 and scorer.canceled == 1


async def test_concurrency_limit_applies_across_requests():
    scorer = Scorer()
    selector = ContextSelector(scorer, max_concurrency=2)
    await asyncio.gather(
        *[selector.select(query="q", documents=DOCS, min_relevance=0.2) for _ in range(4)]
    )
    assert scorer.calls == 16 and scorer.peak == 2 and scorer.active == 0


async def test_caller_cancellation_propagates_and_cleans_up():
    entered = asyncio.Event()
    finished = asyncio.Event()

    class Slow:
        async def score(self, *args):
            entered.set()
            try:
                await asyncio.sleep(10)
            finally:
                finished.set()

    task = asyncio.create_task(
        ContextSelector(Slow()).select(query="q", documents=DOCS[:1], min_relevance=0.2)
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


def test_policy_fingerprint_changes_only_when_selection_changes():
    req = SelectRequest(query="q", documents=DOCS, min_relevance=0.2)
    assert req.policy.version == req.model_copy(update={"shadow": True}).policy.version
    assert req.policy.version != req.model_copy(update={"top_n": 1}).policy.version
    assert (
        req.policy.version != req.model_copy(update={"relevance_guidance": "domain"}).policy.version
    )


def test_replay_requires_all_judgments():
    with pytest.raises(ValueError, match="one judgment"):
        apply_policy(SelectRequest(query="q", documents=DOCS, min_relevance=0.2), [])
