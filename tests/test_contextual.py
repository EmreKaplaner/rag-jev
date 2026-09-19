import asyncio
import json

import httpx2
import pytest
from pydantic import ValidationError
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from benchmarks.public.run import CONTEXTUAL_PROMPT
from rag_jev.models import Document, Policy, SelectRequest
from rag_jev.provider import CONTEXTUAL_INSTRUCTIONS, Jev, ProviderError
from rag_jev.selector import ContextSelector
from rag_jev.workbench import ReplayRequest, ScoredRun, StudyRequest, replay_run, score_run


async def test_contextual_official_sdk_shared_usage_replay_and_original_documents():
    calls = []

    async def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx2.Response(
            200,
            json={
                "model": "jev-test",
                "usage": {"input_tokens": 999, "output_tokens": 44},
                "answers": {"0": {"type": "noul", "noul": 0.9}, "1": {"type": "noul", "noul": 0.1}},
            },
        )

    assert CONTEXTUAL_PROMPT == CONTEXTUAL_INSTRUCTIONS
    docs = [
        Document(id="a", text="First", metadata={"secret": "private"}),
        Document(id="b", text="Second"),
    ]
    async with AsyncTypeSafeClient(
        api_key="fake", transport=httpx2.MockTransport(handler)
    ) as client:
        selector = ContextSelector(Jev(client=client))
        request = SelectRequest(
            query="Question", documents=docs, min_relevance=0.35, scoring_strategy="contextual"
        )
        record = await score_run(StudyRequest(request=request, relevant_ids=["a"]), selector)
        view = replay_run(ReplayRequest(record=record, policy=request.policy))
        assert len(calls) == 1
        assert "private" not in json.dumps(calls)
        assert calls[0]["state"] == {
            "query": "Question",
            "passages": [{"text": "First"}, {"text": "Second"}],
        }
        assert view.selection.documents == [docs[0]]
        assert view.selection.usage.input_tokens == 999
        assert view.selection.usage.output_tokens == 44
        assert all(d.input_tokens is None for d in view.selection.decisions)
        assert view.selection.prompt_version == "contextual-evidence-v1"
        changed = replay_run(ReplayRequest(record=record, policy=Policy(min_relevance=0.05)))
        assert changed.selection.documents == docs
        assert changed.selection.usage.input_tokens == 999 and len(calls) == 1
        tampered = record.model_dump()
        tampered["request"]["scoring_strategy"] = "independent"
        with pytest.raises(ValidationError, match="inputs changed"):
            ScoredRun.model_validate(tampered)
        tampered = record.model_dump()
        tampered["scoring_usage"] = None
        with pytest.raises(ValidationError, match="shared usage"):
            ScoredRun.model_validate(tampered)


async def test_contextual_empty_limit_and_deadline_do_not_fall_back_silently():
    calls = 0
    cancelled = False

    async def slow(request):
        nonlocal calls, cancelled
        calls += 1
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled = True
            raise

    async with AsyncTypeSafeClient(
        api_key="fake", transport=httpx2.MockTransport(slow), retry=RetryPolicy(max_retries=0)
    ) as client:
        selector = ContextSelector(Jev(client=client), timeout_ms=20)
        empty = await selector.select(
            query="q", documents=[], scoring_strategy="contextual", min_relevance=0.2
        )
        assert empty.usage.input_tokens == 0 and calls == 0
        too_many = await selector.select(
            query="q",
            scoring_strategy="contextual",
            min_relevance=0.2,
            documents=[Document(id=str(i), text="x") for i in range(33)],
        )
        assert too_many.status == "bypassed"
        assert too_many.error_code == "contextual_candidate_limit" and calls == 0
        timeout = await selector.select(
            query="q",
            documents=[Document(id="a", text="x")],
            scoring_strategy="contextual",
            min_relevance=0.2,
        )
        assert timeout.error_code == "deadline_exceeded" and cancelled
        assert not timeout.usage.complete and calls == 1


async def test_contextual_provider_failure_and_unsupported_scorer():
    class IndependentOnly:
        async def score(self, *args):
            raise AssertionError("Must not switch strategies silently")

    with pytest.raises(ProviderError, match="unsupported"):
        await ContextSelector(IndependentOnly(), on_error="raise").select(
            query="q",
            documents=[Document(id="a", text="x")],
            min_relevance=0.2,
            scoring_strategy="contextual",
        )
