"""Opt-in real Jev smoke test: RAG_JEV_LIVE=1 uv run pytest tests/test_live.py -q."""

import os

import pytest
from dotenv import load_dotenv

from rag_jev import ContextSelector, Jev
from rag_jev.demo import CASES


@pytest.mark.live
async def test_live_jev_selection():
    if os.getenv("RAG_JEV_LIVE") != "1":
        pytest.skip("set RAG_JEV_LIVE=1 to enable paid live API calls")
    load_dotenv()
    assert os.getenv("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY is required for live verification"
    async with Jev() as provider:
        result = await ContextSelector(provider, timeout_ms=15000, on_error="raise").select(
            query=CASES[0].query,
            documents=CASES[0].documents,
            min_relevance=0.2,
        )
    assert result.status == "applied"
    assert result.usage.completed_documents == 3
    assert result.usage.input_tokens > 0
    assert all(d.relevance is not None for d in result.decisions)
    assert "refund" in result.selected_ids, "Jev removed the explicit answer passage"
