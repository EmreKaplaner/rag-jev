import json

import httpx
import pytest

from rag_jev.generation import Generator
from rag_jev.models import Document


@pytest.mark.parametrize(
    "base,expected",
    [("https://api.openai.com/v1", "openai-only"), ("https://other.example/v1", "")],
)
async def test_openai_key_fallback_is_scoped_to_official_endpoint(monkeypatch, base, expected):
    monkeypatch.setenv("RAG_JEV_GENERATION_BASE_URL", base)
    monkeypatch.setenv("RAG_JEV_GENERATION_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("RAG_JEV_GENERATION_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-only")
    generator = Generator.from_env()
    assert generator.api_key == expected
    await generator.aclose()
    monkeypatch.setenv("RAG_JEV_GENERATION_API_KEY", "explicit")
    generator = Generator.from_env()
    assert generator.api_key == "explicit"
    await generator.aclose()


async def test_quota_error_has_actionable_safe_message():
    response = httpx.Response(
        429, json={"error": {"type": "insufficient_quota", "message": "PRIVATE upstream body"}}
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        result = await Generator("http://test/v1", "test", client=client).generate(
            "q", [Document(id="a", text="evidence")]
        )
    assert result.error_code == "generation_insufficient_quota"
    assert "billing" in result.text
    assert "PRIVATE" not in result.model_dump_json()


async def test_reasoning_budget_failure_preserves_paid_usage_and_settings():
    def respond(request):
        body = json.loads(request.content)
        assert body["reasoning_effort"] == "low" and body["max_completion_tokens"] == 2000
        return httpx.Response(
            200,
            json={
                "model": "gpt-5.6-luna",
                "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 2000},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await Generator(
            "http://test/v1",
            "gpt-5.6-luna",
            client=client,
            reasoning_effort="low",
            max_output_tokens=2000,
        ).generate("q", [Document(id="a", text="evidence")])
    assert result.status == "error" and result.error_code == "generation_output_limit"
    assert result.input_tokens == 80 and result.output_tokens == 2000
    assert result.settings["reasoning_effort"] == "low"


async def test_malformed_finish_reason_returns_error_without_crashing():
    response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "Answer"}, "finish_reason": {"bad": True}}]},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        result = await Generator("http://test/v1", "test", client=client).generate(
            "q", [Document(id="a", text="evidence")]
        )
    assert result.status == "error" and result.error_code == "generation_invalid_response"
