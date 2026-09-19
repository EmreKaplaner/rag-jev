import json
from types import SimpleNamespace

import httpx
import pytest

from benchmarks.public.common import answer_metrics, bm25_order, bootstrap_delta
from benchmarks.public.run import score_case
from rag_jev.generation import Generator
from rag_jev.models import Document


def test_benchmark_metrics_aliases_normalization_and_yes_no_rules():
    assert answer_metrics("The Eiffel Tower!", ["Eiffel Tower"], "hotpotqa") == {"em": 1, "f1": 1}
    assert answer_metrics("York", ["New York"], "hotpotqa")["f1"] == pytest.approx(2 / 3)
    assert answer_metrics("yes indeed", ["yes"], "hotpotqa")["f1"] == 0
    assert answer_metrics("NYC", ["New York", "NYC"], "musique")["em"] == 1
    assert answer_metrics("unanswerable", ["2012"], "musique")["f1"] == 0
    assert answer_metrics("", [""], "hotpotqa")["f1"] == 0
    assert answer_metrics("", [""], "musique")["f1"] == 1


def test_bm25_and_paired_intervals():
    docs = [{"text": "music jazz"}, {"text": "refund return receipt"}, {"text": "music"}]
    assert bm25_order("refund", docs)[0] == 1
    interval = bootstrap_delta([0, 0, 0], [1, 1, 1], samples=100)
    assert interval == {"delta": 1, "ci95": [1, 1]}
    with pytest.raises(ValueError):
        bootstrap_delta([1], [])


async def test_contextual_scoring_does_not_receive_gold_or_metadata():
    class Client:
        async def system_one(self, **kwargs):
            state = kwargs["state"]
            assert state == {"query": "Q", "passages": [{"text": "A"}, {"text": "B"}]}
            encoded = json.dumps(state)
            assert "SECRET_GOLD" not in encoded
            assert "passages[1].text" in kwargs["questions"]["1"].instructions
            return SimpleNamespace(
                nouls={"0": SimpleNamespace(noul=0.9), "1": SimpleNamespace(noul=0.1)},
                model="test",
                usage=SimpleNamespace(input_tokens=123, output_tokens=10),
            )

    case = {
        "query": "Q",
        "documents": [
            {"id": "a", "text": "A", "metadata": {"hidden": True}},
            {"id": "b", "text": "B"},
        ],
        "answers": ["SECRET_GOLD"],
        "relevant_ids": ["a"],
    }
    result = await score_case(
        case, "contextual", SimpleNamespace(client=Client(), model="test"), None
    )
    assert result["scores"] == [0.9, 0.1]
    assert result["input_tokens"] == 123  # Batch usage counted once, not once per document.


async def test_generator_custom_prompt_and_cached_usage():
    def respond(request):
        body = json.loads(request.content)
        assert body["messages"][0]["content"] == "Only short answers"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        answer = await Generator(
            "http://test/v1", "test", client=client, system_prompt="Only short answers"
        ).generate("capital?", [Document(id="a", text="France: Paris")])
    assert answer.cached_input_tokens == 80
    assert answer.input_tokens == 100
