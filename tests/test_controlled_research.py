import json

import httpx
import numpy as np
import pytest

from benchmarks.research.design import choose, isolation_keys
from benchmarks.research.runner import (
    capture_path,
    capture_request,
    capture_response,
    evidence_metrics,
    generation_cost,
)
from benchmarks.research.statistics import holm, paired, summarize
from rag_jev.generation import Answer


def test_bookend_is_a_permutation_of_the_same_thresholded_evidence():
    docs = [{"text": str(i)} for i in range(7)]
    scores = [0.9, 0.7, 0.8, 0.2, 0.3, 0.1, 0.6]
    base = {"kind": "jev", "threshold": 0.2}
    ranked = choose(docs, scores, {**base, "order": "rank"})
    bookend = choose(docs, scores, {**base, "order": "bookend"})
    assert bookend[0] == ranked[0]
    assert bookend[-1] == ranked[1]
    assert len(bookend) == len(set(bookend)) and set(bookend) == set(ranked)
    assert choose(docs, scores, {**base, "order": "original"}) == [0, 1, 2, 3, 4, 6]
    with pytest.raises(ValueError):
        choose(docs, scores, {"kind": "oracle"})
    with pytest.raises(ValueError):
        choose(docs, [1], base)


def test_isolation_catches_shared_single_hop_components_not_just_full_questions():
    a = {"question_decomposition": [{"id": 1}, {"id": 2}]}
    b = {"question_decomposition": [{"id": 2}, {"id": 3}]}
    assert isolation_keys("musique", a) & isolation_keys("musique", b) == {"musique-hop:2"}
    assert isolation_keys("hotpotqa", {"supporting_facts": {"title": ["The Title!"]}}) == {
        "hotpot-title:title"
    }


def test_statistics_pair_questions_and_control_the_primary_family():
    result = paired([0] * 20, [1] * 20, draws=1000, permutations=2000, confidence=0.975)
    assert result["delta"] == 1
    assert result["ci"] == [1, 1]
    assert result["n_questions"] == 20
    assert result["p_one_sided"] < 0.01
    null = paired([1] * 10, [1] * 10, draws=100, permutations=100)
    assert null["p_one_sided"] == 1
    assert holm({"a": 0.01, "b": 0.03, "c": 0.04}) == {"a": 0.03, "b": 0.06, "c": 0.06}
    with pytest.raises(ValueError):
        paired([1], [1])


def test_repeated_answers_are_averaged_within_questions_before_inference():
    def record(score):
        return {
            "f1": score,
            "em": score,
            "api_cost_usd": 1,
            "normalized_api_cost_usd": 2,
            "support_recall": 1,
            "support_precision": 1,
            "all_support": 1,
            "ndcg5": 1,
            "ndcg10": 1,
            "local_cpu_seconds": 0,
            "answer": {"status": "generated", "finish_reason": "stop", "text": str(score)},
            "unknown_usage": False,
            "stage_ms": 100,
        }

    summary, questions = summarize([[record(0), record(1)], [record(1), record(1)]])
    assert summary["questions"] == 2 and summary["branches"] == 4
    assert questions["f1"] == [0.5, 1]
    assert summary["f1"] == 0.75
    assert summary["within_question_f1_sd"] == 0.5
    assert summary["answer_disagreement_fraction"] == 0.5
    with pytest.raises(ValueError):
        summarize([[record(0)], [record(0), record(1)]])


def test_cost_normalization_does_not_confuse_cache_savings_with_filtering():
    rates = {"generator_input": 1, "generator_cached_input": 0.1, "generator_output": 2}
    answer = Answer(
        text="A", status="generated", input_tokens=1000, output_tokens=10, cached_input_tokens=900
    )
    actual, normalized, unknown = generation_cost(answer, rates)
    assert actual == pytest.approx(0.00021)
    assert normalized == pytest.approx(0.00102)
    assert not unknown
    assert generation_cost(Answer(text="Error", status="error"), rates)[2]


def test_ranking_metric_penalizes_missing_support():
    case = {"relevant_ids": ["a", "c"]}
    assert evidence_metrics(case, ["a", "c"])["ndcg5"] == 1
    assert evidence_metrics(case, ["a"])["support_recall"] == 0.5
    assert 0 < evidence_metrics(case, ["a"])["ndcg5"] < 1
    assert evidence_metrics(case, [])["support_precision"] == 0
    assert np.isfinite(evidence_metrics(case, [])["ndcg5"])


async def test_audit_capture_keeps_request_body_and_allowlisted_response_without_credentials(
    tmp_path,
):
    target = tmp_path / "answer.json"
    token = capture_path.set(target)
    try:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200,
                    json={
                        "id": "response-id",
                        "model": "model",
                        "usage": {"prompt_tokens": 3},
                        "choices": [],
                        "debug_auth": "dummy-secret",
                    },
                )
            ),
            event_hooks={"request": [capture_request], "response": [capture_response]},
        ) as client:
            await client.post(
                "https://example.test/chat/completions",
                json={"model": "model"},
                headers={"Authorization": "Bearer dummy-secret"},
            )
    finally:
        capture_path.reset(token)
    assert json.loads(target.with_suffix(".request.json").read_text()) == {"model": "model"}
    response = json.loads(target.with_suffix(".response.json").read_text())
    assert response["id"] == "response-id"
    assert "debug_auth" not in response
    assert all("dummy-secret" not in p.read_text() for p in tmp_path.glob("*.json"))
