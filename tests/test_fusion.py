import json
import subprocess
import sys
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import ValidationError

from rag_jev import ContextSelector, Document, Judgment, ProviderError, SelectRequest
from rag_jev.comparison import RankingRequest, compare_rankings, generate_rankings
from rag_jev.fusion import ranking_signals
from rag_jev.generation import Generator
from rag_jev.models import Policy
from rag_jev.selector import apply_policy
from rag_jev.tokens import count_tokens
from rag_jev.workbench import Price, ScoredRun, input_hash


def record(scores=(0.1, 0.9, 0.8)):
    request = SelectRequest(
        query="Which evidence?",
        documents=[
            Document(id=str(i), text=f"passage {i}", metadata={"page": i})
            for i in range(len(scores))
        ],
        mode="fusion",
        top_n=1,
    )
    return ScoredRun(
        request=request,
        relevant_ids=["1"],
        created_at=datetime.now(UTC),
        source="live",
        input_hash=input_hash(request),
        scoring_elapsed_ms=123,
        judgments=[Judgment(relevance=s, model="controlled", input_tokens=10) for s in scores],
    )


def test_fusion_preserves_a_useful_original_signal_and_explains_positions():
    # Jev demotes the original first hit, but both rankers agree on the second.
    r = record()
    result = apply_policy(r.request, r.judgments)
    assert result.selected_ids == ["1"]
    assert [d.jev_rank for d in result.decisions] == [3, 1, 2]
    assert [d.original_rank for d in result.decisions] == [1, 2, 3]
    assert result.decisions[1].fusion_score == pytest.approx(1 / 62 + 1 / 61)
    assert result.documents[0] is r.request.documents[1]
    # Opposite rankings tie at the ends. Original order is the tie break.
    ranks, scores = ranking_signals([0.1, 0.5, 0.9])
    assert ranks == [3, 2, 1] and scores[0] == scores[2]
    result = apply_policy(record([0.1, 0.5, 0.9]).request, record([0.1, 0.5, 0.9]).judgments)
    assert result.selected_ids == ["0"]
    assert ranking_signals([0.5, 0.5, 0.5])[0] == [1, 2, 3]
    assert ranking_signals([]) == ([], [])


def test_fusion_groups_pins_budget_shadow_and_policy_identity():
    r = record([0.1, 0.9, 0.8, 0.0])
    docs = list(r.request.documents)
    docs[1] = docs[1].model_copy(update={"group_id": "parent"})
    docs[2] = docs[2].model_copy(update={"group_id": "parent"})
    docs[3] = docs[3].model_copy(update={"pinned": True})
    req = SelectRequest(query="q", documents=docs, mode="fusion", top_n=2, shadow=True)
    result = apply_policy(req, r.judgments)
    assert result.selected_ids == ["1", "2", "3"]
    assert result.documents == docs and result.top_n_exceeded
    limited = apply_policy(req.model_copy(update={"max_context_tokens": 1}), r.judgments)
    assert limited.selected_ids == ["3"] and limited.context_token_budget_exceeded
    assert limited.decisions[1].reason == "beyond_token_budget"
    one = apply_policy(
        r.request.model_copy(update={"max_context_tokens": count_tokens(docs[0].text)}), r.judgments
    )
    assert len(one.selected_ids) == 1
    assert Policy(mode="fusion").version != Policy(mode="rerank").version
    assert Policy(mode="fusion", shadow=True).version == Policy(mode="fusion").version
    # Existing exported answers continue to validate against their old policy IDs.
    from rag_jev.workbench import ReplayBundle

    for name in ["improvement", "regression"]:
        from pathlib import Path

        ReplayBundle.model_validate_json(
            Path(f"src/rag_jev/static/replays/{name}.json").read_text()
        )


async def test_fusion_invalid_threshold_and_failure_passthrough():
    class Failing:
        calls = 0

        async def score(self, *args):
            self.calls += 1
            raise ProviderError("provider_error")

    provider = Failing()
    selector = ContextSelector(provider)
    with pytest.raises(ValidationError, match="without a relevance threshold"):
        await selector.select(query="q", documents=[], mode="fusion", min_relevance=0)
    assert provider.calls == 0
    r = record()
    result = await selector.select_request(r.request)
    assert result.documents == r.request.documents and result.status == "bypassed"
    assert all(d.jev_rank is None and d.fusion_score is None for d in result.decisions)
    result = await selector.select(query="q", documents=[], mode="fusion")
    assert result.selected_ids == []


async def test_three_way_equal_limits_no_rescoring_and_generation_reuse():
    body = RankingRequest(record=record(), top_n=1)
    price = Price(input_per_million=1, output_per_million=2)
    report = compare_rankings(body, selection_price=price)
    assert [a.selection.selected_ids for a in report.arms] == [["0"], ["1"], ["1"]]
    assert [a.evidence_recall for a in report.arms] == [0, 1, 1]
    assert [a.ndcg for a in report.arms] == [0, 1, 1]
    assert report.scoring_calls == report.generation_calls == 0
    assert report.arms[0].selector_cost_usd == 0
    assert report.arms[1].selector_cost_usd == pytest.approx(0.00003)
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "controlled",
                "choices": [{"message": {"content": "Answer [S1]"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await generate_rankings(
            body,
            Generator("http://test/v1", "controlled", client=client),
            generation_price=price,
            selection_price=price,
        )
    assert len(calls) == result.generation_calls == 2
    assert calls[0]["messages"][0] == calls[1]["messages"][0]
    assert result.arms[2].answer_reused_from == "jev"
    assert result.arms[1].total_cost_usd == pytest.approx(0.00015)
    assert result.arms[0].total_cost_usd == pytest.approx(0.00012)
    assert compare_rankings(body).arms[1].total_cost_usd is None


def test_three_way_remote_auth_export_and_cli(network_service, tmp_path):
    body = RankingRequest(record=record(), top_n=1).model_dump(mode="json")
    with httpx.Client(base_url=network_service) as client:
        assert client.post("/v1/rankings", json=body).status_code == 401
        client.headers["Authorization"] = "Bearer integration-test-token"
        response = client.post("/v1/rankings", json=body)
        assert response.status_code == 200
        assert len(response.json()["arms"]) == 3
        response = client.post("/v1/compare-rankings", json=body)
        assert response.status_code == 200 and response.json()["generation_calls"] == 2
    path = tmp_path / "record.json"
    path.write_text(record().model_dump_json())
    result = subprocess.run(
        [sys.executable, "-m", "rag_jev.cli", "compare-rankings", str(path), "--top-n", "1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["arms"][2]["selection"]["selected_ids"] == ["1"]
