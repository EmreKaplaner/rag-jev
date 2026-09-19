import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rag_jev import ContextSelector, Document, Judgment, SelectRequest
from rag_jev.demo import CASES, FixtureScorer
from rag_jev.generation import Generator
from rag_jev.models import Policy
from rag_jev.selector import apply_policy
from rag_jev.server import create_app
from rag_jev.workbench import Price, ReplayRequest, StudyRequest, compare_run, replay_run, score_run


def test_group_pin_and_target_keep_whole_evidence_sets():
    docs = [
        Document(id="a", text="direct", group_id="parent"),
        Document(id="b", text="support", group_id="parent"),
        Document(id="c", text="essential", pinned=True),
        Document(id="d", text="noise"),
    ]
    judgments = [Judgment(relevance=p, model="test") for p in [0.9, 0.1, 0.01, 0.02]]
    request = SelectRequest(query="q", documents=docs, min_relevance=0.5, top_n=2)
    result = apply_policy(request, judgments)
    assert result.selected_ids == ["a", "b", "c"]
    assert result.top_n_exceeded
    assert [x.reason for x in result.decisions] == [
        "retained",
        "group_retained",
        "pinned",
        "below_threshold",
    ]
    shadow = apply_policy(request.model_copy(update={"shadow": True}), judgments)
    assert shadow.documents == docs and shadow.selected_ids == result.selected_ids
    assert shadow.outcome == "context_selected"


async def test_standalone_query_reaches_scorer_not_labels():
    seen = []

    class Capture:
        async def score(self, query, document, guidance):
            seen.append((query, guidance))
            return Judgment(relevance=0.1, model="test")

    result = await ContextSelector(Capture()).select(
        query="What about that?",
        retrieval_query="What is the refund deadline?",
        documents=[Document(id="a", text="passage")],
        min_relevance=0.5,
    )
    assert seen == [("What is the refund deadline?", None)]
    assert result.outcome == "no_context_selected"


async def test_replay_reuses_scores_validates_inputs_and_measures_supplied_baseline():
    class Counting(FixtureScorer):
        calls = 0

        async def score(self, *args):
            self.calls += 1
            return await super().score(*args)

    scorer = Counting()
    case = CASES[0]
    study = StudyRequest(
        request=SelectRequest(query=case.query, documents=case.documents, min_relevance=0.2),
        relevant_ids=case.relevant_ids,
        baseline_ids=["refund"],
    )
    record = await score_run(study, ContextSelector(scorer))
    view = replay_run(ReplayRequest(record=record, policy=Policy(min_relevance=0.9)))
    assert scorer.calls == 3
    assert view.baseline_ids == ["refund"]
    assert view.evidence_recall == 0.5 and view.relevant_dropped == ["receipt"]
    assert view.baseline_evidence_recall == 0.5
    assert view.baseline_context_tokens == view.selected_context_tokens
    payload = record.model_dump(mode="json")
    payload["request"]["documents"][0]["text"] = "tampered"
    with pytest.raises(ValidationError, match="inputs changed"):
        ReplayRequest(record=payload, policy=Policy(min_relevance=0.2))
    with pytest.raises(ValidationError, match="guidance"):
        ReplayRequest(record=record, policy=Policy(min_relevance=0.2, relevance_guidance="new"))


async def test_generation_same_prompt_citations_costs_and_no_context():
    requests = []

    def responder(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": "Answer [S1] [S99]"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        gen = Generator("http://test/v1", "test-model", "private", client=client)
        case = CASES[0]
        record = await score_run(
            StudyRequest(
                request=SelectRequest(query=case.query, documents=case.documents, min_relevance=0.9)
            ),
            ContextSelector(FixtureScorer()),
        )
        view = replay_run(ReplayRequest(record=record, policy=record.request.policy))
        result = await compare_run(
            view,
            gen,
            generation_price=Price(input_per_million=1, output_per_million=2),
            selection_price=Price(input_per_million=3, output_per_million=4),
        )
        assert len(requests) == 2
        assert requests[0]["messages"][0] == requests[1]["messages"][0]
        inputs = [json.loads(x["messages"][1]["content"]) for x in requests]
        assert inputs[0]["question"] == inputs[1]["question"]
        assert len(inputs[0]["evidence"]) == 3 and len(inputs[1]["evidence"]) == 1
        assert result.selected.citations == ["refund"]
        assert result.selected.unknown_citations == ["S99"]
        assert result.baseline_cost_usd == pytest.approx(0.00014)
        empty = await gen.generate(case.query, [])
        assert empty.status == "no_context" and len(requests) == 2


@pytest.mark.parametrize(
    "response",
    [httpx.Response(401, text="private secret"), httpx.Response(200, json={"choices": []})],
)
async def test_generation_failures_do_not_echo_upstream(response):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        answer = await Generator("http://test/v1", "test", client=client).generate(
            "q", CASES[0].documents
        )
    assert answer.status == "error" and answer.input_tokens is None
    assert "private" not in answer.model_dump_json()


def test_workbench_auth_import_and_static_security():
    with TestClient(create_app(ContextSelector(FixtureScorer()), api_token="secret")) as client:
        assert client.get("/").status_code == 200
        assert "frame-ancestors 'none'" in client.get("/").headers["content-security-policy"]
        assert client.get("/v1/fixtures/refund").status_code == 401
        h = {"Authorization": "Bearer secret"}
        view = client.get("/v1/fixtures/refund", headers=h).json()
        assert view["record"]["source"] == "fixture"
        bundle = {"record": view["record"], "policy": view["policy"], "comparison": None}
        assert client.post("/v1/import", headers=h, json=bundle).status_code == 200
        assert (
            client.post(
                "/v1/replay",
                headers={**h, "Origin": "https://untrusted.example"},
                json={"record": view["record"], "policy": view["policy"]},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/v1/compare", headers=h, json={"record": view["record"], "policy": view["policy"]}
            ).status_code
            == 503
        )
        bundle["record"]["baseline_ids"] = ["unknown"]
        assert client.post("/v1/import", headers=h, json=bundle).status_code == 422
