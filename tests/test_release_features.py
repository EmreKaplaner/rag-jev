import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_jev.calibration import candidate_policies, choose, held_out, validate_splits
from rag_jev.evaluation import EvalCase, Trace, collect_trace, evaluate, fingerprint
from rag_jev.models import Document, Judgment, Policy, SelectRequest, Usage
from rag_jev.provider import BatchJudgments
from rag_jev.review import summarize_review
from rag_jev.selector import ContextSelector, apply_policy
from rag_jev.server import create_app
from rag_jev.tokens import count_tokens
from rag_jev.workbench import ReplayBundle


def test_budget_reserves_pins_skips_whole_groups_and_can_fill_remaining_space():
    docs = [
        Document(id="large", text=" word" * 30, group_id="large-group"),
        Document(id="peer", text="peer", group_id="large-group"),
        Document(id="small", text="small"),
        Document(id="pin", text="pin", pinned=True),
    ]
    request = SelectRequest(query="q", documents=docs, mode="rerank", max_context_tokens=3)
    judgments = [Judgment(relevance=s, model="test") for s in [0.99, 0.1, 0.8, 0.01]]
    result = apply_policy(request, judgments)
    assert result.selected_ids == ["small", "pin"]
    assert result.selected_context_tokens == 2
    assert result.decisions[0].reason == result.decisions[1].reason == "beyond_token_budget"
    assert not result.context_token_budget_exceeded
    pinned_group = [d.model_copy(update={"pinned": True}) if d.id == "peer" else d for d in docs]
    result = apply_policy(
        request.model_copy(update={"documents": pinned_group, "shadow": True}), judgments
    )
    assert result.context_token_budget_exceeded
    assert set(result.selected_ids) == {"large", "peer", "pin"}
    assert result.documents == pinned_group
    assert result.selected_context_tokens == sum(
        count_tokens(d.text) for d in pinned_group if d.id != "small"
    )


def test_existing_recordings_keep_policy_ids_and_import_after_budget_addition():
    for file in Path("src/rag_jev/static/replays").glob("*.json"):
        if file.name != "catalog.json":
            bundle = ReplayBundle.model_validate_json(file.read_text())
            assert bundle.policy.max_context_tokens is None
            assert bundle.policy.version == bundle.comparison.policy_version
    assert Policy(mode="rerank", max_context_tokens=1).version != Policy(mode="rerank").version


def test_replay_only_never_constructs_upstreams_even_with_environment_keys(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay mode constructed an upstream provider")

    monkeypatch.setattr("rag_jev.server.Jev", forbidden)
    monkeypatch.setattr("rag_jev.server.Generator.from_env", forbidden)
    monkeypatch.setenv("TYPESAFE_API_KEY", "configured-but-must-not-be-used")
    with TestClient(create_app(replay_only=True, api_token="")) as client:
        assert client.get("/v1/config").json()["replay_only"]
        catalog = client.get("/v1/recordings").json()
        assert {r["id"] for r in catalog} == {"improvement", "regression"}
        for entry in catalog:
            bundle = client.get(entry["path"]).json()
            assert client.post("/v1/import", json=bundle).status_code == 200
            body = {"record": bundle["record"], "policy": bundle["policy"]}
            assert client.post("/v1/replay", json=body).status_code == 200
            assert client.post("/v1/compare", json=body).status_code == 503
            assert client.post("/v1/select", json=bundle["record"]["request"]).status_code == 503
            assert (
                client.post("/v1/runs", json={"request": bundle["record"]["request"]}).status_code
                == 503
            )


class BatchFixture:
    async def score(self, *args):
        raise AssertionError("Contextual evaluation switched strategy")

    async def score_context(self, query, documents, guidance):
        assert query.startswith("Standalone")
        return BatchJudgments(
            judgments=[
                Judgment(relevance=0.9 if d.id == "a" else 0.1, model="fixture") for d in documents
            ],
            usage=Usage(input_tokens=777, output_tokens=12, completed_documents=len(documents)),
        )


def cases(prefix):
    return [
        EvalCase(
            id=f"{prefix}{i}",
            query=f"{prefix} question {i}",
            retrieval_query=f"Standalone {i}",
            documents=[Document(id="a", text="answer"), Document(id="b", text="noise noise")],
            relevant_ids=["a"],
            baseline_ids=["b"],
        )
        for i in range(3)
    ]


async def test_contextual_eval_and_calibration_preserve_shared_usage_and_holdout_choice(tmp_path):
    dev, hold = cases("development"), cases("evaluation")
    validate_splits(dev, hold)
    selector = ContextSelector(BatchFixture())
    traces = [await collect_trace(c, selector, scoring_strategy="contextual") for c in dev + hold]
    result = await evaluate(dev, traces[:3], min_relevance=0.2)
    assert result["usage"]["input_tokens"] == 3 * 777
    assert result["summary"]["baseline"]["evidence_recall"] == 0
    assert result["summary"]["filter"]["evidence_recall"] == 1
    policies = candidate_policies([0.2, 0.95], [1, 2], None)
    frozen = choose(dev, traces[:3], policies, "contextual", 0.98)
    before = json.dumps(frozen, sort_keys=True)
    assessment = held_out(hold, traces[3:], policies, "contextual", frozen)
    assert json.dumps(frozen, sort_keys=True) == before
    assert assessment["frozen_policy_held_out"]["summary"]["evidence_recall"] == 1
    assert not choose(dev, traces[:3], candidate_policies([0.99], [], None), "contextual", 0.98)[
        "eligible"
    ]
    for name, rows in [("dev", dev), ("hold", hold), ("traces", traces)]:
        (tmp_path / (name + ".json")).write_text(
            json.dumps([r.model_dump(mode="json") for r in rows])
        )
    command = [
        sys.executable,
        "-m",
        "rag_jev.cli",
        "calibrate",
        str(tmp_path / "dev.json"),
        str(tmp_path / "hold.json"),
        "--replay",
        str(tmp_path / "traces.json"),
        "--output",
        str(tmp_path / "report.json"),
    ]
    process = await asyncio.to_thread(subprocess.run, command, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert json.loads((tmp_path / "report.policy.json").read_text())["shadow"]
    assert (await asyncio.to_thread(subprocess.run, command, capture_output=True)).returncode != 0


def test_calibration_split_and_trace_validation():
    dev, hold = cases("development"), cases("evaluation")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_splits(dev, dev)
    with pytest.raises(ValueError, match="families"):
        validate_splits(
            [c.model_copy(update={"family_id": "shared"}) for c in dev],
            [c.model_copy(update={"family_id": "shared"}) for c in hold],
        )
    traces = [
        Trace(
            case_id=c.id,
            input_hash=fingerprint(c),
            judgments=[Judgment(relevance=0.9, model="fixture")] * 2,
            elapsed_ms=0,
        )
        for c in dev
    ]
    with pytest.raises(ValueError, match="strategy"):
        choose(dev, traces, candidate_policies([0.2], [], None), "contextual", 0.9)


def test_human_review_does_not_invent_completion(tmp_path):
    import csv

    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps({"cases": [{"case_id": "q", "answers": {"A": {}, "B": {}}}]}))
    labels = tmp_path / "labels.csv"
    fields = [
        "case_id",
        "anonymous_answer",
        "reviewer",
        "semantically_correct",
        "ambiguous_question_or_reference",
        "notes",
    ]

    def write(rows):
        with labels.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows(rows)

    write([["q", "A", "", "", "", ""], ["q", "B", "", "", "", ""]])
    assert summarize_review(str(packet), str(labels))["labeled_answers"] == 0
    write(
        [
            ["q", "A", "test-reviewer", "yes", "no", ""],
            ["q", "B", "test-reviewer", "unclear", "yes", ""],
        ]
    )
    summary = summarize_review(str(packet), str(labels))
    assert summary["status"] == "complete" and summary["correctness_counts"] == {
        "yes": 1,
        "unclear": 1,
    }
    write(
        [["q", "A", "test-reviewer", "yes", "no", ""], ["q", "A", "test-reviewer", "no", "no", ""]]
    )
    with pytest.raises(ValueError, match="duplicate"):
        summarize_review(str(packet), str(labels))


def test_family_bootstrap_does_not_count_related_questions_as_independent():
    from rag_jev.calibration import paired_interval

    result = paired_interval([10, 20, 30], [1, 2, 3], ["same"] * 3)
    assert result["ci95"] is None and result["independent_units"] == 1
    result = paired_interval([10, 20, 30], [1, 2, 3], ["a", "a", "b"])
    assert result["independent_units"] == 2 and result["questions"] == 3
