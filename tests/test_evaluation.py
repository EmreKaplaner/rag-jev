import json
import subprocess
import sys

import pytest

from rag_jev import ContextSelector
from rag_jev.demo import CASES, FixtureScorer, fixture_traces
from rag_jev.evaluation import collect_trace, evaluate


async def test_eval_reuses_scores_and_reports_evidence_loss():
    report = await evaluate(CASES, fixture_traces(), min_relevance=0.2)
    assert report["summary"]["filter"]["evidence_recall"] == 1
    assert report["summary"]["filter"]["characters"] < report["summary"]["baseline"]["characters"]
    assert report["summary"]["filter"]["correctly_empty"] == 1
    assert report["answer_quality"] == "not_evaluated"
    aggressive = await evaluate(CASES, fixture_traces(), min_relevance=0.999)
    assert aggressive["summary"]["filter"]["evidence_recall"] == 0
    assert aggressive["summary"]["filter"]["all_evidence_lost"] > 0


async def test_optional_answer_callback_runs_for_every_policy():
    calls = []

    async def generate_and_grade(case, documents):
        calls.append(case.id)
        # A deterministic extractive answerer exercises the hook, not model quality.
        text = " ".join(d.text for d in documents)
        answer = "30 days" if "refund within 30 days" in text else "insufficient evidence"
        return answer == case.reference_answer

    report = await evaluate(
        CASES[:1], fixture_traces()[:1], min_relevance=0.2, answer_evaluator=generate_and_grade
    )
    assert len(calls) == 4
    assert report["summary"]["filter"]["answer_correct"] == 1
    assert report["answer_quality"] == "measured_by_callback"


async def test_replay_rejects_stale_inputs_and_prompt_version():
    for trace in [
        fixture_traces()[0].model_copy(update={"input_hash": "wrong"}),
        fixture_traces()[0].model_copy(update={"prompt_version": "old"}),
    ]:
        with pytest.raises(ValueError, match="stale"):
            await evaluate(CASES[:1], [trace], min_relevance=0.2)


async def test_collect_trace_roundtrip():
    trace = await collect_trace(CASES[0], ContextSelector(FixtureScorer()))
    report = await evaluate(CASES[:1], [trace], min_relevance=0.2, top_n=1)
    assert report["summary"]["baseline"]["precision"] == 0
    assert report["summary"]["filter"]["precision"] == 1


def test_cli_offline_demo_and_replay():
    demo = subprocess.run(
        [sys.executable, "-m", "rag_jev.cli", "demo"], capture_output=True, text=True, check=True
    )
    assert "OFFLINE FIXTURE" in json.loads(demo.stdout)["notice"]
    replay = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_jev.cli",
            "eval",
            "examples/eval-cases.json",
            "--replay",
            "examples/fixture-traces.json",
            "--min-relevance",
            "0.2",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(replay.stdout)["cases"] == 3
