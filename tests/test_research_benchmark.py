"""Guard the experiment's separation of evidence, labels, and shared costs."""

import json
from types import SimpleNamespace

import pytest

from benchmarks.public.common import digest, write_json
from benchmarks.public.research import case_from_row, compact_score, indices, policies_for


def test_musique_conversion_keeps_gold_out_of_documents():
    case = case_from_row(
        "musique",
        {
            "id": "sample",
            "question": "Where?",
            "answerable": True,
            "answer": "SECRET_GOLD",
            "answer_aliases": ["ALIAS"],
            "question_decomposition": [{"answer": "HIDDEN_BRIDGE"}],
            "paragraphs": [
                {"idx": 7, "title": "Title", "paragraph_text": "Evidence", "is_supporting": True}
            ],
        },
    )
    assert case["answers"] == ["SECRET_GOLD", "ALIAS"]
    assert case["relevant_ids"] == ["7"]
    assert case["documents"][0]["text"] == "Title\nEvidence"
    assert "SECRET_GOLD" not in json.dumps(case["documents"])
    assert "is_supporting" not in json.dumps(case["documents"])
    assert "HIDDEN_BRIDGE" not in json.dumps(case)


def test_selection_preserves_order_and_rerank_ties_without_labels():
    docs = [{"text": "A"}, {"text": "B"}, {"text": "C"}, {"text": "D"}]
    policy = {"name": "test", "threshold": 0.2}
    assert indices(docs, [0.2, 0.9, 0.9, 0.19], policy) == [0, 1, 2]
    assert indices(docs, [0.2, 0.9, 0.9, 0.19], {**policy, "order": "relevance"}) == [1, 2, 0]
    assert indices(docs, [], {"name": "all_context"}) == [0, 1, 2, 3]
    assert indices(docs, [0.1] * 4, policy) == []


async def test_compact_scoring_sends_only_evidence_and_counts_usage_once():
    class Client:
        async def system_one(self, **kwargs):
            assert kwargs["state"] == {"query": "Q", "passages": [{"text": "A"}, {"text": "B"}]}
            assert "passages[1].text" in kwargs["questions"]["1"].instructions
            return SimpleNamespace(
                nouls={"0": SimpleNamespace(noul=0.9), "1": SimpleNamespace(noul=0.2)},
                model="test",
                usage=SimpleNamespace(input_tokens=100, output_tokens=4),
            )

    result = await compact_score(
        {
            "query": "Q",
            "documents": [{"text": "A", "metadata": {"gold": True}}, {"text": "B"}],
            "answers": ["SECRET"],
            "relevant_ids": ["a"],
        },
        SimpleNamespace(model="test", client=Client()),
    )
    assert result["input_tokens"] == 100
    assert result["scores"] == [0.9, 0.2]


def test_evaluation_requires_matching_frozen_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.public.research.ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        policies_for("evaluation", {"v": 2})
    policy = {"name": "selected"}
    write_json(
        tmp_path / "frozen-policy.json", {"protocol_hash": digest({"v": 2}), "policy": policy}
    )
    assert policies_for("evaluation", {"v": 2})[-1] == policy
    with pytest.raises(ValueError, match="another protocol"):
        policies_for("evaluation", {"v": 3})
