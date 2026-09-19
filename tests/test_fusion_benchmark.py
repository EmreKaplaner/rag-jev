import gzip
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.fusion.evaluate import evaluate, metrics


def test_fusion_portable_replay_preserves_all_historical_baseline_scores():
    root = Path("benchmarks/fusion")
    raw = gzip.decompress((root / "inputs.json.gz").read_bytes())
    protocol = json.loads((root / "protocol.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == protocol["input_sha256"]
    rows, groups = evaluate(json.loads(raw))
    historical = {
        (r["id"], r["retriever"]): r
        for r in json.loads(Path("benchmarks/ecosystem/per-query.json").read_text())
    }
    assert len(rows) == 600 and len(groups) == 6
    for row in rows:
        old = historical[row["id"], row["retriever"]]
        for new, previous in [("original", "baseline"), ("jev", "jev_rerank"), ("bge", "bge")]:
            assert row["arms"][new]["ndcg10"] == pytest.approx(old["arms"][previous]["ndcg10"])
            assert row["arms"][new]["selected_ids"] == old["arms"][previous]["selected_ids"]
    saved = json.loads((root / "results.json").read_text())
    assert groups == saved["groups"]
    assert rows == json.loads((root / "per-query.json").read_text())


def test_graded_ndcg_uses_missing_gold_and_keeps_zero_hit_cases():
    result = metrics(["weak"], {"strong": 3, "weak": 1})
    assert 0 < result["ndcg10"] < 1 / 3
    assert result["recall10"] == 0.5
    assert metrics(["noise"], {"gold": 1}) == {"ndcg10": 0, "recall10": 0}
