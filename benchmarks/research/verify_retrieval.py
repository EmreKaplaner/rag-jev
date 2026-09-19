"""Independent trec_eval parity check; run in the pinned isolated uvx environment."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pytrec_eval

ROOT = Path("artifacts/retrieval-v1")


def main():
    qrels = json.loads((ROOT / "qrels.json").read_text())
    runs = json.loads((ROOT / "runs.json").read_text())
    expected = json.loads((ROOT / "per-query.json").read_text())
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.10", "recip_rank"})
    mapping = {"ndcg_cut_10": "ndcg10", "recall_10": "recall10", "recip_rank": "mrr10"}
    count, maximum_error = 0, 0.0
    assert len(qrels) == len(expected) == 300
    assert set(runs) == {"bm25", "jev", "ettin"}
    for arm, run in runs.items():
        assert set(run) == set(qrels)
        assert all(len(documents) == 10 for documents in run.values())
        observed = evaluator.evaluate(run)
        assert set(observed) == set(qrels)
        for qid, metrics in observed.items():
            for official, local in mapping.items():
                error = abs(metrics[official] - expected[qid][arm][local])
                maximum_error = max(maximum_error, error)
                assert error < 1e-12, (arm, qid, local, error)
                count += 1
    result = {
        "status": "passed",
        "implementation": "pytrec-eval-terrier",
        "version": version("pytrec-eval-terrier"),
        "metric_values_checked": count,
        "maximum_absolute_error": maximum_error,
        "note": "MRR is truncated at 10 because exported runs contain only the returned top10.",
        "sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "qrels.json",
                "runs.json",
                "per-query.json",
                "protocol.json",
                "report.json",
            ]
        },
    }
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
