"""Verify BERGEN's real rerank stage against paid, contextual Jev score artifacts.

This is a stage replication, not a claim to have run BERGEN's Wikipedia RAG suite.
Upstream BERGEN stays separately cloned under its CC BY-NC-SA license.
"""

import sys

from benchmarks.ecosystem.common import ROOT, protocol
from benchmarks.ecosystem.retrieve import all_cases
from benchmarks.ecosystem.score import load_score
from benchmarks.public.common import digest, write_json


class ContextScoreAdapter:
    """Hydra-instantiated BERGEN model, with context-preserving precomputed API scores.

    BERGEN hardcodes .model.to('cuda'). The Identity has no parameters or tensors;
    moving it allocates no CUDA memory. Actual inference happened at the provider.
    It is not a local CUDA model and no GPU speed/cost claim is made.
    """

    def __init__(self, model_name="jev-contextual-cached", scores=None):
        import torch

        self.model_name = model_name
        self.model = torch.nn.Identity()
        self.scores = dict(scores or {})

    def collate_fn(self, batch):
        return {
            "q_id": [r["q_id"] for r in batch],
            "d_id": [r["d_id"] for r in batch],
            "keys": [r["score_key"] for r in batch],
        }

    def __call__(self, kwargs):
        import torch

        return {
            "score": torch.tensor([self.scores[k] for k in kwargs["keys"]], dtype=torch.float64)
        }


def run():
    p = protocol()
    sys.path.insert(0, str((ROOT / "upstream/bergen").resolve()))
    from modules.rerank import Rerank

    scores, rows, expected = {}, [], {}
    for case in all_cases():
        result = load_score(case, "jev", p)
        if result["status"] != "ok":
            raise ValueError("Cannot replicate failed scores")
        qid = case["id"] + "-" + case["retriever"]
        values = result["scores"]
        expected[qid] = [
            case["documents"][i]["id"]
            for i in sorted(range(len(values)), key=lambda i: (-values[i], i))
        ]
        for i, doc in enumerate(case["documents"]):
            key = digest([qid, doc["id"]])
            scores[key] = values[i]
            rows.append({"q_id": qid, "d_id": doc["id"], "score_key": key})
    stage = Rerank(
        init_args={"_target_": "benchmarks.ecosystem.bergen.ContextScoreAdapter", "scores": scores},
        batch_size=64,
    )
    output = stage.eval(rows)
    actual = dict(zip(output["q_id"], output["doc_id"], strict=True))
    assert actual == expected
    write_json(
        ROOT / "bergen-verification.json",
        {
            "upstream_sha": p["upstream"]["bergen"],
            "protocol_hash": digest(p),
            "queries": len(expected),
            "pairs": len(rows),
            "all_rankings_match": True,
            "scope": (
                "Actual native BERGEN Rerank.eval with context-preserving paid Jev scores. "
                "No new API inference; not full BERGEN generation pipeline."
            ),
        },
    )
    print("BERGEN native rerank parity", len(expected), "queries", flush=True)


if __name__ == "__main__":
    run()
