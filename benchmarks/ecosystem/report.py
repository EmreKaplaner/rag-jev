"""Report every pilot case, graded retrieval metrics, paired uncertainty and spend."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.ecosystem.answer import selected
from benchmarks.ecosystem.common import ARMS, ROOT, Ledger, fingerprint_files, protocol
from benchmarks.ecosystem.retrieve import all_cases
from benchmarks.ecosystem.score import load_score
from benchmarks.public.common import answer_metrics, digest, read_json, write_json
from benchmarks.research.retrieval import cluster_interval, clusters


def retrieval_metrics(ids, qrels, k=10):
    import pytrec_eval

    evaluator = pytrec_eval.RelevanceEvaluator(
        {"q": qrels}, {f"ndcg_cut_{k}", f"recall_{k}", "recip_rank"}
    )
    run = {"q": {doc: float(len(ids) - i) for i, doc in enumerate(ids[:k])}}
    result = evaluator.evaluate(run).get("q", {})
    return {
        "ndcg10": result.get(f"ndcg_cut_{k}", 0.0),
        "recall10": result.get(f"recall_{k}", 0.0),
        "mrr10": result.get("recip_rank", 0.0),
    }


def stage_cost(case, arm, p):
    if arm in {"jev_rerank", "jev_filter"}:
        return load_score(case, "jev", p)["api_cost_usd"]
    if arm == "bge_jev_filter":
        return load_score(case, "jev_post_bge", p)["api_cost_usd"]
    return 0.0


def summarize(rows, metric_names):
    return {
        arm: {m: float(np.mean([r["arms"][arm][m] for r in rows])) for m in metric_names}
        for arm in ARMS
    }


def main():
    p = protocol()
    cases = all_cases()
    per_query = []
    for case in cases:
        arms = {}
        for arm in ARMS:
            docs = selected(case, arm, p)
            entry = {
                "selected_ids": [d["id"] for d in docs],
                "selected_documents": len(docs),
                "selector_api_usd": stage_cost(case, arm, p),
            }
            if case["dataset"] == "crag":
                result = read_json(ROOT / "answers" / f"{case['id']}-{arm}.json")
                entry.update(answer_metrics(result["prediction"], case["references"], "crag"))
                entry["answer_status"] = result["answer"]["status"]
                entry["finish_reason"] = result["answer"]["finish_reason"]
                for cost in ["api_cost_usd", "normalized_api_cost_usd"]:
                    entry[cost] = (
                        result[cost] + entry["selector_api_usd"]
                        if result[cost] is not None and entry["selector_api_usd"] is not None
                        else None
                    )
                entry["generation_elapsed_ms"] = result["answer"]["elapsed_ms"]
                entry["input_tokens"] = result["answer"]["input_tokens"]
            else:
                entry.update(retrieval_metrics(entry["selected_ids"], case["qrels"]))
            arms[arm] = entry
        per_query.append(
            {
                "id": case["id"],
                "dataset": case["dataset"],
                "retriever": case["retriever"],
                "split": case["split"],
                "arms": arms,
            }
        )
    groups = defaultdict(list)
    for row in per_query:
        groups[f"{row['dataset']}/{row['retriever']}/{row['split']}"].append(row)
    report = {
        "protocol_hash": digest(p),
        "budget": Ledger().summary(),
        "groups": {},
        "analysis_code": fingerprint_files([Path(__file__)]),
    }
    for name, rows in groups.items():
        dataset, retriever, split = name.split("/")
        source = [
            c
            for c in cases
            if c["dataset"] == dataset and c["retriever"] == retriever and c["split"] == split
        ]
        assert [c["id"] for c in source] == [r["id"] for r in rows]
        if dataset == "crag":
            metrics = [
                "em",
                "f1",
                "selected_documents",
                "api_cost_usd",
                "normalized_api_cost_usd",
                "generation_elapsed_ms",
                "input_tokens",
            ]
            units = [[i] for i in range(len(rows))]
            primary = "f1"
        else:
            metrics = ["ndcg10", "recall10", "mrr10", "selected_documents", "selector_api_usd"]
            units = clusters(
                [
                    {"relevant_ids": [d for d, value in c["qrels"].items() if value > 0]}
                    for c in source
                ]
            )
            primary = "ndcg10"
        # Unknown cost/usage stays unknown, never silently zero or excluded-case mean.
        usable = [
            m for m in metrics if all(r["arms"][a][m] is not None for r in rows for a in ARMS)
        ]
        item = {
            "queries": len(rows),
            "summary": summarize(rows, usable),
            "unknown_metrics": sorted(set(metrics) - set(usable)),
            "comparisons_descriptive": {},
        }
        for comparator in ["baseline", "bge"]:
            for arm in ["jev_rerank", "jev_filter", "bge_jev_filter"]:
                item["comparisons_descriptive"][f"{arm}_vs_{comparator}"] = cluster_interval(
                    [r["arms"][comparator][primary] for r in rows],
                    [r["arms"][arm][primary] for r in rows],
                    units,
                )
        if dataset != "crag":
            item["candidate_recall20"] = float(
                np.mean(
                    [
                        len(
                            {d["id"] for d in c["documents"]}
                            & {d for d, v in c["qrels"].items() if v > 0}
                        )
                        / sum(v > 0 for v in c["qrels"].values())
                        for c in source
                    ]
                )
            )
            item["corpus_documents"] = p["datasets"][dataset]["documents"]
        else:
            item["generation_errors"] = sum(
                r["arms"][a]["answer_status"] == "error" for r in rows for a in ARMS
            )
            item["output_limit_branches"] = sum(
                r["arms"][a]["finish_reason"] == "length" for r in rows for a in ARMS
            )
        report["groups"][name] = item
    diagnostics = {}
    for dataset in ["fiqa", "nfcorpus", "scifact"]:
        diagnostics[dataset] = read_json(ROOT / "retrieval" / f"{dataset}-dense.json")[
            "diagnostics"
        ]
    bge_rows = [load_score(c, "bge", p) for c in cases]
    report["local_compute"] = {
        "bge_cpu_seconds": sum(r["cpu_seconds"] for r in bge_rows),
        "bge_elapsed_seconds": sum(r["elapsed_ms"] for r in bge_rows) / 1000,
        "bge_truncated_pairs": sum(r["truncated_pairs"] for r in bge_rows),
        "dense_truncation": diagnostics,
        "monetary_cost_usd": None,
        "note": (
            "MPS accelerator wall time plus host CPU seconds; not an invoice. Retrieval "
            "shared across arms, local compute is not free."
        ),
    }
    report["limitations"] = [
        (
            "Exploratory small query subsets, single generation replicate; descriptive "
            "intervals, no superiority claim."
        ),
        "Complete BEIR corpora, but shallow20-candidate retrieval and custom BM25 baseline.",
        (
            "SciFact is previously exposed replication; unknown model pretraining "
            "contamination on all datasets."
        ),
        (
            "CRAG uses official supplied pages, not independent web search. Short-answer "
            "EM/F1 are NOT official CRAG semantic judge scores."
        ),
        (
            "FlashRAG executes the actual SequentialPipeline with frozen paid selector "
            "traces. BERGEN check is a rerank-stage replication, not the entire BERGEN "
            "pipeline."
        ),
        (
            "Canonical full-Wikipedia NQ/TriviaQA/HotpotQA and independent human "
            "adjudication remain pending."
        ),
    ]
    write_json(ROOT / "per-query.json", per_query)
    write_json(ROOT / "report.json", report)
    write_json("benchmarks/ecosystem/results.json", report)
    # Blind arm labels and order; no claimed human judgments until someone completes it.
    packet, key = [], []
    for case in [c for c in cases if c["dataset"] == "crag" and c["split"] == "evaluation"]:
        answers = []
        for index, arm in enumerate(sorted(ARMS, key=lambda a: digest(["blind", case["id"], a]))):
            label = chr(65 + index)
            raw = read_json(ROOT / "answers" / f"{case['id']}-{arm}.json")
            answers.append(
                {
                    "label": label,
                    "answer": raw["prediction"],
                    "evidence": raw["prompt"]["documents"],
                    "correct": None,
                    "supported": None,
                    "notes": "",
                }
            )
            key.append({"id": case["id"], "label": label, "arm": arm})
        packet.append(
            {
                "id": case["id"],
                "query": case["query"],
                "references": case["references"],
                "answers": answers,
            }
        )
    write_json(ROOT / "human-review-blinded.json", packet)
    write_json(ROOT / "human-review-key.private.json", key)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
