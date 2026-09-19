"""Replay fixed RRF against published ecosystem candidates, without inference.

--export reads the original local caches and writes a portable, text-free input.
The default command reproduces the report and chart using that committed input.
"""

import argparse
import gzip
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean

from benchmarks.public.common import digest, read_json, write_json
from benchmarks.research.retrieval import cluster_interval, clusters
from rag_jev.fusion import RRF_K, ranking_signals
from rag_jev.tokens import count_tokens

ROOT = Path("benchmarks/fusion")
ORIGINAL = Path("artifacts/ecosystem-v1")
ARMS = ["original", "bge", "jev", "fusion"]
SOURCE_COMMIT = "3cb06c7a8d04ebe54b126137573febfc0c87dad6"


def export():
    protocol = read_json(ORIGINAL / "protocol.json")
    # Validate the historical inference implementation against its immutable revision,
    # not against new product code. Never rewrite the old campaign's hashes.
    for path, expected in protocol["code"].items():
        raw = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{path}"])
        assert hashlib.sha256(raw).hexdigest() == expected, path
    rows = []
    for dataset in ["fiqa", "nfcorpus", "scifact"]:
        for method in ["bm25", "dense"]:
            data = read_json(ORIGINAL / "retrieval" / f"{dataset}-{method}.json")
            assert data["protocol_hash"] == digest(protocol)
            for case in data["cases"]:
                if case["split"] != "evaluation":
                    continue
                scores = {}
                for stage in ["jev", "bge"]:
                    path = ORIGINAL / "scores" / f"{case['id']}-{method}-{stage}.json"
                    score = read_json(path)
                    assert score["fingerprint"] == digest(
                        {
                            "query": case["query"],
                            "documents": case["documents"],
                            "stage": stage,
                            "protocol": digest(protocol),
                        }
                    )
                    assert score["status"] == "ok"
                    assert len(score["scores"]) == len(case["documents"])
                    if stage == "jev":
                        assert score["document_ids"] == [d["id"] for d in case["documents"]]
                    scores[stage] = score
                rows.append(
                    {
                        "id": case["id"],
                        "dataset": dataset,
                        "retriever": method,
                        "document_ids": [d["id"] for d in case["documents"]],
                        "passage_tokens": [count_tokens(d["text"]) for d in case["documents"]],
                        "qrels": case["qrels"],
                        "scores": scores,
                    }
                )
    payload = {
        "source_commit": SOURCE_COMMIT,
        "source_protocol_sha256": digest(protocol),
        "rows": rows,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    (ROOT / "inputs.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    write_json(
        ROOT / "protocol.json",
        {
            "name": "fusion-v1",
            "source_commit": SOURCE_COMMIT,
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "arms": ARMS,
            "rrf_k": 60,
            "weights": [1, 1],
            "top_n": 10,
            "ties": "Jev ties and fused ties preserve original input order.",
            "scope": (
                "Post-hoc replay of all 600 evaluation query/retriever cases from "
                "ecosystem-v1. Fixed k=60, equal weights; no weight or cutoff "
                "search. No new model calls. Existing test outcomes previously "
                "examined: exploratory replication, not a new holdout."
            ),
            "statistics": (
                "10,000 paired bootstrap resamples of connected query clusters "
                "sharing positive qrels. Descriptive 95% intervals; no "
                "multiplicity correction or SOTA claim."
            ),
            "metrics": (
                "Graded linear-gain nDCG@10, positive-qrel recall@10, passage "
                "token estimates. Full qrels define ideal ranking, including "
                "documents absent from candidates. All queries included."
            ),
            "cost": (
                "Recorded Jev API cost per replayed deployment, zero new benchmark"
                " spend. Original and local BGE API costs are zero; compute is "
                "unknown, not free. Retrieval/generation excluded. Latency is "
                "historical stage timing, not fresh deployment latency."
            ),
        },
    )


def metrics(ids, qrels):
    gains = [max(0, qrels.get(d, 0)) for d in ids[:10]]
    ideal = sorted([v for v in qrels.values() if v > 0], reverse=True)[:10]

    def dcg(values):
        return sum(v / math.log2(i + 2) for i, v in enumerate(values))

    gold = {d for d, v in qrels.items() if v > 0}
    return {
        "ndcg10": dcg(gains) / dcg(ideal) if ideal else 0,
        "recall10": len(set(ids[:10]) & gold) / len(gold) if gold else 0,
    }


def evaluate(payload):
    rows = []
    seen = set()
    for case in payload["rows"]:
        key = (case["dataset"], case["retriever"], case["id"])
        assert key not in seen
        seen.add(key)
        n = len(case["document_ids"])
        assert n == len(set(case["document_ids"])) == len(case["passage_tokens"])
        assert all(len(case["scores"][s]["scores"]) == n for s in ["bge", "jev"])
        _, fused = ranking_signals(case["scores"]["jev"]["scores"])
        orders = {
            "original": list(range(n)),
            "bge": sorted(range(n), key=lambda i: -case["scores"]["bge"]["scores"][i]),
            "jev": sorted(range(n), key=lambda i: -case["scores"]["jev"]["scores"][i]),
            "fusion": sorted(range(n), key=lambda i: -fused[i]),
        }
        arms = {}
        for arm, order in orders.items():
            chosen = order[:10]
            ids = [case["document_ids"][i] for i in chosen]
            arms[arm] = {
                **metrics(ids, case["qrels"]),
                "selected_ids": ids,
                "passage_tokens": sum(case["passage_tokens"][i] for i in chosen),
                "selector_api_usd": case["scores"]["jev"]["api_cost_usd"]
                if arm in {"jev", "fusion"}
                else 0,
                "recorded_selector_ms": case["scores"]["jev" if arm == "fusion" else arm][
                    "elapsed_ms"
                ]
                if arm != "original"
                else 0,
            }
        rows.append(
            {
                "id": case["id"],
                "dataset": case["dataset"],
                "retriever": case["retriever"],
                "arms": arms,
            }
        )
    groups = defaultdict(list)
    for row, case in zip(rows, payload["rows"], strict=True):
        groups[f"{row['dataset']}/{row['retriever']}"].append((row, case))
    result = {}
    for group, pairs in groups.items():
        units = clusters(
            [{"relevant_ids": [d for d, v in c["qrels"].items() if v > 0]} for _, c in pairs]
        )
        result[group] = {
            "queries": len(pairs),
            "clusters": len(units),
            "largest_cluster": max(map(len, units)),
            "summary": {
                a: {
                    m: mean(r["arms"][a][m] for r, _ in pairs)
                    for m in [
                        "ndcg10",
                        "recall10",
                        "passage_tokens",
                        "selector_api_usd",
                        "recorded_selector_ms",
                    ]
                }
                for a in ARMS
            },
            "fusion_minus": {
                a: cluster_interval(
                    [r["arms"][a]["ndcg10"] for r, _ in pairs],
                    [r["arms"]["fusion"]["ndcg10"] for r, _ in pairs],
                    units,
                )
                for a in ARMS
                if a != "fusion"
            },
        }
    return rows, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    if args.export:
        export()
    raw = gzip.decompress((ROOT / "inputs.json.gz").read_bytes())
    protocol = read_json(ROOT / "protocol.json")
    assert hashlib.sha256(raw).hexdigest() == protocol["input_sha256"]
    assert protocol["rrf_k"] == RRF_K == 60
    rows, groups = evaluate(json.loads(raw))
    report = {
        "protocol": protocol,
        "groups": groups,
        "new_api_spend_usd": 0,
        "code_sha256": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in ["benchmarks/fusion/evaluate.py", "src/rag_jev/fusion.py"]
        },
    }
    write_json(ROOT / "results.json", report)
    write_json(ROOT / "per-query.json", rows)
    lines = [
        "# Fusion replay: same candidates, an additional ranking signal",
        "",
        protocol["scope"],
        "",
        "| Dataset / retrieval | Original | BGE reranker | Jev | Fusion | Fusion − Jev (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, group in groups.items():
        values = [f"{group['summary'][a]['ndcg10'] * 100:.2f}" for a in ARMS]
        ci = group["fusion_minus"]["jev"]
        lines.append(
            f"| {name} | "
            + " | ".join(values)
            + f" | {ci['delta'] * 100:+.2f} "
            + f"[{ci['ci95'][0] * 100:+.2f}, {ci['ci95'][1] * 100:+.2f}] |"
        )
    lines += [
        "",
        "Values are graded nDCG@10 × 100, not answer accuracy.",
        "",
        "![All six settings](../../src/rag_jev/static/fusion-results.svg)",
        "",
        "## Interpretation and limits",
        "",
        (
            "Fusion preserves the original retriever signal. Whether it helps "
            "depends on that signal: averaging in a weaker ranking can reduce "
            "Jev's gains. Read all settings, not just the best case."
        ),
        "",
        (
            "These are the same 100 test queries per corpus under two "
            "retrievers, not 600 independent questions. NFCorpus has only 10 "
            "connected query clusters (largest 91/100); SciFact and all pilot "
            "outcomes were previously exposed. The bootstrap intervals are "
            "descriptive and do not establish SOTA."
        ),
        "",
        (
            "Full original qrels, including missing candidates, determine the "
            "denominator. No zero-hit queries were dropped. Original means "
            "custom BM25 or E5-small-v2 dense order; BGE is "
            "bge-reranker-v2-m3, not the BGE-M3 embedding model used in the "
            "independent catalog study. Jev is the original contextual Noul "
            "scorer, not that study's four-level Score rubric. This is not an "
            "exact replication of either external study."
        ),
        "",
        (
            "No new inference or generated-answer benchmark was run here. API "
            "stage costs and latency are copied from the historical traces; "
            "RRF adds local arithmetic, whose production latency is not "
            "benchmarked. No claim of lower total RAG cost or better answers "
            "follows from this replay. See results.json for each arm's token, "
            "recall, cost and timing diagnostics."
        ),
        "",
        "## Reproduce",
        "",
        "```sh",
        "uv run --group benchmark python -m benchmarks.fusion.evaluate",
        "```",
        "",
        (
            "The committed compressed input includes candidate IDs, human "
            "dataset qrels, scores, token counts and recorded usage. It omits "
            "passage text. SHA-256 verification precedes scoring. `--export` "
            "is for the original local ecosystem caches; it validates their "
            "fingerprints and historical code against the pinned source "
            "commit. Existing studies and frozen hashes are unchanged."
        ),
        "",
        "[Product integration and comparison guide](../../docs/FUSION.md)",
        "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    plot(groups)
    print("\n".join(lines[:13]))


def plot(groups):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"svg.hashsalt": "rag-jev-fusion", "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained", facecolor="#f6f3ec")
    colors = ["#87998e", "#253b36", "#b84f2d", "#c58a35"]
    for ax, (name, g) in zip(axes.flat, groups.items(), strict=True):
        ax.set_facecolor("#f6f3ec")
        values = [g["summary"][a]["ndcg10"] * 100 for a in ARMS]
        bars = ax.bar(np.arange(4), values, color=colors)
        ax.set_xticks(np.arange(4), ["Original", "BGE", "Jev", "Fusion"])
        ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_title(name + f" · {g['queries']} queries")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("nDCG@10 × 100")
    fig.suptitle(
        (
            "Fixed RRF replay / all six settings\nPreviously exposed pilot "
            "samples · retrieval quality, not answer accuracy"
        ),
        fontsize=14,
    )
    path = Path("src/rag_jev/static/fusion-results.svg")
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
