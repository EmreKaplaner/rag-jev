"""Frozen design, label-free selection, and standalone sample reconstruction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.public.common import QA_SYSTEM, digest, normalize, read_json, write_json
from benchmarks.public.research import case_from_row
from rag_jev.provider import CONTEXTUAL_INSTRUCTIONS, CRITERIA

ROOT = Path("artifacts/research-v3")
SEED = "rag-jev-controlled-v3-2026-09-18"
SCHEMA = "rag-jev-research-v3"
ALL = {"name": "all_context", "kind": "all"}
FACTORIAL = [
    {"name": f"jev_{threshold}_{order}", "kind": "jev", "threshold": threshold, "order": order}
    for threshold in [0.2, 0.35]
    for order in ["original", "rank"]
]
CANDIDATES = FACTORIAL + [
    {"name": f"jev_{threshold}_bookend", "kind": "jev", "threshold": threshold, "order": "bookend"}
    for threshold in [0.2, 0.35]
]
NEURAL = [{"name": f"ettin_top{k}", "kind": "ettin", "top_k": k} for k in [3, 5, 8]]
ORACLE = {"name": "oracle_support_only", "kind": "oracle"}
DEV_ARMS = [ALL, *CANDIDATES, *NEURAL, ORACLE]


def choose(documents, scores, policy):
    """No labels/answer/dataset input; same whole passages for every real pipeline."""
    if policy["kind"] == "all":
        return list(range(len(documents)))
    if len(scores) != len(documents):
        raise ValueError("Expected exactly one score per passage")
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    if policy["kind"] == "ettin":
        return ranked[: policy["top_k"]]
    if policy["kind"] != "jev":
        raise ValueError("Oracle labels are never allowed in production selection")
    kept = [i for i in ranked if scores[i] >= policy["threshold"]]
    if policy["order"] == "original":
        return sorted(kept)
    if policy["order"] == "rank":
        return kept
    if policy["order"] == "bookend":
        # First, third, ... at the front; ..., fourth, second at the end.
        return kept[::2] + kept[1::2][::-1]
    raise ValueError("Unknown ordering")


def raw_rows():
    import pyarrow.parquet as pq

    source = Path("artifacts/public-benchmark/data")
    return {
        "hotpotqa": pq.read_table(source / "hotpotqa.parquet").to_pylist(),
        "musique": [json.loads(s) for s in (source / "musique.jsonl").read_text().splitlines()],
    }


def isolation_keys(dataset, row):
    if dataset == "musique":
        return {"musique-hop:" + str(p["id"]) for p in row["question_decomposition"]}
    return {"hotpot-title:" + normalize(t) for t in row["supporting_facts"]["title"]}


def source_manifest():
    files = [
        Path("benchmarks/research") / n
        for n in ["design.py", "runner.py", "neural.py", "statistics.py"]
    ] + [
        Path("benchmarks/public/common.py"),
        Path("benchmarks/public/research.py"),
        Path("src/rag_jev/provider.py"),
        Path("src/rag_jev/generation.py"),
        Path("src/rag_jev/models.py"),
        Path("uv.lock"),
        Path("benchmarks/research/prior-exposure.json"),
    ]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def prepare():
    if (ROOT / "protocol.json").exists():
        raise ValueError("Protocol already exists; never overwrite an experiment")
    previous = read_json("benchmarks/research/prior-exposure.json")
    old_protocol = read_json("benchmarks/public/protocol.json")
    for source in old_protocol["sources"]:
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Source checksum mismatch")
    rows = raw_rows()
    old_questions, blocked_keys = set(), set()
    for dataset, source in rows.items():
        exposed = set(previous[dataset])
        for row in source:
            if row["id"] in exposed:
                old_questions.add(normalize(row["question"]))
                blocked_keys.update(isolation_keys(dataset, row))
    datasets, isolation = {}, {}
    for dataset, source in rows.items():
        exposed = set(previous[dataset])
        development_rows = sorted(
            (r for r in source if r["id"] in exposed),
            key=lambda r: digest([SEED, "development", dataset, r["id"]]),
        )[:48]
        evaluation_rows = []
        for row in sorted(source, key=lambda r: digest([SEED, "evaluation", dataset, r["id"]])):
            question, keys = normalize(row["question"]), isolation_keys(dataset, row)
            if question in old_questions or keys & blocked_keys:
                continue
            evaluation_rows.append(row)
            old_questions.add(question)
            blocked_keys.update(keys)
            if len(evaluation_rows) == 200:
                break
        if len(evaluation_rows) != 200:
            raise ValueError(f"Not enough isolated {dataset} cases; amend design before scoring")
        datasets[dataset] = {}
        for split, selected in [("development", development_rows), ("evaluation", evaluation_rows)]:
            cases = []
            for row in selected:
                case = case_from_row(dataset, row)
                case["stratum"] = row["type"] if dataset == "hotpotqa" else row["id"].split("__")[0]
                cases.append(case)
                isolation[dataset + "/" + row["id"]] = sorted(isolation_keys(dataset, row))
            datasets[dataset][split] = cases
    protocol = {
        "schema": SCHEMA,
        "prepared_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "case_hash": digest(datasets),
        "sources": old_protocol["sources"],
        "author_source_verification": read_json("benchmarks/research/source-verification.json"),
        "prior_exposure_hash": digest(previous),
        "isolation_hash": digest(isolation),
        "sample": {"development_per_dataset": 48, "evaluation_per_dataset": 200},
        "split_rule": "Exclude all 640 previously observed questions. MuSiQue: no shared "
        "single-hop "
        "IDs with prior cases or another evaluation case. Hotpot: no shared supporting titles "
        "with prior cases or another evaluation case. Also normalized-question deduplication.",
        "scope": "Isolated public validation subsets, supplied candidates; not full-"
        "corpus retrieval. "
        "Strict isolation changes the sampled population. Model pretraining contamination unknown.",
        "scoring": {
            "model": "jev-1.13.0",
            "instructions": CONTEXTUAL_INSTRUCTIONS,
            "criteria": CRITERIA,
            "repetitions": 1,
            "retries": 0,
        },
        "neural": {
            **read_json(ROOT / "reranker-source.json"),
            "device": "cpu",
            "dtype": "float32",
            "threads": 4,
            "batch_size": 8,
            "max_length": 7999,
            "attention": "eager",
            "truncate": False,
            "warmup_excluded": True,
        },
        "generation": {
            "model": "gpt-5.6-luna",
            "reasoning_effort": "low",
            "max_completion_tokens": 2000,
            "system_prompt": QA_SYSTEM,
            "temperature": "provider default (not sent)",
            "seed": "not sent; independent provider samples",
            "retries": 0,
        },
        "repetitions": {"development": 2, "evaluation": 3},
        "concurrency": 12,
        "development_arms": DEV_ARMS,
        "evaluation_arms": "All context + complete 2x2 cutoff/order factorial + frozen winner "
        "if distinct + development-selected Ettin top-k. Oracle is development-only, ineligible.",
        "selection_rule": "Highest macro mean answer F1 over two fresh development replicates "
        "among "
        "Jev candidates cheaper than all-context on each dataset using "
        "uncached normalized API cost. "
        "Ties: lower cost, then name. Ettin: highest macro F1 among k=3,5,8, ties smaller k. "
        "No qualifying Jev candidate means no promotion, but best F1 is evaluated with flag.",
        "randomization": "All (case, arm, repetition) jobs SHA-256 shuffled; no reuse of old "
        "answers. "
        "Identical inputs across arms are still generated independently. "
        "Resume matching artifacts only.",
        "primary": "Frozen Jev winner vs all-context answer F1, separately on each dataset. "
        "Question is the unit: average three repetitions before inference. "
        "Require positive simultaneous "
        "97.5% two-sided paired bootstrap lower bounds (Bonferroni, two endpoints), Holm-adjusted "
        "one-sided paired randomization p<.05, and negative normalized-cost "
        "difference upper bounds "
        "on BOTH datasets, with no errors/truncation/unknown usage. Otherwise"
        " no superiority claim.",
        "statistics": {
            "bootstrap_draws": 10000,
            "randomization_draws": 20000,
            "seed": 31918,
            "primary_ci": 0.975,
            "secondary_ci": 0.95,
            "replicate_unit": "question, not individual generation",
        },
        "secondary": "Factorial cutoff main effect, order main effect, interaction; Ettin "
        "comparison; "
        "support precision/recall/all-support, NDCG@5/10; exact match; "
        "within-question answer variance; "
        "subgroups. Secondary CIs descriptive, not confirmatory wins. Latency"
        " exploratory, not SLA.",
        "pricing_usd_per_million": old_protocol["pricing_usd_per_million"],
        "cost_scope": "Actual returned API usage AND cache-normalized cost at uncached rates. "
        "Charge Jev fully in each pipeline repetition though scores shared for the experiment. "
        "Ettin local compute is NOT free: report CPU seconds and break-even $/CPU-hour; "
        "its total monetary cost is unknown. Common retrieval/hosting/network excluded.",
        "failure_policy": "Retain every failed/empty/truncated branch, score unavailable "
        "answers zero. "
        "No selective retries. An orphaned started request blocks resume to "
        "avoid unknown duplicate spend.",
        "usage_stop_usd": 10,
        "code": source_manifest(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {
                n: importlib.metadata.version(n)
                for n in [
                    "typesafe-sdk",
                    "httpx",
                    "sentence-transformers",
                    "torch",
                    "transformers",
                    "numpy",
                    "scipy",
                ]
            },
        },
    }
    write_json(ROOT / "cases.json", datasets)
    write_json(ROOT / "isolation.json", isolation)
    write_json(ROOT / "protocol.json", protocol)
    write_json("benchmarks/research/protocol.json", protocol)
    print("Prepared 96 development and 400 strictly isolated evaluation questions")


def inputs(check_code=True):
    protocol, datasets = read_json(ROOT / "protocol.json"), read_json(ROOT / "cases.json")
    if digest(datasets) != protocol["case_hash"]:
        raise ValueError("Frozen case content changed")
    if check_code and source_manifest() != protocol["code"]:
        raise ValueError("Runtime source/lock changed; do not silently amend the experiment")
    return protocol, datasets


def policies(split, protocol):
    if split == "development":
        return DEV_ARMS
    frozen = read_json(ROOT / "frozen.json")
    if frozen["protocol_hash"] != digest(protocol):
        raise ValueError("Frozen policy/protocol mismatch")
    arms = {p["name"]: p for p in [ALL, *FACTORIAL, frozen["candidate"], frozen["neural"]]}
    return list(arms.values())


def score_path(split, case, scorer):
    return ROOT / split / case["dataset"] / case["id"] / ("score_" + scorer + ".json")


def answer_path(split, case, policy, repetition):
    return ROOT / split / case["dataset"] / case["id"] / f"{policy['name']}_r{repetition}.json"


if __name__ == "__main__":
    prepare()
