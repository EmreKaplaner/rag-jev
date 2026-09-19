"""uv run --group benchmark python -m benchmarks.public.prepare"""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from benchmarks.public.common import QA_SYSTEM, SEED, digest, normalize, read_json, write_json
from rag_jev.models import Document, SelectRequest

ROOT = Path("artifacts/public-benchmark")


def prepare():
    if (ROOT / "protocol.json").exists():
        raise SystemExit("Prepared protocol already exists; refusing to replace frozen inputs")
    sources = read_json(ROOT / "data/sources.json")
    for source in sources:
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Dataset checksum mismatch")
    hotpot = pq.read_table(ROOT / "data/hotpotqa.parquet").to_pylist()
    musique = [json.loads(s) for s in (ROOT / "data/musique.jsonl").read_text().splitlines()]
    datasets = {}
    all_questions = set()
    for name, rows in [("hotpotqa", hotpot), ("musique", musique)]:
        cases = []
        for row in sorted(rows, key=lambda r: digest([SEED, name, r["id"]])):
            q = normalize(row["question"])
            if q in all_questions:
                continue
            all_questions.add(q)
            if name == "hotpotqa":
                pairs = list(zip(row["context"]["title"], row["context"]["sentences"], strict=True))
                documents = [
                    Document(id=str(i), text=title + "\n" + "".join(sentences))
                    for i, (title, sentences) in enumerate(pairs)
                ]
                relevant = [
                    str(i)
                    for i, (title, _) in enumerate(pairs)
                    if title in row["supporting_facts"]["title"]
                ]
                aliases = [row["answer"]]
            else:
                if not row["answerable"]:
                    raise ValueError("Expected MuSiQue-Ans, not Full")
                documents = [
                    Document(id=str(p["idx"]), text=p["title"] + "\n" + p["paragraph_text"])
                    for p in row["paragraphs"]
                ]
                relevant = [str(p["idx"]) for p in row["paragraphs"] if p["is_supporting"]]
                aliases = [row["answer"], *row["answer_aliases"]]
            request = SelectRequest(query=row["question"], documents=documents, mode="rerank")
            cases.append(
                {
                    "id": row["id"],
                    "dataset": name,
                    "query": request.query,
                    "documents": [d.model_dump() for d in documents],
                    "relevant_ids": relevant,
                    "answers": aliases,
                }
            )
            if len(cases) == 120:
                break
        datasets[name] = {"development": cases[:20], "evaluation": cases[20:]}
    write_json(ROOT / "cases.json", datasets)
    protocol = {
        "version": 1,
        "seed": SEED,
        "case_hash": digest(datasets),
        "sources": sources,
        "sample": "20 development + 100 disjoint evaluation cases per dataset; SHA-256 ID order",
        "scope": (
            "Public validation-set subsets with dataset-provided candidates, not "
            "full-corpus retrieval or official test leaderboard"
        ),
        "query_expansion": "Not evaluated; supplied candidate lists are fixed",
        "generation": {
            "model": "gpt-5.6-luna",
            "reasoning_effort": "low",
            "max_completion_tokens": 2000,
            "system_prompt": QA_SYSTEM,
        },
        "scorer_model": "jev-1.13.0",
        "baselines": ["all_context", "bm25_top5"],
        "development_candidates": {
            "scorers": ["independent", "contextual"],
            "thresholds": [0.1, 0.2, 0.35, 0.5],
            "top_k": [3, 5, 8],
        },
        "selection_rule": (
            "Maximize dev answer F1 among strategies with lower total cost than "
            "all-context; freeze before evaluation. Report all development trials. If none "
            "qualifies, retain best F1 without a cost claim."
        ),
        "pricing_usd_per_million": {
            "generator_input": 0.2,
            "generator_cached_input": 0.02,
            "generator_output": 1.2,
            "jev_input": 0.042,
            "jev_output": 0,
        },
        "cost_scope": (
            "Reported generation usage and scoring usage; cached input discount when "
            "reported; missing cache details explicitly unknown. Excludes "
            "hosting/network/retrieval. No per-case free retries."
        ),
        "success_rule": (
            "Positive paired F1 difference with bootstrap 95% CI above zero and lower total "
            "API cost on each evaluation subset versus all-context; also report versus BM25 "
            "top5. No success claim if failures/cost gaps remain."
        ),
        "limits": (
            "Subset size, benchmark contamination unknown, one model sample per branch, no "
            "modern cross-encoder baseline yet"
        ),
    }
    write_json(ROOT / "protocol.json", protocol)
    write_json(Path("benchmarks/public/protocol.json"), protocol)
    for name, splits in datasets.items():
        print(name, {split: len(cases) for split, cases in splits.items()})


if __name__ == "__main__":
    prepare()
