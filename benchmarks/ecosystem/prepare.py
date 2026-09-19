"""Verify public data, hash-sample official splits, freeze a bounded pilot before scoring."""

import bz2
import csv
import hashlib
import io
import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

from benchmarks.ecosystem.common import ARMS, ROOT, SEED, fingerprint_files, sample
from benchmarks.public.common import digest, read_json, write_json


def beir(name):
    with ZipFile(ROOT / "data" / f"{name}.zip") as z:
        corpus = [json.loads(x) for x in z.read(f"{name}/corpus.jsonl").splitlines()]
        queries = {
            r["_id"]: r["text"]
            for r in map(json.loads, z.read(f"{name}/queries.jsonl").splitlines())
        }
        docs = [
            {"id": r["_id"], "text": r["title"] + "\n" + r["text"]}
            for r in sorted(corpus, key=lambda x: x["_id"])
        ]
        cases = []
        for split, source in [
            ("development", "train" if name == "scifact" else "dev"),
            ("evaluation", "test"),
        ]:
            qrels = defaultdict(dict)
            for r in csv.DictReader(
                io.StringIO(z.read(f"{name}/qrels/{source}.tsv").decode()), delimiter="\t"
            ):
                qrels[r["query-id"]][r["corpus-id"]] = int(r["score"])
            for qid in sample(qrels, 16 if split == "development" else 100):
                cases.append(
                    {
                        "id": f"{name}-{qid}",
                        "source_id": qid,
                        "dataset": name,
                        "split": split,
                        "query": queries[qid],
                        "qrels": qrels[qid],
                    }
                )
        dev = {c["source_id"] for c in cases if c["split"] == "development"}
        assert not dev & {c["source_id"] for c in cases if c["split"] == "evaluation"}
        write_json(ROOT / "data" / f"{name}.corpus.json", docs)
        write_json(ROOT / "data" / f"{name}.cases.json", cases)
        return {
            "documents": len(docs),
            "cases_hash": digest(cases),
            "corpus_hash": digest(docs),
            "development": 16,
            "evaluation": 100,
        }


def crag():
    # Two streaming passes: retain only selected cases, never the expanded HTML dataset.
    ids = defaultdict(list)
    with bz2.open(ROOT / "data/crag.bz2", "rt") as f:
        for line in f:
            r = json.loads(line)
            ids[r["split"]].append(r["interaction_id"])
    selected = {
        **dict.fromkeys(sample(ids[0], 16), "development"),
        **dict.fromkeys(sample(ids[1], 40), "evaluation"),
    }
    cases = []
    from bs4 import BeautifulSoup

    with bz2.open(ROOT / "data/crag.bz2", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["interaction_id"] not in selected:
                continue
            docs = []
            for page_i, page in enumerate(r["search_results"]):
                soup = BeautifulSoup(page["page_result"], "lxml")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                text = " ".join(soup.get_text(" ", strip=True).split())
                words = (page["page_name"] + " " + page["page_snippet"] + " " + text).split()
                for offset in range(0, len(words), 180):
                    chunk = " ".join(words[offset : offset + 220])
                    if chunk:
                        docs.append(
                            {"id": f"p{page_i}-w{offset}", "text": chunk, "url": page["page_url"]}
                        )
            cases.append(
                {
                    "id": "crag-" + r["interaction_id"],
                    "source_id": r["interaction_id"],
                    "dataset": "crag",
                    "split": selected[r["interaction_id"]],
                    "query": r["query"] + "\nQuery time: " + r["query_time"],
                    "question": r["query"],
                    "references": [r["answer"], *r["alt_ans"]],
                    "domain": r["domain"],
                    "question_type": r["question_type"],
                    "static_or_dynamic": r["static_or_dynamic"],
                    "documents": docs,
                }
            )
    cases.sort(key=lambda c: c["id"])
    write_json(ROOT / "data/crag.cases.json", cases)
    return {
        "cases_hash": digest(cases),
        "development": 16,
        "evaluation": 40,
        "available_splits": {str(k): len(v) for k, v in ids.items()},
        "scope": (
            "Task 1 official supplied five search pages; no external web search. All page "
            "text chunked into 220-word windows, stride 180. BM25 top20 chunks. Query time "
            "is included. No gold passage insertion."
        ),
    }


def main():
    if (ROOT / "protocol.json").exists():
        raise ValueError("Protocol already frozen")
    sources = {}
    for name in ["fiqa", "nfcorpus", "scifact", "crag"]:
        source = read_json(ROOT / "data" / f"{name}.source.json")
        path = ROOT / "data" / ("crag.bz2" if name == "crag" else name + ".zip")
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        assert source["sha256"] == actual
        if name == "crag":
            assert actual == "afa29f2b3facfb5d15aa9cded00d5ec90ff76f3e67279e7b99cfe86659a641ca"
        sources[name] = source
    datasets = {name: beir(name) for name in ["fiqa", "nfcorpus", "scifact"]}
    datasets["crag"] = crag()
    upstream = {
        p.name: subprocess.check_output(
            ["git", "-C", str(p), "rev-parse", "HEAD"], text=True
        ).strip()
        for p in (ROOT / "upstream").iterdir()
        if p.is_dir()
    }
    p = {
        "schema": "rag-jev-ecosystem-pilot-v1",
        "prepared_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "budget_usd": 25,
        "sources": sources,
        "datasets": datasets,
        "upstream": upstream,
        "arms": ARMS,
        "candidate_depth": 20,
        "output_depth": 10,
        "threshold": 0.2,
        "policy": (
            "Fixed a priori .20 cutoff, no pilot winner selection/tuning. Jev ranks full20; "
            "post-BGE filters freshly rescored BGE top10 in BGE order. All errors retained, "
            "never selective retry."
        ),
        "splits": (
            "Hash sample official dev/test separately. SciFact train supplies development. "
            "Evaluation labels used only for reporting. SciFact test previously exposed: "
            "replication, not pristine holdout. Other model training exposure unknown."
        ),
        "dense": {
            "model": "intfloat/e5-small-v2",
            "revision": "ffb93f3bd4047442299a41ebb6fa998a38507c52",
            "max_length": 512,
            "device": "mps",
            "batch_size": 32,
            "prefixes": ["query: ", "passage: "],
            "similarity": "normalized dot product, exact full corpus",
        },
        "bge": {
            "model": "BAAI/bge-reranker-v2-m3",
            "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "max_length": 1024,
            "device": "mps",
            "batch_size": 4,
        },
        "bm25": (
            "Existing audited Okapi Unicode-word index; k1=1.5 b=.75, no stemming. Custom "
            "baseline, not Anserini/Pyserini."
        ),
        "truncation": (
            "Dense encoder truncates at512 model tokens; BGE at1024. Count and report "
            "affected documents/pairs; Jev receives full candidate text subject to explicit "
            "provider limit."
        ),
        "jev": "jev-1.13.0",
        "generator": "gpt-5.6-luna",
        "generation_max_tokens": 2000,
        "reasoning_effort": "low",
        "repetitions": 1,
        "pricing_usd_per_million": {
            "jev_input": 0.042,
            "generator_input": 0.2,
            "generator_cached_input": 0.02,
            "generator_output": 1.2,
        },
        "cost": (
            "Returned API usage at frozen public rates, including selector. Also normalize "
            "cache. Local embedding/reranking compute reported separately, monetary cost "
            "unknown; no total-cost superiority claim. No invoice access."
        ),
        "statistics": (
            "Exploratory paired95% cluster bootstrap, 10000 draws. BEIR cluster queries "
            "sharing relevant docs; CRAG question units. No multiplicity-corrected "
            "superiority/SOTA claim from pilot. Graded nDCG@10/Recall@10 and candidate "
            "Recall@20 via pytrec_eval."
        ),
        "qa_metrics": (
            "CRAG short-answer EM/F1 diagnostics, explicitly not official CRAG semantic "
            "judge scores. Native FlashRAG pipeline execution. RAGChecker optional "
            "hash-selected paired diagnosis; same-model judge bias disclosed. Human blinded "
            "review remains independent and pending."
        ),
        "code": fingerprint_files(
            [
                *Path("benchmarks/ecosystem").glob("*.py"),
                Path("src/rag_jev/provider.py"),
                Path("src/rag_jev/generation.py"),
                Path("src/rag_jev/models.py"),
                Path("benchmarks/public/common.py"),
                Path("benchmarks/research/retrieval.py"),
            ]
        ),
    }
    write_json(ROOT / "protocol.json", p)
    write_json("benchmarks/ecosystem/protocol.json", p)
    print(
        "FROZEN",
        {n: {k: d[k] for k in ["development", "evaluation"]} for n, d in datasets.items()},
        flush=True,
    )


if __name__ == "__main__":
    main()
