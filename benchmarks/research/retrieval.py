"""Separate, frozen BEIR/SciFact corpus retrieval check; no answer generation."""

import argparse
import asyncio
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time
from zipfile import ZipFile

import numpy as np
from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from benchmarks.public.common import digest, read_json, write_json
from rag_jev.models import Document
from rag_jev.provider import CONTEXTUAL_INSTRUCTIONS, CRITERIA, Jev

ROOT = Path("artifacts/retrieval-v1")


class BM25:
    """Same Okapi formula/tokenization as the earlier baseline, indexed once."""

    def __init__(self, documents):
        self.postings = defaultdict(list)
        self.lengths = np.zeros(len(documents))
        for i, document in enumerate(documents):
            tokens = re.findall(r"\w+", document["text"].lower())
            self.lengths[i] = len(tokens)
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))
        self.average = float(self.lengths.mean()) or 1

    def rank(self, query, k=20):
        scores = np.zeros(len(self.lengths))
        for term in set(re.findall(r"\w+", query.lower())):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            ids = np.asarray([p[0] for p in postings])
            tf = np.asarray([p[1] for p in postings])
            idf = math.log(1 + (len(scores) - len(postings) + 0.5) / (len(postings) + 0.5))
            scores[ids] += (
                idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths[ids] / self.average))
            )
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return order, [float(scores[i]) for i in order]


def prepare():
    if (ROOT / "protocol.json").exists():
        raise ValueError("Retrieval protocol already frozen")
    source = read_json(ROOT / "source.json")
    contents = (ROOT / "scifact.zip").read_bytes()
    assert hashlib.sha256(contents).hexdigest() == source["sha256"]
    assert hashlib.md5(contents).hexdigest() == "5f7d1de60b170fc8027bb7898e2efca1"
    with ZipFile(io.BytesIO(contents)) as z:
        corpus = [json.loads(s) for s in z.read("scifact/corpus.jsonl").splitlines()]
        queries = {
            r["_id"]: r["text"]
            for r in [json.loads(s) for s in z.read("scifact/queries.jsonl").splitlines()]
        }
        labels = list(
            csv.DictReader(io.StringIO(z.read("scifact/qrels/test.tsv").decode()), delimiter="\t")
        )
    qrels = defaultdict(dict)
    for row in labels:
        assert row["score"] == "1"  # This study uses binary relevance only.
        qrels[row["query-id"]][row["corpus-id"]] = 1
    documents = [
        {"id": r["_id"], "text": r["title"] + "\n" + r["text"]}
        for r in sorted(corpus, key=lambda r: r["_id"])
    ]
    index = BM25(documents)
    cases = []
    for qid in sorted(qrels):
        indices, scores = index.rank(queries[qid])
        cases.append(
            {
                "id": qid,
                "query": queries[qid],
                "documents": [documents[i] for i in indices],
                "bm25_scores": scores,
                "relevant_ids": list(qrels[qid]),
            }
        )
    main_protocol = read_json("artifacts/research-v3/protocol.json")
    protocol = {
        "prepared_at": datetime.now(UTC).isoformat(),
        "dataset": "BEIR/SciFact test",
        "source": source,
        "cases_hash": digest(cases),
        "queries": len(cases),
        "corpus_documents": len(documents),
        "bm25": "Unicode word tokens, lowercase, k1=1.5 b=.75, "
        "log(1+(N-df+.5)/(df+.5)), no stemming",
        "candidate_depth": 20,
        "output_depth": 10,
        "arms": ["bm25", "jev", "ettin"],
        "scoring": main_protocol["scoring"],
        "neural": main_protocol["neural"],
        "hypotheses": "Separate exploratory retrieval-only check; no tuning and no answer-"
        "accuracy claim. "
        "Full 300-query split, all queries retained even when relevant "
        "evidence is missing from top20. "
        "Report nDCG@10, Recall@10, MRR@10 and first-stage recall@20; no injected gold passages.",
        "uncertainty": "10,000 paired bootstrap draws over connected components of queries "
        "sharing a relevant paper. "
        "Query-weighted mean; resample clusters, ratio of sums to question "
        "counts. Descriptive 95% intervals.",
        "training_overlap": "Unknown; a model may have seen SciFact during training. No "
        "generalization/SOTA claim.",
        "cost": "Jev API input $.042/M, output free. Ettin CPU seconds reported "
        "separately, not free. No generation calls.",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    assert len(cases) == 300
    write_json(ROOT / "cases.json", cases)
    write_json(ROOT / "qrels.json", qrels)
    write_json(ROOT / "protocol.json", protocol)
    write_json("benchmarks/research/retrieval-protocol.json", protocol)
    print("Frozen retrieval check", len(cases), "queries", len(documents), "corpus documents")


def inputs():
    protocol, cases = read_json(ROOT / "protocol.json"), read_json(ROOT / "cases.json")
    assert digest(cases) == protocol["cases_hash"]
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == protocol["code_sha256"]
    return protocol, cases


async def jev():
    protocol, cases = inputs()
    load_dotenv(".env", override=False)
    assert protocol["scoring"]["instructions"] == CONTEXTUAL_INSTRUCTIONS
    assert protocol["scoring"]["criteria"] == CRITERIA
    semaphore = asyncio.Semaphore(4)
    async with AsyncTypeSafeClient(retry=RetryPolicy(max_retries=0)) as client:
        provider = Jev(client=client, model=protocol["scoring"]["model"])

        async def job(case):
            target = ROOT / "scores" / (case["id"] + "_jev.json")
            fingerprint = digest({"case": case, "protocol": protocol})
            if target.exists():
                assert read_json(target)["fingerprint"] == fingerprint
                return
            marker = target.with_suffix(".started.json")
            if marker.exists():
                raise ValueError("Unresolved request; do not silently retry")
            async with semaphore:
                write_json(
                    marker,
                    {"fingerprint": fingerprint, "started_at": datetime.now(UTC).isoformat()},
                )
                start = perf_counter()
                batch = await provider.score_context(
                    case["query"], [Document(**d) for d in case["documents"]], None
                )
                write_json(
                    target,
                    {
                        "fingerprint": fingerprint,
                        "scores": [j.relevance for j in batch.judgments],
                        "input_tokens": batch.usage.input_tokens,
                        "output_tokens": batch.usage.output_tokens,
                        "api_cost_usd": batch.usage.input_tokens * 0.042 / 1e6,
                        "model": batch.judgments[0].model,
                        "elapsed_ms": (perf_counter() - start) * 1000,
                    },
                )
                print("SCORED", case["id"], "jev", flush=True)

        await asyncio.gather(*(job(c) for c in cases))


def neural():
    import torch
    from sentence_transformers import CrossEncoder

    protocol, cases = inputs()
    c = protocol["neural"]
    torch.set_num_threads(c["threads"])
    model = CrossEncoder(
        c["model"],
        revision=c["revision"],
        device="cpu",
        max_length=c["max_length"],
        model_kwargs={"attn_implementation": c["attention"]},
    )
    model.predict([("Warmup", "Warmup passage")], show_progress_bar=False)
    for case in cases:
        target = ROOT / "scores" / (case["id"] + "_ettin.json")
        fingerprint = digest({"case": case, "protocol": protocol})
        if target.exists():
            assert read_json(target)["fingerprint"] == fingerprint
            continue
        pairs = [(case["query"], d["text"]) for d in case["documents"]]
        lengths = [len(model.tokenizer(q, p, truncation=False)["input_ids"]) for q, p in pairs]
        assert max(lengths) <= c["max_length"]
        start, cpu = perf_counter(), process_time()
        scores = model.predict(pairs, batch_size=c["batch_size"], show_progress_bar=False).tolist()
        write_json(
            target,
            {
                "fingerprint": fingerprint,
                "scores": scores,
                "cpu_seconds": process_time() - cpu,
                "elapsed_ms": (perf_counter() - start) * 1000,
                "max_pair_tokens": max(lengths),
                "model": c["model"],
                "revision": c["revision"],
                "truncated_pairs": 0,
            },
        )
        print("SCORED", case["id"], "ettin", flush=True)


def metrics(ids, gold, k=10):
    selected = ids[:k]
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return {
        "ndcg10": sum(1 / math.log2(i + 2) for i, d in enumerate(selected) if d in gold) / ideal,
        "recall10": len(set(selected) & set(gold)) / len(gold),
        "mrr10": next((1 / (i + 1) for i, d in enumerate(selected) if d in gold), 0),
    }


def clusters(cases):
    parent = list(range(len(cases)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, case in enumerate(cases):
        for doc_id in case["relevant_ids"]:
            if doc_id in owners:
                parent[root(i)] = root(owners[doc_id])
            owners[doc_id] = i
    groups = defaultdict(list)
    for i in range(len(cases)):
        groups[root(i)].append(i)
    return list(groups.values())


def cluster_interval(left, right, groups):
    differences = np.asarray(right) - left
    sums = np.array([sum(differences[i] for i in g) for g in groups])
    sizes = np.array([len(g) for g in groups])
    rng = np.random.default_rng(91826)
    indices = rng.integers(len(groups), size=(10000, len(groups)))
    draws = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    return {
        "delta": float(differences.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "query_clusters": len(groups),
        "queries": len(differences),
    }


def report():
    protocol, cases = inputs()
    groups = clusters(cases)
    per_query, runs, costs, cpu = {}, {a: {} for a in protocol["arms"]}, 0, 0
    for case in cases:
        per_query[case["id"]] = {}
        for arm in protocol["arms"]:
            trace = (
                read_json(ROOT / "scores" / (case["id"] + "_" + arm + ".json"))
                if arm != "bm25"
                else None
            )
            if trace:
                assert trace["fingerprint"] == digest({"case": case, "protocol": protocol})
            scores = trace["scores"] if trace else case["bm25_scores"]
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            ids = [case["documents"][i]["id"] for i in order]
            per_query[case["id"]][arm] = metrics(ids, case["relevant_ids"])
            # Rank-surrogate scores preserve our specified tie ordering in trec_eval.
            runs[arm][case["id"]] = {doc_id: float(20 - i) for i, doc_id in enumerate(ids[:10])}
            if arm == "jev":
                costs += trace["api_cost_usd"]
            elif arm == "ettin":
                cpu += trace["cpu_seconds"]
    summaries = {
        arm: {
            metric: float(np.mean([r[arm][metric] for r in per_query.values()]))
            for metric in ["ndcg10", "recall10", "mrr10"]
        }
        for arm in protocol["arms"]
    }
    comparisons = {
        "jev_vs_" + arm: cluster_interval(
            [per_query[c["id"]][arm]["ndcg10"] for c in cases],
            [per_query[c["id"]]["jev"]["ndcg10"] for c in cases],
            groups,
        )
        for arm in ["bm25", "ettin"]
    }
    recall20 = float(
        np.mean(
            [
                len({d["id"] for d in c["documents"]} & set(c["relevant_ids"]))
                / len(c["relevant_ids"])
                for c in cases
            ]
        )
    )
    output = {
        "protocol_hash": digest(protocol),
        "summary": summaries,
        "ndcg10_comparisons_descriptive": comparisons,
        "first_stage_recall20": recall20,
        "jev_total_api_usd": costs,
        "ettin_total_cpu_seconds": cpu,
        "scope": "Full BEIR/SciFact test split; custom BM25 top20 then rerank top10. "
        "No answer generation.",
        "queries": len(cases),
        "corpus_documents": protocol["corpus_documents"],
    }
    write_json(ROOT / "per-query.json", per_query)
    write_json(ROOT / "runs.json", runs)
    write_json(ROOT / "report.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "jev", "neural", "report"])
    args = parser.parse_args()
    if args.stage == "jev":
        asyncio.run(jev())
    else:
        {"prepare": prepare, "neural": neural, "report": report}[args.stage]()
