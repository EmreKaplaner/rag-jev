"""Full-corpus BEIR lexical/dense retrieval; CRAG supplied-page chunk retrieval."""

import argparse
from time import perf_counter, process_time

import numpy as np

from benchmarks.ecosystem.common import ROOT, protocol
from benchmarks.public.common import digest, read_json, write_json
from benchmarks.research.retrieval import BM25


def run(dataset, method):
    p = protocol()
    source = read_json(ROOT / "data" / f"{dataset}.cases.json")
    assert digest(source) == p["datasets"][dataset]["cases_hash"]
    target = ROOT / "retrieval" / f"{dataset}-{method}.json"
    if target.exists():
        assert read_json(target)["protocol_hash"] == digest(p)
        return
    start, cpu = perf_counter(), process_time()
    cases, diagnostics = [], {}
    if dataset == "crag":
        if method != "bm25":
            raise ValueError("CRAG pilot specifies BM25 over supplied page chunks")
        for case in source:
            docs = case["documents"]
            order, scores = BM25(docs).rank(case["query"], 20)
            cases.append(
                {
                    **case,
                    "documents": [docs[i] for i in order],
                    "retrieval_scores": scores,
                    "retriever": method,
                    "corpus_documents": len(docs),
                }
            )
    else:
        docs = read_json(ROOT / "data" / f"{dataset}.corpus.json")
        assert digest(docs) == p["datasets"][dataset]["corpus_hash"]
        if method == "bm25":
            model = BM25(docs)
            for case in source:
                order, scores = model.rank(case["query"], 20)
                cases.append(
                    {
                        **case,
                        "documents": [docs[i] for i in order],
                        "retrieval_scores": scores,
                        "retriever": method,
                        "corpus_documents": len(docs),
                    }
                )
        else:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(4)
            c = p["dense"]
            model = SentenceTransformer(c["model"], revision=c["revision"], device=c["device"])
            model.max_seq_length = c["max_length"]
            passages = ["passage: " + d["text"] for d in docs]
            lengths = [
                len(x)
                for x in model.tokenizer(passages, truncation=False, padding=False)["input_ids"]
            ]
            diagnostics = {
                "truncated_corpus_documents": sum(n > c["max_length"] for n in lengths),
                "max_corpus_tokens": max(lengths),
            }
            embeddings = model.encode(
                passages,
                batch_size=c["batch_size"],
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            queries = ["query: " + case["query"] for case in source]
            query_lengths = [
                len(x)
                for x in model.tokenizer(queries, truncation=False, padding=False)["input_ids"]
            ]
            diagnostics["truncated_queries"] = sum(n > c["max_length"] for n in query_lengths)
            qvecs = model.encode(
                queries,
                batch_size=c["batch_size"],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for case, qvec in zip(source, qvecs, strict=True):
                scores = embeddings @ qvec
                order = np.argsort(-scores, kind="stable")[:20]
                cases.append(
                    {
                        **case,
                        "documents": [docs[i] for i in order],
                        "retrieval_scores": [float(scores[i]) for i in order],
                        "retriever": method,
                        "corpus_documents": len(docs),
                    }
                )
    write_json(
        target,
        {
            "protocol_hash": digest(p),
            "cases": cases,
            "diagnostics": diagnostics,
            "elapsed_seconds": perf_counter() - start,
            "cpu_seconds": process_time() - cpu,
            "local_monetary_cost": None,
        },
    )
    print("RETRIEVED", dataset, method, len(cases), flush=True)


def all_cases():
    p = protocol()
    cases = []
    for dataset in p["datasets"]:
        for method in ["bm25"] if dataset == "crag" else ["bm25", "dense"]:
            data = read_json(ROOT / "retrieval" / f"{dataset}-{method}.json")
            assert data["protocol_hash"] == digest(p)
            cases.extend(data["cases"])
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["fiqa", "nfcorpus", "scifact", "crag"])
    parser.add_argument("method", choices=["bm25", "dense"])
    args = parser.parse_args()
    run(args.dataset, args.method)
