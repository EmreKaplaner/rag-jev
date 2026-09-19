"""Pinned BGE and live Jev scores with resumable, budgeted, non-retrying calls."""

import argparse
import asyncio
from time import perf_counter, process_time

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from benchmarks.ecosystem.common import ROOT, Ledger, protocol, select_indices
from benchmarks.ecosystem.retrieve import all_cases
from benchmarks.public.common import digest, read_json, write_json
from rag_jev.models import Document
from rag_jev.provider import Jev, ProviderError


def path(case, name):
    return ROOT / "scores" / f"{case['id']}-{case['retriever']}-{name}.json"


def fingerprint(case, name, p):
    return digest(
        {
            "query": case["query"],
            "documents": case["documents"],
            "stage": name,
            "protocol": digest(p),
        }
    )


def load_score(case, name, p):
    r = read_json(path(case, name))
    assert r["fingerprint"] == fingerprint(case, name, p)
    return r


def bge(limit=None, cases=None):
    import torch
    from sentence_transformers import CrossEncoder

    p = protocol()
    c = p["bge"]
    torch.set_num_threads(4)
    model = CrossEncoder(
        c["model"],
        revision=c["revision"],
        device=c["device"],
        max_length=c["max_length"],
        model_kwargs={"dtype": torch.float16},
    )
    model.predict([("warmup", "warmup")], show_progress_bar=False)
    cases = all_cases() if cases is None else cases
    for case in cases[:limit]:
        if path(case, "bge").exists():
            load_score(case, "bge", p)
            continue
        pairs = [(case["query"], d["text"]) for d in case["documents"]]
        lengths = [
            len(model.tokenizer(q, text, truncation=False)["input_ids"]) for q, text in pairs
        ]
        start, cpu = perf_counter(), process_time()
        scores = (
            model.predict(pairs, batch_size=c["batch_size"], show_progress_bar=False).tolist()
            if pairs
            else []
        )
        write_json(
            path(case, "bge"),
            {
                "fingerprint": fingerprint(case, "bge", p),
                "scores": scores,
                "status": "ok",
                "model": c["model"],
                "revision": c["revision"],
                "elapsed_ms": (perf_counter() - start) * 1000,
                "cpu_seconds": process_time() - cpu,
                "truncated_pairs": sum(n > c["max_length"] for n in lengths),
                "pair_token_lengths": lengths,
                "local_monetary_cost": None,
            },
        )
        print("SCORED", case["id"], case["retriever"], "bge", flush=True)


async def jev(limit=None, cases=None):
    p = protocol()
    load_dotenv(".env", override=False)
    ledger = Ledger(ceiling=p["budget_usd"])
    async with AsyncTypeSafeClient(retry=RetryPolicy(max_retries=0)) as client:
        provider = Jev(model=p["jev"], client=client)

        async def job(case, name):
            if path(case, name).exists():
                return load_score(case, name, p)
            docs = case["documents"]
            if name == "jev_post_bge":
                indices = select_indices("bge", len(docs), bge=load_score(case, "bge", p)["scores"])
                docs = [docs[i] for i in indices]
            fp = fingerprint(case, name, p)
            key = f"score/{case['id']}/{case['retriever']}/{name}"
            ledger.reserve(key, fp, 0.02)
            start = perf_counter()
            cost = None
            try:
                batch = await provider.score_context(
                    case["query"], [Document(id=d["id"], text=d["text"]) for d in docs], None
                )
                cost = batch.usage.input_tokens * p["pricing_usd_per_million"]["jev_input"] / 1e6
                r = {
                    "scores": [j.relevance for j in batch.judgments],
                    "usage": batch.usage.model_dump(),
                    "models": sorted({j.model for j in batch.judgments}),
                    "status": "ok",
                }
            except ProviderError as exc:
                r = {"scores": None, "status": "error", "error_code": exc.code}
            r.update(
                {
                    "fingerprint": fp,
                    "document_ids": [d["id"] for d in docs],
                    "elapsed_ms": (perf_counter() - start) * 1000,
                    "api_cost_usd": cost,
                }
            )
            write_json(path(case, name), r)
            ledger.finish(key, cost, r["status"])
            print("SCORED", case["id"], case["retriever"], name, r["status"], flush=True)
            return r

        jobs = [
            (c, name)
            for c in (all_cases() if cases is None else cases)
            for name in ["jev", "jev_post_bge"]
        ]
        jobs.sort(key=lambda pair: digest([pair[0]["id"], pair[0]["retriever"], pair[1]]))
        jobs = jobs[:limit]
        for offset in range(0, len(jobs), 4):
            results = await asyncio.gather(*(job(c, n) for c, n in jobs[offset : offset + 4]))
            if any(r["status"] != "ok" for r in results):
                raise RuntimeError("Scoring error retained; stop to inspect before further spend")
    print(ledger.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["bge", "jev"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.stage == "bge":
        bge(args.limit)
    else:
        asyncio.run(jev(args.limit))
