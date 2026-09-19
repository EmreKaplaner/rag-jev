"""Resumable experiment runner with per-attempt provenance and no selective retries."""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import math
from datetime import UTC, datetime
from time import perf_counter

import httpx
from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from benchmarks.public.common import answer_metrics, digest, read_json, write_json
from benchmarks.research.design import (
    ROOT,
    answer_path,
    choose,
    inputs,
    policies,
    score_path,
)
from rag_jev.generation import Generator
from rag_jev.models import Document
from rag_jev.provider import Jev

capture_path = contextvars.ContextVar("research_capture_path")


def fingerprint(case, policy, repetition, protocol, selected_ids):
    return digest(
        {
            "case_hash": digest(case),
            "policy": policy,
            "repetition": repetition,
            "protocol_hash": digest(protocol),
            "selected_ids": selected_ids,
        }
    )


def evidence_metrics(case, selected_ids):
    gold, selected = set(case["relevant_ids"]), set(selected_ids)

    def ndcg(k):
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
        actual = sum(
            1 / math.log2(i + 2) for i, doc_id in enumerate(selected_ids[:k]) if doc_id in gold
        )
        return actual / ideal if ideal else 0

    return {
        "support_recall": len(gold & selected) / len(gold),
        "support_precision": len(gold & selected) / len(selected) if selected else 0,
        "all_support": float(gold <= selected),
        "ndcg5": ndcg(5),
        "ndcg10": ndcg(10),
    }


def generation_cost(answer, rates):
    unknown = answer.status != "no_context" and (
        answer.input_tokens is None
        or answer.output_tokens is None
        or answer.cached_input_tokens is None
    )
    # Unknown usage is a flagged known-usage subtotal, never eligible for economic claims.
    inputs, outputs, cached = (
        answer.input_tokens or 0,
        answer.output_tokens or 0,
        answer.cached_input_tokens or 0,
    )
    actual = (
        (inputs - cached) * rates["generator_input"]
        + cached * rates["generator_cached_input"]
        + outputs * rates["generator_output"]
    ) / 1e6
    normalized = (inputs * rates["generator_input"] + outputs * rates["generator_output"]) / 1e6
    return actual, normalized, unknown


def known_spend():
    total = 0.0
    for split in ["development", "evaluation"]:
        for file in (ROOT / split).glob("*/*/*.json"):
            if file.name.endswith((".started.json", ".request.json", ".response.json")):
                continue
            row = read_json(file)
            if file.name == "score_jev.json":
                total += row["api_cost_usd"]
            elif "generation_api_cost_usd" in row:
                total += row["generation_api_cost_usd"]
    return total


async def score(split):
    protocol, datasets = inputs()
    if split == "evaluation":
        policies(split, protocol)
    load_dotenv(".env", override=False)
    rates = protocol["pricing_usd_per_million"]
    semaphore = asyncio.Semaphore(8)
    async with AsyncTypeSafeClient(retry=RetryPolicy(max_retries=0)) as client:
        provider = Jev(client=client, model=protocol["scoring"]["model"])

        async def job(case):
            target = score_path(split, case, "jev")
            identity = digest(
                {
                    "query": case["query"],
                    "documents": case["documents"],
                    "scoring": protocol["scoring"],
                }
            )
            if target.exists():
                if read_json(target)["fingerprint"] != identity:
                    raise ValueError("Scoring cache changed")
                return
            started_path = target.with_suffix(".started.json")
            if started_path.exists():
                raise ValueError("Orphaned scoring attempt; inspect ledger before any new spend")
            async with semaphore:
                started = perf_counter()
                write_json(
                    started_path,
                    {"fingerprint": identity, "started_at": datetime.now(UTC).isoformat()},
                )
                batch = await provider.score_context(
                    case["query"], [Document.model_validate(d) for d in case["documents"]], None
                )
                result = {
                    "fingerprint": identity,
                    "scores": [j.relevance for j in batch.judgments],
                    "models": sorted({j.model for j in batch.judgments}),
                    "input_tokens": batch.usage.input_tokens,
                    "output_tokens": batch.usage.output_tokens,
                    "api_cost_usd": batch.usage.input_tokens * rates["jev_input"] / 1e6,
                    "elapsed_ms": (perf_counter() - started) * 1000,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "cpu_seconds": 0,
                }
                write_json(target, result)
                print("SCORED", case["dataset"], case["id"], flush=True)

        outcomes = await asyncio.gather(
            *(job(c) for data in datasets.values() for c in data[split]), return_exceptions=True
        )
        if any(isinstance(r, BaseException) for r in outcomes):
            raise RuntimeError("Scoring failures retained in attempt ledger; no selective retries")


async def capture_request(request):
    target = capture_path.get()
    write_json(target.with_suffix(".request.json"), json.loads(request.content))


async def capture_response(response):
    await response.aread()
    target = capture_path.get()
    if response.status_code == 200:
        payload = response.json()
        # Only allowlisted response fields. Credentials and headers are never captured.
        safe = {
            k: payload[k]
            for k in ["id", "model", "created", "system_fingerprint", "usage", "choices"]
            if k in payload
        }
    else:
        safe = {"status_code": response.status_code}
    write_json(target.with_suffix(".response.json"), safe)


async def answer(split):
    protocol, datasets = inputs()
    arms = policies(split, protocol)
    load_dotenv(".env", override=False)
    generator = Generator.from_env()
    if generator is None or generator.model != protocol["generation"]["model"]:
        raise ValueError("Configure the pinned generation model")
    await generator.aclose()
    generator.client = httpx.AsyncClient(
        timeout=60,
        follow_redirects=False,
        event_hooks={"request": [capture_request], "response": [capture_response]},
    )
    generator.system_prompt = protocol["generation"]["system_prompt"]
    generator.reasoning_effort = protocol["generation"]["reasoning_effort"]
    generator.max_output_tokens = protocol["generation"]["max_completion_tokens"]
    rates, semaphore = (
        protocol["pricing_usd_per_million"],
        asyncio.Semaphore(protocol["concurrency"]),
    )
    spent = known_spend()
    jobs = [
        (c, p, r)
        for data in datasets.values()
        for c in data[split]
        for p in arms
        for r in range(protocol["repetitions"][split])
    ]
    jobs.sort(
        key=lambda job: digest(
            [protocol["seed"], split, job[0]["dataset"], job[0]["id"], job[1]["name"], job[2]]
        )
    )

    async def job(case, policy, rep):
        nonlocal spent
        target = answer_path(split, case, policy, rep)
        trace = (
            read_json(score_path(split, case, policy["kind"]))
            if policy["kind"] in {"jev", "ettin"}
            else None
        )
        if policy["kind"] == "oracle":
            if split != "development":
                raise ValueError("Oracle only permitted as a development diagnostic")
            selected = [
                i for i, d in enumerate(case["documents"]) if d["id"] in case["relevant_ids"]
            ]
        else:
            selected = choose(case["documents"], trace["scores"] if trace else [], policy)
        documents = [Document.model_validate(case["documents"][i]) for i in selected]
        selected_ids = [d.id for d in documents]
        identity = fingerprint(case, policy, rep, protocol, selected_ids)
        if target.exists():
            if read_json(target)["fingerprint"] != identity:
                raise ValueError("Answer cache changed")
            return
        started_path = target.with_suffix(".started.json")
        if started_path.exists():
            raise ValueError("Orphaned generation attempt; no automatic duplicate call")
        async with semaphore:
            if spent >= protocol["usage_stop_usd"]:
                raise RuntimeError("Usage-based stop threshold reached")
            capture_path.set(target)
            start = datetime.now(UTC).isoformat()
            write_json(started_path, {"fingerprint": identity, "started_at": start})
            result = await generator.generate(case["query"], documents)
            actual, normalized, unknown = generation_cost(result, rates)
            spent += actual
            score_cost = trace["api_cost_usd"] if trace and policy["kind"] == "jev" else 0
            metrics = answer_metrics(
                result.text if result.status == "generated" else "",
                case["answers"],
                case["dataset"],
            )
            record = {
                "case_id": case["id"],
                "dataset": case["dataset"],
                "stratum": case["stratum"],
                "policy": policy,
                "repetition": rep,
                "fingerprint": identity,
                "selected_ids": selected_ids,
                "answer": result.model_dump(),
                **metrics,
                **evidence_metrics(case, selected_ids),
                "unknown_usage": unknown,
                "api_cost_usd": actual + score_cost,
                "normalized_api_cost_usd": normalized + score_cost,
                "generation_api_cost_usd": actual,
                "scoring_api_cost_usd": score_cost,
                "local_cpu_seconds": trace["cpu_seconds"] if trace else 0,
                "total_monetary_cost_known": policy["kind"] != "ettin" and not unknown,
                "stage_ms": result.elapsed_ms + (trace["elapsed_ms"] if trace else 0),
                "started_at": start,
                "finished_at": datetime.now(UTC).isoformat(),
            }
            write_json(target, record)
            print(
                "ANSWERED",
                split,
                case["dataset"],
                case["id"],
                policy["name"],
                rep,
                result.status,
                flush=True,
            )

    try:
        outcomes = await asyncio.gather(*(job(*j) for j in jobs), return_exceptions=True)
    finally:
        await generator.aclose()
    failures = [str(e) for e in outcomes if isinstance(e, BaseException)]
    if failures:
        write_json(ROOT / f"{split}-runner-failures.json", failures)
        raise RuntimeError(f"{len(failures)} jobs incomplete; see attempt ledger")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["score", "answer"])
    parser.add_argument("split", choices=["development", "evaluation"])
    args = parser.parse_args()
    asyncio.run(score(args.split) if args.stage == "score" else answer(args.split))
