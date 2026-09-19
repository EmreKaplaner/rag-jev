"""Frozen synthetic robustness/latency diagnostics. Never a public benchmark claim."""

import argparse
import asyncio
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from rag_jev.evaluation import EvalCase, measure
from rag_jev.provider import Jev
from rag_jev.selector import ContextSelector

ROOT = Path("artifacts/robustness-v1")


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare():
    if (ROOT / "protocol.json").exists():
        raise ValueError("Do not overwrite the frozen workload")
    cases = json.loads(Path("benchmarks/robustness/cases.json").read_text())
    protocol = {
        "prepared_at": datetime.now(UTC).isoformat(),
        "cases_hash": checksum(cases),
        "model": "jev-1.13.0",
        "strategies": ["independent", "contextual"],
        "client_concurrency": [1, 4, 8],
        "selector_concurrency": 8,
        "timeout_ms": 10000,
        "retries": 0,
        "threshold": 0.2,
        "mode": "filter_and_rerank",
        "input_usd_per_million": 0.042,
        "scope": "18 synthetic, hand-labeled stress cases; no answer generation. "
        "Quality is diagnostic, not representative accuracy. No tuning from these results.",
        "latency": "Per-call selection duration includes the selector semaphore queue; "
        "client queue+stage includes waiting for the workload concurrency slot. "
        "Cold first call included. p95 is nearest rank on 18 cases per condition; "
        "not a production SLA or hardware-neutral comparison.",
        "code": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path("benchmarks/robustness/run.py"), *Path("src/rag_jev").glob("*.py")]
        },
    }
    save(ROOT / "protocol.json", protocol)
    save(Path("benchmarks/robustness/protocol.json"), protocol)
    print("Frozen synthetic workload", len(cases), "cases")


async def run():
    load_dotenv(".env", override=False)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    cases = json.loads(await asyncio.to_thread(Path("benchmarks/robustness/cases.json").read_text))
    assert checksum(cases) == protocol["cases_hash"]
    for path, sha in protocol["code"].items():
        assert hashlib.sha256(await asyncio.to_thread(Path(path).read_bytes)).hexdigest() == sha
    async with AsyncTypeSafeClient(retry=RetryPolicy(max_retries=0)) as client:
        provider = Jev(client=client, model=protocol["model"])
        for strategy in protocol["strategies"]:
            for concurrency in protocol["client_concurrency"]:
                selector = ContextSelector(
                    provider,
                    timeout_ms=protocol["timeout_ms"],
                    max_concurrency=protocol["selector_concurrency"],
                )
                semaphore = asyncio.Semaphore(concurrency)
                workload = list(cases)
                random.Random(91926).shuffle(workload)

                async def job(
                    raw,
                    strategy=strategy,
                    concurrency=concurrency,
                    semaphore=semaphore,
                    selector=selector,
                ):
                    path = ROOT / "runs" / f"{strategy}-{concurrency}-{raw['id']}.json"
                    identity = checksum(
                        {
                            "protocol": protocol,
                            "case": raw,
                            "strategy": strategy,
                            "concurrency": concurrency,
                        }
                    )
                    if path.exists():
                        assert json.loads(path.read_text())["identity"] == identity
                        return
                    marker = path.with_suffix(".started.json")
                    if marker.exists():
                        raise ValueError("Unresolved attempt; do not silently retry")
                    start = perf_counter()
                    async with semaphore:
                        save(
                            marker,
                            {"identity": identity, "started_at": datetime.now(UTC).isoformat()},
                        )
                        case = EvalCase.model_validate(
                            {k: v for k, v in raw.items() if k != "category"}
                        )
                        result = await selector.select(
                            query=case.query,
                            retrieval_query=case.retrieval_query,
                            documents=case.documents,
                            scoring_strategy=strategy,
                            mode=protocol["mode"],
                            min_relevance=protocol["threshold"],
                        )
                    metric = (
                        measure(case, result.documents).model_dump()
                        if result.status != "bypassed"
                        else None
                    )
                    save(
                        path,
                        {
                            "identity": identity,
                            "case_id": case.id,
                            "category": raw["category"],
                            "strategy": strategy,
                            "concurrency": concurrency,
                            "result": result.model_dump(mode="json"),
                            "metrics": metric,
                            "client_queue_plus_stage_ms": (perf_counter() - start) * 1000,
                        },
                    )
                    print(strategy, concurrency, case.id, result.status, flush=True)

                await asyncio.gather(*(job(c) for c in workload))


def report():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    cases = json.loads(Path("benchmarks/robustness/cases.json").read_text())
    grouped = defaultdict(list)
    for strategy in protocol["strategies"]:
        for concurrency in protocol["client_concurrency"]:
            for case in cases:
                row = json.loads(
                    (ROOT / "runs" / f"{strategy}-{concurrency}-{case['id']}.json").read_text()
                )
                assert row["identity"] == checksum(
                    {
                        "protocol": protocol,
                        "case": case,
                        "strategy": strategy,
                        "concurrency": concurrency,
                    }
                )
                grouped[(strategy, concurrency)].append(row)
    output = {
        "protocol_hash": checksum(protocol),
        "scope": protocol["scope"],
        "latency": protocol["latency"],
        "conditions": [],
    }
    for (strategy, concurrency), rows in grouped.items():
        elapsed = sorted(r["result"]["elapsed_ms"] for r in rows)
        categories = {}
        for category in sorted({r["category"] for r in rows}):
            group = [r for r in rows if r["category"] == category]
            measured = [r["metrics"] for r in group if r["metrics"] is not None]
            categories[category] = {
                "cases": len(group),
                "bypassed": len(group) - len(measured),
                "evidence_recall": mean(
                    m["evidence_recall"] for m in measured if m["evidence_recall"] is not None
                )
                if any(m["evidence_recall"] is not None for m in measured)
                else None,
                "correctly_empty": mean(
                    m["correctly_empty"] for m in measured if m["correctly_empty"] is not None
                )
                if any(m["correctly_empty"] is not None for m in measured)
                else None,
                "selected_ids": {r["case_id"]: r["result"]["selected_ids"] for r in group},
            }
        output["conditions"].append(
            {
                "strategy": strategy,
                "concurrency": concurrency,
                "calls": len(rows),
                "bypassed": sum(r["result"]["status"] == "bypassed" for r in rows),
                "unknown_usage": sum(not r["result"]["usage"]["complete"] for r in rows),
                "p50_ms": median(elapsed),
                "p95_ms": elapsed[math.ceil(0.95 * len(elapsed)) - 1],
                "mean_client_queue_plus_stage_ms": mean(
                    r["client_queue_plus_stage_ms"] for r in rows
                ),
                "known_scoring_api_usd": sum(r["result"]["usage"]["input_tokens"] for r in rows)
                * protocol["input_usd_per_million"]
                / 1e6,
                "categories": categories,
            }
        )
    save(ROOT / "report.json", output)
    save(Path("benchmarks/robustness/results.json"), output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "report"])
    stage = parser.parse_args().stage
    if stage == "run":
        asyncio.run(run())
    else:
        {"prepare": prepare, "report": report}[stage]()
