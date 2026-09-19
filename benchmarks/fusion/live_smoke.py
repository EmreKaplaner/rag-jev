"""Explicit paid HTTP smoke: SQLite retrieval -> Jev -> three-way answer comparison.

Run from this checkout with configured .env. Uses the existing $25 pilot ledger;
reserves before calls, never silently retries this campaign, and publishes only
the repository's example passages and allowlisted results. This is not a benchmark.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from benchmarks.ecosystem.common import Ledger
from benchmarks.public.common import digest, read_json, write_json
from examples.local_rag import retrieve
from rag_jev.comparison import RankingReport
from rag_jev.workbench import RunView


def main():
    load_dotenv(override=False)
    p = read_json("benchmarks/ecosystem/protocol.json")
    if os.getenv("RAG_JEV_GENERATION_MODEL") != p["generator"]:
        raise ValueError("Configure the original pilot generator before this smoke test")
    ledger = Ledger(ceiling=25)
    output = Path("artifacts/fusion-live")
    output.mkdir(parents=True, exist_ok=True)
    public = Path("benchmarks/fusion/live-smoke.json")
    if public.exists():
        print("Already completed; read benchmarks/fusion/live-smoke.json. No new calls.")
        return
    query = "How many days do I have to request a refund, and what do I need?"
    documents = retrieve(Path("examples/knowledge"), query, 20)
    body = {
        "request": {
            "query": query,
            "documents": [d.model_dump() for d in documents],
            "mode": "fusion",
            "top_n": 2,
            "scoring_strategy": "contextual",
        }
    }
    assert documents and sum(len(d.text) for d in documents) < 10000
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "RAG_JEV_MODEL": p["jev"],
        "RAG_JEV_TIMEOUT_MS": "30000",
        "RAG_JEV_GENERATION_MAX_OUTPUT_TOKENS": "2000",
        "RAG_JEV_GENERATION_REASONING_EFFORT": p["reasoning_effort"],
        "RAG_JEV_SCORING_INPUT_PER_MILLION": "0.042",
        "RAG_JEV_SCORING_OUTPUT_PER_MILLION": "0",
        "RAG_JEV_GENERATION_INPUT_PER_MILLION": "0.2",
        "RAG_JEV_GENERATION_OUTPUT_PER_MILLION": "1.2",
    }
    token = env.get("RAG_JEV_API_TOKEN", "")
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "rag_jev.cli", "serve", "--port", str(port)],
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                timeout=210,
                headers={"Authorization": "Bearer " + token} if token else {},
            ) as client:
                for _ in range(100):
                    try:
                        if client.get("/healthz", timeout=0.5).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Local smoke service did not start")
                ledger.reserve("fusion-live/scoring-v1", digest(body), 0.1)
                response = client.post("/v1/runs", json=body)
                if response.status_code != 200:
                    ledger.finish("fusion-live/scoring-v1", None, "failed")
                    raise RuntimeError("Live scoring failed; reservation retained")
                view = RunView.model_validate(response.json())
                write_json(output / "run.json", view.model_dump(mode="json"))
                cost = view.selection.usage.input_tokens * 0.042 / 1e6
                ledger.finish("fusion-live/scoring-v1", cost)
                comparison_body = {"record": view.record.model_dump(mode="json"), "top_n": 2}
                ledger.reserve("fusion-live/generation-v1", digest(comparison_body), 0.1)
                response = client.post("/v1/compare-rankings", json=comparison_body)
                if response.status_code != 200:
                    ledger.finish("fusion-live/generation-v1", None, "failed")
                    raise RuntimeError("Live comparison failed; reservation retained")
                report = RankingReport.model_validate(response.json())
                write_json(output / "comparison.json", report.model_dump(mode="json"))
                costs = []
                for arm in report.arms:
                    if arm.answer_reused_from is not None:
                        continue
                    a = arm.answer
                    if a is None or a.input_tokens is None or a.output_tokens is None:
                        ledger.finish("fusion-live/generation-v1", None, "usage_unknown")
                        raise RuntimeError("Unknown generation usage; reservation retained")
                    cached = a.cached_input_tokens or 0
                    costs.append(
                        ((a.input_tokens - cached) * 0.2 + cached * 0.02 + a.output_tokens * 1.2)
                        / 1e6
                    )
                ledger.finish("fusion-live/generation-v1", sum(costs))
                assert all(a.answer.status == "generated" for a in report.arms)
                assert all(not a.answer.unknown_citations for a in report.arms)
                assert all(a.answer.finish_reason != "length" for a in report.arms)
                assert report.scoring_calls == 0
                assert all(d.metadata for d in view.record.request.documents)
                write_json(
                    public,
                    {
                        "scope": (
                            "One public example question over SQLite FTS5/BM25 retrieval; live "
                            "HTTP Jev and configured generation. Integration smoke only, "
                            "no quality benchmark or independent grading."
                        ),
                        "record": view.record.model_dump(mode="json"),
                        "comparison": report.model_dump(mode="json"),
                        "new_usage_cost_usd": cost + sum(costs),
                        "pilot_budget_after_smoke": ledger.summary(),
                        "price_basis": (
                            "Original pilot rates; returned token usage with cached inputs "
                            "accounted for. Not invoice verified."
                        ),
                    },
                )
                print(
                    json.dumps(
                        {
                            "status": "passed",
                            "candidates": len(documents),
                            "new_usage_cost_usd": cost + sum(costs),
                            "generation_calls": report.generation_calls,
                            "pilot_budget": ledger.summary(),
                        },
                        indent=2,
                    )
                )
        finally:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
