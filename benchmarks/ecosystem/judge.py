"""Small, predeclared native RAGChecker diagnosis, with the same campaign spend cap."""

import hashlib
import json
from pathlib import Path

import httpx
from dotenv import load_dotenv

from benchmarks.ecosystem.common import ARMS, ROOT, Ledger, protocol, sample
from benchmarks.public.common import digest, read_json, write_json
from rag_jev.generation import Generator


class AuditedJudge:
    """RefChecker's documented custom_llm_api_func(prompts)->list[str] bridge.

    Preserve upstream prompts exactly. No automatic retries or hidden vendor SDK calls.
    Cache identical prompts, including shared ground-truth extraction across arms.
    """

    def __init__(self, p):
        load_dotenv(".env", override=False)
        gen = Generator.from_env()
        if gen is None or gen.model != p["generator"]:
            raise ValueError("Configure the frozen judge model")
        self.base, self.key, self.model = gen.base_url, gen.api_key, gen.model
        self.p = p

    def __call__(self, prompts):
        output = []
        ledger = Ledger()
        with httpx.Client(timeout=120) as client:
            for prompt in prompts:
                messages = (
                    [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
                )
                body = {
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": 2000,
                    "reasoning_effort": "low",
                }
                fp = digest({"body": body, "protocol": digest(self.p)})
                target = ROOT / "judge-requests" / f"{fp}.json"
                if target.exists():
                    record = read_json(target)
                    if record["status"] != "ok":
                        raise RuntimeError("Prior judge error retained; no implicit retry")
                    output.append(record["text"])
                    continue
                input_bound = len(json.dumps(body).encode()) + 4096
                if input_bound > 250000:
                    raise ValueError("Judge input exceeds bounded pilot request")
                key = "judge/" + fp
                ledger.reserve(key, fp, input_bound * 0.2 / 1e6 + 0.0024 + 0.01)
                record = {"request": body, "status": "error", "cost_usd": None}
                try:
                    response = client.post(
                        self.base + "/chat/completions",
                        headers={"Authorization": "Bearer " + self.key},
                        json=body,
                    )
                    record["http_status"] = response.status_code
                    response.raise_for_status()
                    data = response.json()
                    usage = data.get("usage", {})
                    inputs, outputs = usage.get("prompt_tokens"), usage.get("completion_tokens")
                    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                    if (
                        any(type(n) is not int or n < 0 for n in [inputs, outputs, cached])
                        or cached > inputs
                    ):
                        raise ValueError("Unknown judge usage")
                    cost = ((inputs - cached) * 0.2 + cached * 0.02 + outputs * 1.2) / 1e6
                    record.update(
                        {
                            "cost_usd": cost,
                            "usage": usage,
                            "model": data.get("model"),
                            "choice": data["choices"][0],
                        }
                    )
                    choice = data["choices"][0]
                    if (
                        choice.get("finish_reason") != "stop"
                        or not choice["message"].get("content", "").strip()
                    ):
                        raise ValueError("Judge did not complete")
                    record.update({"status": "ok", "text": choice["message"]["content"]})
                except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                    record["error_type"] = type(exc).__name__
                write_json(target, record)
                ledger.finish(key, record["cost_usd"], record["status"])
                if record["status"] != "ok":
                    raise RuntimeError("Judge error retained; stop before further spend")
                output.append(record["text"])
        return output


def main():
    from ragchecker import RAGChecker, RAGResults
    from ragchecker.container import RAGResult, RetrievedDoc

    p = protocol()
    cases = [
        c
        for c in read_json(ROOT / "retrieval/crag-bm25.json")["cases"]
        if c["split"] == "evaluation"
    ]
    ids = sample([c["id"] for c in cases], 8)
    judge_protocol = {
        "base_protocol_hash": digest(p),
        "case_ids": ids,
        "arms": ARMS,
        "model": p["generator"],
        "max_completion_tokens": 2000,
        "reasoning_effort": "low",
        "scope": (
            "8 hash-selected paired CRAG test questions. All five arms. Native RAGChecker "
            "diagnostics; same-model generation/judging bias; not independent human "
            "adjudication or official CRAG scores."
        ),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    frozen = ROOT / "judge-protocol.json"
    if frozen.exists():
        assert read_json(frozen) == judge_protocol
    else:
        write_json(frozen, judge_protocol)
    records = []
    for case in cases:
        if case["id"] not in ids:
            continue
        for arm in ARMS:
            answer = read_json(ROOT / "answers" / f"{case['id']}-{arm}.json")
            records.append(
                RAGResult(
                    query_id=case["id"] + "/" + arm,
                    query=case["query"],
                    gt_answer=case["references"][0],
                    response=answer["prediction"],
                    retrieved_context=[
                        RetrievedDoc(doc_id=d["id"], text=d["text"])
                        for d in answer["prompt"]["documents"]
                    ],
                )
            )
    checkpoint = ROOT / "ragchecker-full.json"
    results = (
        RAGResults.from_json(checkpoint.read_text())
        if checkpoint.exists()
        else RAGResults(results=records)
    )
    assert [r.query_id for r in results.results] == [r.query_id for r in records]
    checker = RAGChecker(
        extractor_name=p["generator"],
        checker_name=p["generator"],
        batch_size_extractor=8,
        batch_size_checker=8,
        extractor_max_new_tokens=2000,
        custom_llm_api_func=AuditedJudge(p),
    )
    checker.evaluate(results, save_path=str(checkpoint))
    output = {
        "protocol": judge_protocol,
        "per_arm": {},
        "coverage": {
            "cases": len(results.results),
            "empty_response_claims": sum(not r.response_claims for r in results.results),
            "empty_reference_claims": sum(not r.gt_answer_claims for r in results.results),
        },
        "budget": Ledger().summary(),
    }
    for arm in ARMS:
        group = RAGResults(results=[r for r in results.results if r.query_id.endswith("/" + arm)])
        checker.evaluate(group)
        output["per_arm"][arm] = group.metrics
    write_json(ROOT / "ragchecker-report.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
