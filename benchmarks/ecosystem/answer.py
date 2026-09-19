"""Five-arm CRAG answer pilot executed by native FlashRAG with frozen selector traces."""

import argparse
import asyncio

from dotenv import load_dotenv

from benchmarks.ecosystem.common import ARMS, ROOT, Ledger, protocol, select_indices
from benchmarks.ecosystem.flashrag import run_native
from benchmarks.ecosystem.score import load_score
from benchmarks.public.common import QA_SYSTEM, digest, read_json, write_json
from rag_jev.generation import Answer, Generator
from rag_jev.models import Document


def selected(case, arm, p):
    args = {}
    if arm in {"bge", "bge_jev_filter"}:
        args["bge"] = load_score(case, "bge", p)["scores"]
    if arm in {"jev_rerank", "jev_filter"}:
        args["jev"] = load_score(case, "jev", p)["scores"]
    if arm == "bge_jev_filter":
        args["post"] = load_score(case, "jev_post_bge", p)["scores"]
    ids = select_indices(
        arm, len(case["documents"]), threshold=p["threshold"], top_k=p["output_depth"], **args
    )
    return [case["documents"][i] for i in ids]


class FrozenRetriever:
    """Replays full-corpus/page retrieval and already-paid selector scores, not answers."""

    def __init__(self, cases, arm, p):
        self.cases = {c["query"]: c for c in cases}
        if len(self.cases) != len(cases):
            raise ValueError("Duplicate query needs explicit case addressing")
        self.arm, self.p = arm, p

    def batch_search(self, queries):
        return [
            [
                {"id": d["id"], "contents": d["text"]}
                for d in selected(self.cases[q], self.arm, self.p)
            ]
            for q in queries
        ]


def generation_cost(answer, p, normalized=False):
    if answer.input_tokens is None or answer.output_tokens is None:
        return None
    rates = p["pricing_usd_per_million"]
    cached = 0 if normalized else (answer.cached_input_tokens or 0)
    return (
        (answer.input_tokens - cached) * rates["generator_input"]
        + cached * rates["generator_cached_input"]
        + answer.output_tokens * rates["generator_output"]
    ) / 1e6


class BudgetedGenerator:
    def __init__(self, cases, arm, p):
        self.cases = {c["query"]: c for c in cases}
        self.arm, self.p = arm, p

    def generate(self, prompts):
        return asyncio.run(self.agenerate(prompts))

    async def agenerate(self, prompts):
        load_dotenv(".env", override=False)
        gen = Generator.from_env()
        if gen is None or gen.model != self.p["generator"]:
            raise ValueError("Frozen generator must be configured in .env")
        gen.max_output_tokens = self.p["generation_max_tokens"]
        gen.reasoning_effort = self.p["reasoning_effort"]
        gen.system_prompt = QA_SYSTEM
        ledger = Ledger(ceiling=self.p["budget_usd"])

        async def one(prompt):
            case = self.cases[prompt["query"]]
            path = ROOT / "answers" / f"{case['id']}-{self.arm}.json"
            fp = digest(
                {"prompt": prompt, "system": QA_SYSTEM, "protocol": digest(self.p), "arm": self.arm}
            )
            if path.exists():
                r = read_json(path)
                assert r["fingerprint"] == fp
                return r["prediction"]
            key = f"answer/{case['id']}/{self.arm}"
            # Byte length upper-bounds ordinary text token count; add ample framing margin.
            import json

            input_bound = len(json.dumps(prompt).encode()) + len(QA_SYSTEM.encode()) + 4096
            if input_bound > 250000:
                raise ValueError("Prompt exceeds pilot per-request bound")
            reserve = input_bound * 0.2 / 1e6 + gen.max_output_tokens * 1.2 / 1e6 + 0.01
            ledger.reserve(key, fp, reserve)
            docs = [Document(**d) for d in prompt["documents"]]
            if docs:
                answer = await gen.generate(prompt["query"], docs)
            else:
                answer = Answer(
                    text="unanswerable", status="no_context", input_tokens=0, output_tokens=0
                )
            prediction = answer.text if answer.status in {"generated", "no_context"} else ""
            cost = generation_cost(answer, self.p)
            write_json(
                path,
                {
                    "fingerprint": fp,
                    "prompt": prompt,
                    "answer": answer.model_dump(),
                    "prediction": prediction,
                    "api_cost_usd": cost,
                    "normalized_api_cost_usd": generation_cost(answer, self.p, True),
                },
            )
            ledger.finish(key, cost, answer.status)
            print("ANSWERED", case["id"], self.arm, answer.status, flush=True)
            return prediction

        try:
            output = []
            for offset in range(0, len(prompts), 4):
                output.extend(
                    await asyncio.gather(*(one(prompt) for prompt in prompts[offset : offset + 4]))
                )
            return output
        finally:
            await gen.aclose()


def run(limit=None):
    p = protocol()
    data = read_json(ROOT / "retrieval/crag-bm25.json")
    assert data["protocol_hash"] == digest(p)
    cases = sorted(data["cases"], key=lambda c: digest(c["id"]))[:limit]
    # Interleave arm order by case hash to reduce provider/time/cache ordering confounds.
    for case in cases:
        for arm in sorted(ARMS, key=lambda arm: digest([case["id"], arm])):
            result = run_native(
                [case], FrozenRetriever([case], arm, p), BudgetedGenerator([case], arm, p)
            )
            assert len(result.pred) == 1
    print(Ledger().summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    run(parser.parse_args().limit)
