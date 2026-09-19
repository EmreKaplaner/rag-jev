"""Two live, repeated native FlashRAG pipeline runs; separate from benchmark outcomes."""

import asyncio

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from benchmarks.ecosystem.answer import generation_cost
from benchmarks.ecosystem.common import ROOT, Ledger, protocol
from benchmarks.ecosystem.flashrag import run_native
from benchmarks.ecosystem.managed_flashrag import ManagedJevFlashRAGRetriever
from benchmarks.public.common import QA_SYSTEM, digest, read_json, write_json
from rag_jev.generation import Generator
from rag_jev.models import Document
from rag_jev.provider import Jev
from rag_jev.selector import ContextSelector


class OwnedJev(Jev):
    async def aclose(self):
        await self.client.aclose()


def main():
    p = protocol()
    target = ROOT / "live-flashrag-integration.json"
    if target.exists():
        assert read_json(target)["complete"]
        return
    load_dotenv(".env", override=False)
    case = next(
        c
        for c in read_json(ROOT / "retrieval/crag-bm25.json")["cases"]
        if c["split"] == "development"
    )
    source = [
        {"id": d["id"], "contents": d["text"], "metadata": {"url": d["url"]}}
        for d in case["documents"]
    ]

    class Retriever:
        def batch_search(self, queries):
            return [source for _ in queries]

    class LiveGenerator:
        answer = None

        def generate(self, prompts):
            async def run():
                gen = Generator.from_env()
                assert gen is not None and gen.model == p["generator"]
                gen.max_output_tokens = p["generation_max_tokens"]
                gen.reasoning_effort = p["reasoning_effort"]
                gen.system_prompt = QA_SYSTEM
                try:
                    self.answer = await gen.generate(
                        prompts[0]["query"], [Document(**d) for d in prompts[0]["documents"]]
                    )
                    return [self.answer.text]
                finally:
                    await gen.aclose()

            return asyncio.run(run())

    ledger, rows = Ledger(), []
    provider = OwnedJev(
        model=p["jev"], client=AsyncTypeSafeClient(retry=RetryPolicy(max_retries=0))
    )
    selector = ContextSelector(provider, timeout_ms=30000, on_error="raise")
    with ManagedJevFlashRAGRetriever(
        Retriever(),
        selector,
        close_provider=True,
        scoring_strategy="contextual",
        mode="rerank",
        top_n=10,
    ) as retriever:
        for repetition in range(2):
            key = f"integration/flashrag/{repetition}"
            ledger.reserve(
                key,
                digest({"case": case["id"], "protocol": digest(p), "repetition": repetition}),
                0.04,
            )
            generator = LiveGenerator()
            result = run_native([case], retriever, generator)
            trace = retriever.traces[0]
            score_cost = trace.usage.input_tokens * 0.042 / 1e6
            gen_cost = generation_cost(generator.answer, p)
            cost = None if gen_cost is None else score_cost + gen_cost
            rows.append(
                {
                    "repetition": repetition,
                    "selector": trace.model_dump(),
                    "answer": generator.answer.model_dump(),
                    "api_cost_usd": cost,
                    "returned_source_ids": [r["id"] for r in result.retrieval_result[0]],
                }
            )
            status = "ok" if generator.answer.status == "generated" else "error"
            ledger.finish(key, cost, status)
            write_json(
                target,
                {
                    "protocol_hash": digest(p),
                    "complete": False,
                    "case_id": case["id"],
                    "rows": rows,
                },
            )
            assert status == "ok" and len(result.retrieval_result[0]) == 10
            assert all(r in source and "metadata" in r for r in result.retrieval_result[0])
    write_json(
        target,
        {
            "protocol_hash": digest(p),
            "complete": True,
            "case_id": case["id"],
            "rows": rows,
            "scope": (
                "Two real Jev->selection->Luna executions in the same native FlashRAG "
                "pipeline adapter lifetime. No score/answer replay; "
                "not added to benchmark metrics."
            ),
        },
    )
    print("Two live native FlashRAG passes succeeded", ledger.summary())


if __name__ == "__main__":
    main()
