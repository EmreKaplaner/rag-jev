"""Test-only service with an HTTP Jev double; never shipped in the Python wheel."""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import Request
from fastapi.responses import JSONResponse
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from rag_jev import ContextSelector, Jev
from rag_jev.demo import CASES, SCORES
from rag_jev.generation import Generator
from rag_jev.server import create_app


def create_network_app():
    sdk = AsyncTypeSafeClient(
        api_key="test-only-not-a-real-key",
        base_url=os.environ["TEST_UPSTREAM_URL"],
        retry=RetryPolicy(max_retries=0),
    )
    selector = ContextSelector(Jev(client=sdk), timeout_ms=200, max_concurrency=2)
    generator = Generator(os.environ["TEST_UPSTREAM_URL"] + "/v1", "controlled-http-model")
    app = create_app(selector, api_token="integration-test-token", generator=generator)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with original_lifespan(app):
            try:
                yield
            finally:
                await sdk.aclose()
                await generator.aclose()

    app.router.lifespan_context = lifespan

    @app.post("/v1/systemone")
    async def mock_jev(request: Request):
        payload = await request.json()
        state = payload["state"]
        if state["query"] == "__fail__":
            return JSONResponse({"error": "deliberate test failure"}, status_code=503)
        if state["query"] == "__timeout__":
            await asyncio.sleep(2)
        if "passages" in state:
            known = {
                doc.text: score
                for case, values in zip(CASES, SCORES, strict=True)
                for doc, score in zip(case.documents, values, strict=True)
            }
            return {
                "model": "fixture-http-not-jev",
                "usage": {"input_tokens": 60, "output_tokens": 3},
                "answers": {
                    str(i): {"type": "noul", "noul": known[doc["text"]]}
                    for i, doc in enumerate(state["passages"])
                },
            }
        for case, values in zip(CASES, SCORES, strict=True):
            for doc, value in zip(case.documents, values, strict=True):
                if state["passage"]["text"] == doc.text:
                    return {
                        "model": "fixture-http-not-jev",
                        "usage": {"input_tokens": 25, "output_tokens": 1},
                        "answers": {"relevant": {"type": "noul", "noul": value}},
                    }
        return JSONResponse({"error": "unknown test fixture"}, status_code=400)

    @app.post("/v1/chat/completions")
    async def mock_generator(request: Request):
        import json

        payload = await request.json()
        content = json.loads(payload["messages"][1]["content"])
        text = " ".join(f"{item['text']} [{item['label']}]" for item in content["evidence"])
        return {
            "model": "controlled-http-model",
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": len(content["evidence"]) * 20 + 50, "completion_tokens": 40},
        }

    return app
