"""HTTP transport for the same selection engine used by the Python library."""

from __future__ import annotations

import hmac
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_jev.comparison import RankingReport, RankingRequest, compare_rankings, generate_rankings
from rag_jev.generation import Generator
from rag_jev.models import SelectionResult, SelectRequest
from rag_jev.provider import Jev, ProviderError
from rag_jev.selector import ContextSelector
from rag_jev.workbench import (
    Comparison,
    ImportedRun,
    Price,
    ReplayBundle,
    ReplayRequest,
    RunView,
    ScoredRun,
    StudyRequest,
    compare_run,
    input_hash,
    replay_run,
    score_run,
)


class BodyLimit:
    def __init__(self, app: ASGIApp, max_bytes: int = 2_000_000) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await JSONResponse({"error": "request_too_large"}, status_code=413)(
                    scope, receive, send
                )
                return
            if not message.get("more_body", False):
                break
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)


def create_app(
    selector: ContextSelector | None = None,
    *,
    api_token: str | None = None,
    generator: Generator | None = None,
    replay_only: bool = False,
) -> FastAPI:
    token = api_token if api_token is not None else os.getenv("RAG_JEV_API_TOKEN")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        provider: Jev | None = None
        if replay_only:
            app.state.selector = None
        elif selector is None:
            provider = Jev(model=os.getenv("RAG_JEV_MODEL", "jev-latest"))
            app.state.selector = ContextSelector(
                provider,
                timeout_ms=float(os.getenv("RAG_JEV_TIMEOUT_MS", "5000")),
                max_concurrency=int(os.getenv("RAG_JEV_MAX_CONCURRENCY", "8")),
                on_error=os.getenv("RAG_JEV_ON_ERROR", "passthrough"),  # type: ignore[arg-type]
            )
        else:
            app.state.selector = selector
        app.state.generator = None if replay_only else generator or Generator.from_env()
        app.state.generation_price = Price.from_env("RAG_JEV_GENERATION")
        app.state.selection_price = Price.from_env("RAG_JEV_SCORING")
        try:
            yield
        finally:
            if provider:
                await provider.aclose()
            if app.state.generator and generator is None:
                await app.state.generator.aclose()

    app = FastAPI(
        title="rag-jev",
        version="0.2.0",
        description="Select useful context from existing retrieval results. No ingestion required.",
        lifespan=lifespan,
    )
    app.add_middleware(BodyLimit)

    @app.middleware("http")
    async def browser_origin(request: Request, call_next):  # type: ignore[no-untyped-def]
        origin = request.headers.get("origin")
        expected = str(request.base_url).rstrip("/")
        if request.method == "POST" and origin and origin != expected:
            return JSONResponse({"error": "cross_origin_request"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        if request.url.path == "/":
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
            )
        return response

    async def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        if token and not hmac.compare_digest(
            (authorization or "").encode(), ("Bearer " + token).encode()
        ):
            raise HTTPException(status_code=401, detail="invalid_service_token")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default error echoes the input. Return useful paths without private content.
        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_request",
                "issues": [
                    {"loc": list(e["loc"]), "type": e["type"], "message": e["msg"]}
                    for e in exc.errors()
                ],
            },
        )

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    assets = Path(__file__).parent / "static"
    app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    async def playground() -> FileResponse:
        return FileResponse(assets / "index.html")

    @app.get("/benchmarks", include_in_schema=False)
    async def public_benchmarks() -> FileResponse:
        return FileResponse(assets / "benchmarks.html")

    @app.get("/research", include_in_schema=False)
    async def research_index() -> FileResponse:
        return FileResponse(assets / "research.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(assets / "brand/favicon.ico", media_type="image/vnd.microsoft.icon")

    @app.get("/review", include_in_schema=False)
    async def human_review() -> FileResponse:
        return FileResponse(assets / "review.html")

    @app.get("/v1/config", dependencies=[Depends(authorize)])
    async def config(request: Request) -> dict[str, object]:
        gen = request.app.state.generator
        return {
            "replay_only": replay_only,
            "scoring_configured": not replay_only,
            "generation_configured": gen is not None,
            "generation_model": gen.model if gen else None,
        }

    @app.get("/v1/recordings", dependencies=[Depends(authorize)])
    async def recordings() -> list[dict[str, object]]:
        return json.loads((assets / "replays" / "catalog.json").read_text())  # type: ignore[no-any-return]

    @app.get("/v1/examples", dependencies=[Depends(authorize)])
    async def examples() -> list[dict[str, object]]:
        from rag_jev.demo import CASES

        return [case.model_dump() for case in CASES]

    @app.get("/v1/fixtures/{case_id}", response_model=RunView, dependencies=[Depends(authorize)])
    async def fixture(case_id: str) -> RunView:
        from rag_jev.demo import CASES, fixture_traces

        for case, trace in zip(CASES, fixture_traces(), strict=True):
            if case.id != case_id:
                continue
            req = SelectRequest(query=case.query, documents=case.documents, min_relevance=0.2)
            record = ScoredRun(
                request=req,
                relevant_ids=case.relevant_ids,
                created_at=datetime.now(UTC),
                source="fixture",
                input_hash=input_hash(req),
                judgments=trace.judgments,
                scoring_elapsed_ms=0,
            )
            return replay_run(ReplayRequest(record=record, policy=req.policy))
        raise HTTPException(404, "unknown_fixture")

    @app.post("/v1/runs", response_model=RunView, dependencies=[Depends(authorize)])
    async def runs(body: StudyRequest, request: Request) -> RunView:
        if replay_only:
            raise HTTPException(503, "replay_only_live_scoring_disabled")
        try:
            record = await score_run(body, request.app.state.selector)
            return replay_run(ReplayRequest(record=record, policy=body.request.policy))
        except ProviderError as exc:
            raise HTTPException(504 if exc.code == "deadline_exceeded" else 502, exc.code) from None

    @app.post("/v1/replay", response_model=RunView, dependencies=[Depends(authorize)])
    async def replay(body: ReplayRequest) -> RunView:
        return replay_run(body)

    @app.post("/v1/import", response_model=ImportedRun, dependencies=[Depends(authorize)])
    async def import_run(body: ReplayBundle) -> ImportedRun:
        return ImportedRun(view=replay_run(body), comparison=body.comparison)

    @app.post("/v1/compare", response_model=Comparison, dependencies=[Depends(authorize)])
    async def compare(body: ReplayRequest, request: Request) -> Comparison:
        if request.app.state.generator is None:
            raise HTTPException(503, "generation_not_configured")
        return await compare_run(
            replay_run(body),
            request.app.state.generator,
            generation_price=request.app.state.generation_price,
            selection_price=request.app.state.selection_price,
        )

    @app.post("/v1/select", response_model=SelectionResult, dependencies=[Depends(authorize)])
    async def select(body: SelectRequest, request: Request) -> SelectionResult:
        if replay_only:
            raise HTTPException(503, "replay_only_live_scoring_disabled")
        engine: ContextSelector = request.app.state.selector
        try:
            return await engine.select_request(body)
        except ProviderError as exc:
            status = 504 if exc.code == "deadline_exceeded" else 502
            raise HTTPException(status_code=status, detail=exc.code) from None

    @app.post("/v1/rankings", response_model=RankingReport, dependencies=[Depends(authorize)])
    async def rankings(body: RankingRequest, request: Request) -> RankingReport:
        return compare_rankings(body, selection_price=request.app.state.selection_price)

    @app.post(
        "/v1/compare-rankings", response_model=RankingReport, dependencies=[Depends(authorize)]
    )
    async def ranking_answers(body: RankingRequest, request: Request) -> RankingReport:
        if request.app.state.generator is None:
            raise HTTPException(503, "generation_not_configured")
        return await generate_rankings(
            body,
            request.app.state.generator,
            generation_price=request.app.state.generation_price,
            selection_price=request.app.state.selection_price,
        )

    return app
