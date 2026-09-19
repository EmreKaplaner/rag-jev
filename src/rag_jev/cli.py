from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError
from typesafe_sdk import TypeSafeError

from rag_jev.demo import CASES, FixtureScorer, fixture_traces
from rag_jev.evaluation import EvalCase, Trace, collect_trace, evaluate
from rag_jev.models import SelectRequest
from rag_jev.provider import Jev, ProviderError
from rag_jev.selector import ContextSelector


def emit(value: Any, path: str | None = None) -> None:
    output = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path:
        Path(path).write_text(output)
    else:
        sys.stdout.write(output)


def live_selector(provider: Jev, args: argparse.Namespace) -> ContextSelector:
    return ContextSelector(
        provider,
        timeout_ms=args.timeout_ms,
        max_concurrency=args.max_concurrency,
        on_error="raise",
    )


async def run(args: argparse.Namespace) -> None:
    if args.command == "compare-rankings":
        from rag_jev.comparison import RankingRequest, compare_rankings, generate_rankings
        from rag_jev.generation import Generator
        from rag_jev.workbench import Price

        raw = json.loads(await asyncio.to_thread(Path(args.input).read_text))
        body = RankingRequest(
            record=raw.get("record", raw),
            top_n=args.top_n,
            max_context_tokens=args.max_context_tokens,
        )
        price = Price.from_env("RAG_JEV_SCORING")
        if args.generate:
            generator = Generator.from_env()
            if generator is None:
                raise ValueError("generation not configured")
            try:
                ranking_report = await generate_rankings(
                    body,
                    generator,
                    selection_price=price,
                    generation_price=Price.from_env("RAG_JEV_GENERATION"),
                )
            finally:
                await generator.aclose()
        else:
            ranking_report = compare_rankings(body, selection_price=price)
        emit(ranking_report.model_dump(mode="json"), args.output)
        return
    if args.command == "review-results":
        from rag_jev.review import summarize_review

        emit(
            await asyncio.to_thread(summarize_review, args.packet, args.labels, args.key),
            args.output,
        )
        return
    if args.command == "calibrate":
        from rag_jev.calibration import run_calibration

        await run_calibration(args)
        return
    if args.command == "demo":
        selector = ContextSelector(FixtureScorer())
        case = CASES[0]
        selected = await selector.select(
            query=case.query, documents=case.documents, min_relevance=0.2
        )
        report = await evaluate(CASES, fixture_traces(), min_relevance=0.2)
        emit(
            {
                "notice": "OFFLINE FIXTURE DEMO: scores are hand-authored, not Jev predictions",
                "selection": selected.model_dump(),
                "evaluation": report,
            }
        )
        return
    if args.command == "select":
        raw = await asyncio.to_thread(
            sys.stdin.read if args.input == "-" else Path(args.input).read_text
        )
        request = SelectRequest.model_validate_json(raw)
        async with Jev(model=args.model) as provider:
            result = await live_selector(provider, args).select_request(request)
        emit(result.model_dump(), args.output)
        return
    if args.command == "eval":
        raw = await asyncio.to_thread(Path(args.input).read_text)
        cases = [EvalCase.model_validate(x) for x in json.loads(raw)]
        if args.replay:
            raw = await asyncio.to_thread(Path(args.replay).read_text)
            traces = [Trace.model_validate(x) for x in json.loads(raw)]
        else:
            async with Jev(model=args.model) as provider:
                selector = live_selector(provider, args)
                traces = [
                    await collect_trace(case, selector, scoring_strategy=args.scoring_strategy)
                    for case in cases
                ]
            if args.save_traces:
                emit([t.model_dump() for t in traces], args.save_traces)
        if any(t.scoring_strategy != args.scoring_strategy for t in traces):
            raise ValueError("Replay strategy differs; pass the matching --scoring-strategy")
        report = await evaluate(cases, traces, min_relevance=args.min_relevance, top_n=args.top_n)
        emit(report, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select RAG context with TypeSafe Jev")
    parser.add_argument("--env-file", default=".env", help="Loaded without overriding environment")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Offline, hand-authored fixture demonstration")
    commands.add_parser("schema", help="Print the service OpenAPI schema")
    ranking = commands.add_parser(
        "compare-rankings", help="Compare original, Jev, and fusion from a scored replay"
    )
    ranking.add_argument("input", help="Exported replay bundle, RunView, or ScoredRun JSON")
    ranking.add_argument("--top-n", type=int, default=10)
    ranking.add_argument("--max-context-tokens", type=int)
    ranking.add_argument(
        "--generate", action="store_true", help="Make up to three generation calls"
    )
    ranking.add_argument("--output")
    review = commands.add_parser(
        "review-results", help="Validate supplied human labels and report completeness"
    )
    review.add_argument("packet")
    review.add_argument("labels")
    review.add_argument("--key", help="Unblind only after independent review")
    review.add_argument("--output")
    serve = commands.add_parser("serve", help="Start the real Jev selection service")
    serve.add_argument("--host", default=None, help="Defaults to RAG_JEV_HOST or 127.0.0.1")
    serve.add_argument("--port", type=int, default=None, help="Defaults to PORT or 8000")
    serve.add_argument(
        "--replay-only",
        action="store_true",
        help="Browse saved runs without keys or inference calls",
    )
    select = commands.add_parser("select", help="Select using a JSON request file, or - for stdin")
    select.add_argument("input")
    evaluation = commands.add_parser(
        "eval", help="Compare policies on labeled retrieved candidates"
    )
    evaluation.add_argument("input", help="JSON array of evaluation cases")
    evaluation.add_argument("--replay", help="Use saved traces; no API requests")
    evaluation.add_argument("--save-traces", help="Save live judgments for offline policy tuning")
    evaluation.add_argument("--min-relevance", type=float, required=True)
    evaluation.add_argument("--top-n", type=int)
    evaluation.add_argument(
        "--scoring-strategy", choices=["independent", "contextual"], default="independent"
    )
    calibration = commands.add_parser(
        "calibrate",
        help="Choose a policy on development questions, then assess a separate held-out set",
    )
    calibration.add_argument("development")
    calibration.add_argument("evaluation")
    calibration.add_argument("--replay", help="Saved traces covering both splits; no API calls")
    calibration.add_argument(
        "--scoring-strategy", choices=["independent", "contextual"], default="contextual"
    )
    calibration.add_argument("--cutoffs", type=float, nargs="+", default=[0.2, 0.35, 0.5])
    calibration.add_argument("--top-ns", type=int, nargs="+", default=[3, 5, 8])
    calibration.add_argument("--max-context-tokens", type=int)
    calibration.add_argument("--min-recall", type=float, default=0.98)
    for command in (select, evaluation, calibration):
        command.add_argument("--model")
        command.add_argument("--timeout-ms", type=float)
        command.add_argument("--max-concurrency", type=int)
        command.add_argument("--output", required=command is calibration)
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    try:
        if args.command in ("select", "eval", "calibrate"):
            args.model = args.model or os.getenv("RAG_JEV_MODEL", "jev-latest")
            if args.timeout_ms is None:
                args.timeout_ms = float(os.getenv("RAG_JEV_TIMEOUT_MS", "5000"))
            if args.max_concurrency is None:
                args.max_concurrency = int(os.getenv("RAG_JEV_MAX_CONCURRENCY", "8"))
        if args.command == "serve":
            import uvicorn

            from rag_jev.server import create_app

            args.host = args.host or os.getenv("RAG_JEV_HOST", "127.0.0.1")
            args.port = args.port if args.port is not None else int(os.getenv("PORT", "8000"))
            if not 1 <= args.port <= 65535:
                parser.error("port must be between 1 and 65535")

            if not args.replay_only and not os.getenv("TYPESAFE_API_KEY", "").strip():
                parser.error("set TYPESAFE_API_KEY in the environment or .env before serving")
            if args.host not in ("127.0.0.1", "localhost", "::1") and not os.getenv(
                "RAG_JEV_API_TOKEN"
            ):
                parser.error("set RAG_JEV_API_TOKEN before binding outside localhost")
            uvicorn.run(
                create_app(replay_only=args.replay_only),
                host=args.host,
                port=args.port,
                access_log=False,
            )
        elif args.command == "schema":
            from rag_jev.server import create_app

            emit(create_app().openapi())
        else:
            asyncio.run(run(args))
    except ValidationError as exc:
        emit(
            {
                "error": "invalid_input",
                "issues": [
                    {"loc": list(e["loc"]), "message": e["msg"]}
                    for e in exc.errors(include_input=False)
                ],
            }
        )
        raise SystemExit(2) from None
    except ProviderError as exc:
        print(f"Jev request failed: {exc.code}", file=sys.stderr)
        raise SystemExit(1) from None
    except TypeSafeError:
        print("TypeSafe client failed; check API key and configuration.", file=sys.stderr)
        raise SystemExit(1) from None
    except (ValueError, OSError) as exc:
        print(f"Invalid input or configuration ({type(exc).__name__}).", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
