"""Development-only policy selection with a frozen, separate held-out assessment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from rag_jev.evaluation import EvalCase, Trace, collect_trace, fingerprint, measure
from rag_jev.models import Policy, ScoringStrategy, SelectRequest
from rag_jev.provider import Jev
from rag_jev.selector import ContextSelector, apply_policy


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_splits(development: list[EvalCase], evaluation: list[EvalCase]) -> None:
    if len(development) < 2 or len(evaluation) < 2:
        raise ValueError("Each split needs at least two questions")
    ids: set[str] = set()
    questions: set[str] = set()
    for case in development + evaluation:
        question = re.sub(r"\W+", " ", case.query.casefold()).strip()
        if not question or case.id in ids or question in questions:
            raise ValueError("Duplicate case ID or normalized question across calibration inputs")
        ids.add(case.id)
        questions.add(question)
    if {c.family_id for c in development if c.family_id} & {
        c.family_id for c in evaluation if c.family_id
    }:
        raise ValueError("Related question families cross development/evaluation splits")
    if not any(c.relevant_ids for c in development):
        raise ValueError("Development requires labeled relevant evidence for recall calibration")


def candidate_policies(
    cutoffs: Sequence[float], top_ns: Sequence[int], budget: int | None
) -> list[Policy]:
    policies = [
        Policy(mode=mode, min_relevance=cutoff, max_context_tokens=budget)
        for mode in ("filter", "filter_and_rerank")
        for cutoff in sorted(set(cutoffs))
    ]
    policies += [
        Policy(mode="rerank", top_n=k, max_context_tokens=budget) for k in sorted(set(top_ns))
    ]
    if not policies:
        raise ValueError("At least one candidate policy is required")
    return policies


def assess(
    cases: list[EvalCase], traces: list[Trace], policy: Policy | None, strategy: ScoringStrategy
) -> dict[str, Any]:
    by_id = {t.case_id: t for t in traces}
    if len(by_id) != len(traces) or set(by_id) != {c.id for c in cases}:
        raise ValueError("Trace IDs must match this split exactly")
    rows = []
    for case in cases:
        trace = by_id[case.id]
        request = SelectRequest(
            query=case.query,
            retrieval_query=case.retrieval_query,
            documents=case.documents,
            scoring_strategy=strategy,
            **(policy or Policy(mode="rerank")).model_dump(),
        )
        if (
            trace.input_hash != fingerprint(case)
            or trace.scoring_strategy != strategy
            or trace.prompt_version != request.prompt_version
            or trace.relevance_guidance is not None
        ):
            raise ValueError("Trace inputs, strategy, prompt or guidance do not match calibration")
        if len(trace.judgments) != len(case.documents):
            raise ValueError("Incomplete trace judgments")
        result = apply_policy(request, trace.judgments, usage=trace.scoring_usage)
        by_doc = {d.id: d for d in case.documents}
        selected = (
            result.documents
            if policy
            else [by_doc[i] for i in case.baseline_ids]
            if case.baseline_ids is not None
            else case.documents
        )
        row = measure(case, selected).model_dump()
        row.update(
            {
                "case_id": case.id,
                "selected_ids": [d.id for d in selected],
                "budget_exceeded": result.context_token_budget_exceeded if policy else None,
            }
        )
        rows.append(row)
    keys = [
        "evidence_recall",
        "precision",
        "ndcg",
        "context_tokens",
        "documents",
        "correctly_empty",
        "all_evidence_lost",
    ]
    summary = {}
    for key in keys:
        values = [float(r[key]) for r in rows if r[key] is not None]
        summary[key] = mean(values) if values else None
    return {"summary": summary, "rows": rows}


def choose(
    cases: list[EvalCase],
    traces: list[Trace],
    policies: list[Policy],
    strategy: ScoringStrategy,
    min_recall: float,
) -> dict[str, Any]:
    if not math.isfinite(min_recall) or not 0 <= min_recall <= 1:
        raise ValueError("min_recall must be finite and between zero and one")
    results = {p.version: assess(cases, traces, p, strategy) for p in policies}
    eligible = [
        p
        for p in policies
        if results[p.version]["summary"]["evidence_recall"] is not None
        and results[p.version]["summary"]["evidence_recall"] >= min_recall
    ]
    if not eligible:
        return {
            "eligible": False,
            "policy": None,
            "development": results,
            "reason": "No development policy meets the requested observed evidence recall",
        }
    selected = min(
        eligible,
        key=lambda p: (
            results[p.version]["summary"]["context_tokens"],
            -(results[p.version]["summary"]["ndcg"] or 0),
            p.version,
        ),
    )
    return {"eligible": True, "policy": selected.model_dump(), "development": results}


def paired_interval(
    baseline: list[float], selected: list[float], families: list[str] | None = None
) -> dict[str, Any]:
    differences = [b - a for a, b in zip(baseline, selected, strict=True)]
    grouped: dict[str, list[float]] = defaultdict(list)
    for family, difference in zip(
        families or [str(i) for i in range(len(differences))], differences, strict=True
    ):
        grouped[family].append(difference)
    groups = list(grouped.values())
    if len(groups) < 2:
        return {
            "delta": mean(differences),
            "ci95": None,
            "questions": len(differences),
            "independent_units": len(groups),
            "scope": "Too few independent families for an interval",
        }
    rng = random.Random(20260919)
    draws = sorted(
        mean(v for group in rng.choices(groups, k=len(groups)) for v in group) for _ in range(2000)
    )
    return {
        "delta": mean(differences),
        "ci95": [draws[49], draws[1949]],
        "questions": len(differences),
        "independent_units": len(groups),
        "scope": "Descriptive paired bootstrap of supplied families (else questions); "
        "query-weighted mean, 2,000 draws",
    }


def held_out(
    cases: list[EvalCase],
    traces: list[Trace],
    policies: list[Policy],
    strategy: ScoringStrategy,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    baseline = assess(cases, traces, None, strategy)
    results = {
        p.version: {"policy": p.model_dump(), **assess(cases, traces, p, strategy)}
        for p in policies
    }
    report: dict[str, Any] = {
        "baseline": baseline,
        "policies": results,
        "frozen_policy": frozen["policy"],
        "eligible": frozen["eligible"],
        "answer_quality": "not_evaluated",
        "selection_rule": "Minimize development context tokens subject to observed recall; "
        "ties higher NDCG then policy ID",
        "scope": "Evidence already in candidate sets. No semantic accuracy or total-cost "
        "claim. Held-out policies are descriptive; never reselect from this table.",
    }
    if frozen["eligible"]:
        policy = Policy.model_validate(frozen["policy"])
        selected = results[policy.version]
        report["frozen_policy_held_out"] = selected
        report["context_token_difference"] = paired_interval(
            [r["context_tokens"] for r in baseline["rows"]],
            [r["context_tokens"] for r in selected["rows"]],
            [
                "family:" + c.family_id if c.family_id is not None else "question:" + c.id
                for c in cases
            ],
        )
    return report


async def run_calibration(args: argparse.Namespace) -> None:
    if not math.isfinite(args.min_recall) or not 0 <= args.min_recall <= 1:
        raise ValueError("min_recall must be finite and between zero and one")
    development = [
        EvalCase.model_validate(r)
        for r in json.loads(await asyncio.to_thread(Path(args.development).read_text))
    ]
    evaluation = [
        EvalCase.model_validate(r)
        for r in json.loads(await asyncio.to_thread(Path(args.evaluation).read_text))
    ]
    validate_splits(development, evaluation)
    policies = candidate_policies(args.cutoffs, args.top_ns, args.max_context_tokens)
    output = Path(args.output)
    frozen_path = output.with_suffix(".frozen.json")
    config_path = output.with_suffix(".policy.json")
    traces_path = output.with_suffix(".traces.json")
    if any(p.exists() for p in [output, frozen_path, config_path, traces_path]):
        raise ValueError(
            "Use a fresh output path; previous calibration artifacts are never overwritten"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    cached = (
        [
            Trace.model_validate(r)
            for r in json.loads(await asyncio.to_thread(Path(args.replay).read_text))
        ]
        if args.replay
        else []
    )
    all_ids = {c.id for c in development + evaluation}
    if args.replay and (
        len({t.case_id for t in cached}) != len(cached) or {t.case_id for t in cached} != all_ids
    ):
        raise ValueError("Replay traces must cover both splits exactly")

    def save(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    collected: list[Trace] = []

    async def traces(cases: list[EvalCase], selector: ContextSelector | None) -> list[Trace]:
        if args.replay:
            ids = {c.id for c in cases}
            return [t for t in cached if t.case_id in ids]
        assert selector is not None
        subset = []
        for case in cases:
            trace = await collect_trace(case, selector, scoring_strategy=args.scoring_strategy)
            subset.append(trace)
            collected.append(trace)
            save(traces_path, [t.model_dump(mode="json") for t in collected])
        return subset

    async def execute(selector: ContextSelector | None) -> None:
        dev_traces = await traces(development, selector)
        frozen = choose(development, dev_traces, policies, args.scoring_strategy, args.min_recall)
        frozen.update(
            {
                "scoring_strategy": args.scoring_strategy,
                "min_recall": args.min_recall,
                "development_hash": digest([c.model_dump() for c in development]),
                "evaluation_hash": digest([c.model_dump() for c in evaluation]),
                "candidate_policies": [p.model_dump() for p in policies],
                "sources": sorted({t.source for t in dev_traces}),
                "models": sorted({j.model for t in dev_traces for j in t.judgments}),
            }
        )
        # Persist the choice before inspecting held-out metrics or making held-out calls.
        save(frozen_path, frozen)
        eval_traces = await traces(evaluation, selector)
        report = held_out(evaluation, eval_traces, policies, args.scoring_strategy, frozen)
        report.update(
            {
                "freeze_hash": digest(frozen),
                "development_questions": len(development),
                "evaluation_questions": len(evaluation),
                "minimum_development_recall": args.min_recall,
                "sources": sorted({t.source for t in dev_traces + eval_traces}),
                "models": sorted({j.model for t in dev_traces + eval_traces for j in t.judgments}),
                "scoring_usage": {
                    "input_tokens": sum(
                        t.scoring_usage.input_tokens
                        if t.scoring_usage
                        else sum(j.input_tokens for j in t.judgments)
                        for t in dev_traces + eval_traces
                    ),
                    "output_tokens": sum(
                        t.scoring_usage.output_tokens
                        if t.scoring_usage
                        else sum(j.output_tokens for j in t.judgments)
                        for t in dev_traces + eval_traces
                    ),
                },
                "data_warning": "Artifacts include query and passage text; "
                "keep private inputs local.",
            }
        )
        save(output, report)
        if frozen["eligible"]:
            save(
                config_path,
                {**frozen["policy"], "scoring_strategy": args.scoring_strategy, "shadow": True},
            )
        print(
            json.dumps(
                {
                    "report": str(output),
                    "frozen": str(frozen_path),
                    "policy": str(config_path) if frozen["eligible"] else None,
                    "eligible": frozen["eligible"],
                    "answer_quality": "not_evaluated",
                }
            )
        )

    if args.replay:
        await execute(None)
    else:
        async with Jev(model=args.model) as provider:
            await execute(
                ContextSelector(
                    provider,
                    timeout_ms=args.timeout_ms,
                    max_concurrency=args.max_concurrency,
                    on_error="raise",
                )
            )
