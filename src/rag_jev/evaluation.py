"""Evaluate selection from stored judgments, or score once and compare all policies.

Retrieval metrics are not answer correctness. An optional application callback can
generate and grade actual answers; the report states when that was not measured.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Awaitable, Callable
from statistics import mean
from typing import Any, Self

from pydantic import Field, model_validator

from rag_jev.models import (
    PROMPT_VERSION,
    Contract,
    Document,
    Judgment,
    Mode,
    Probability,
    ScoringStrategy,
    SelectRequest,
    Usage,
)
from rag_jev.provider import ProviderError
from rag_jev.selector import ContextSelector, apply_policy
from rag_jev.tokens import count_tokens


class EvalCase(Contract):
    id: str
    query: str
    documents: list[Document]
    relevant_ids: list[str]
    reference_answer: str | None = None
    baseline_ids: list[str] | None = None
    retrieval_query: str | None = None
    family_id: str | None = None

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        SelectRequest(query=self.query, documents=self.documents, mode="rerank")
        if not set(self.relevant_ids) <= {d.id for d in self.documents}:
            raise ValueError("relevant_ids must refer to documents in the candidate set")
        if len(self.relevant_ids) != len(set(self.relevant_ids)):
            raise ValueError("relevant_ids must be unique")
        if self.baseline_ids is not None and (
            len(set(self.baseline_ids)) != len(self.baseline_ids)
            or not set(self.baseline_ids) <= {d.id for d in self.documents}
        ):
            raise ValueError("baseline_ids must be unique candidate IDs")
        if self.retrieval_query is not None and not self.retrieval_query.strip():
            raise ValueError("retrieval_query cannot be blank")
        return self


def fingerprint(case: EvalCase) -> str:
    # Labels are deliberately excluded: no ground-truth relevance reaches the provider.
    value = {
        "query": case.query,
        "documents": [
            d.model_dump(
                exclude={"group_id", "pinned"} if d.group_id is None and not d.pinned else set()
            )
            for d in case.documents
        ],
    }
    if case.retrieval_query is not None:
        value["retrieval_query"] = case.retrieval_query
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Trace(Contract):
    case_id: str
    input_hash: str
    prompt_version: str = PROMPT_VERSION
    relevance_guidance: str | None = None
    judgments: list[Judgment]
    elapsed_ms: float = Field(ge=0)
    source: str = "live"
    scoring_strategy: ScoringStrategy = "independent"
    scoring_usage: Usage | None = None

    @model_validator(mode="after")
    def validate_usage(self) -> Self:
        if self.scoring_strategy == "contextual":
            if (
                self.scoring_usage is None
                or not self.scoring_usage.complete
                or self.scoring_usage.completed_documents != len(self.judgments)
            ):
                raise ValueError("contextual trace requires complete shared usage")
        elif self.scoring_usage is not None:
            raise ValueError("independent trace uses per-passage usage")
        return self


AnswerEvaluator = Callable[[EvalCase, list[Document]], Awaitable[bool]]


async def collect_trace(
    case: EvalCase,
    selector: ContextSelector,
    guidance: str | None = None,
    scoring_strategy: ScoringStrategy = "independent",
) -> Trace:
    result = await selector.select(
        query=case.query,
        documents=case.documents,
        mode="rerank",
        relevance_guidance=guidance,
        scoring_strategy=scoring_strategy,
        retrieval_query=case.retrieval_query,
    )
    if result.status == "bypassed":
        raise ProviderError(result.error_code or "evaluation_bypassed")
    judgments = []
    for decision in result.decisions:
        if any(
            x is None
            for x in (
                decision.relevance,
                decision.model,
            )
        ):
            raise ValueError("incomplete evaluation judgment")
        judgments.append(
            Judgment.model_validate(
                {
                    "relevance": decision.relevance,
                    "model": decision.model,
                    "input_tokens": decision.input_tokens or 0,
                    "output_tokens": decision.output_tokens or 0,
                }
            )
        )
    return Trace(
        case_id=case.id,
        input_hash=fingerprint(case),
        relevance_guidance=guidance,
        judgments=judgments,
        elapsed_ms=result.elapsed_ms,
        prompt_version=result.prompt_version,
        scoring_strategy=scoring_strategy,
        scoring_usage=result.usage if scoring_strategy == "contextual" else None,
    )


class Metrics(Contract):
    precision: float
    evidence_recall: float | None
    ndcg: float | None
    relevant_dropped: int
    documents: int
    characters: int
    context_tokens: int
    all_evidence_lost: bool
    correctly_empty: bool | None
    answer_correct: bool | None = None


def measure(case: EvalCase, selected: list[Document], cutoff: int | None = None) -> Metrics:
    gold = set(case.relevant_ids)
    hits = sum(d.id in gold for d in selected)
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(selected) if d.id in gold)
    # Use a fixed cutoff across policies. Returning fewer documents must not shrink the ideal.
    k = cutoff if cutoff is not None else len(case.documents)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return Metrics(
        precision=hits / len(selected) if selected else 0,
        evidence_recall=hits / len(gold) if gold else None,
        ndcg=dcg / ideal if ideal else (0 if gold else None),
        relevant_dropped=len(gold) - hits,
        documents=len(selected),
        characters=sum(len(d.text) for d in selected),
        context_tokens=sum(count_tokens(d.text) for d in selected),
        all_evidence_lost=bool(gold) and hits == 0,
        correctly_empty=not selected if not gold else None,
    )


async def evaluate(
    cases: list[EvalCase],
    traces: list[Trace],
    *,
    min_relevance: Probability,
    top_n: int | None = None,
    answer_evaluator: AnswerEvaluator | None = None,
) -> dict[str, Any]:
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("evaluation requires nonempty cases with unique IDs")
    if len({t.case_id for t in traces}) != len(traces):
        raise ValueError("duplicate trace IDs")
    by_id = {t.case_id: t for t in traces}
    if set(by_id) != {c.id for c in cases}:
        raise ValueError("trace IDs must exactly match evaluation case IDs")
    policies: list[Mode] = ["filter", "rerank", "filter_and_rerank"]
    rows: list[dict[str, Any]] = []
    for case in cases:
        trace = by_id[case.id]
        expected_prompt = SelectRequest(
            query=case.query, documents=[], mode="rerank", scoring_strategy=trace.scoring_strategy
        ).prompt_version
        if trace.input_hash != fingerprint(case) or trace.prompt_version != expected_prompt:
            raise ValueError(f"stale or mismatched trace for case {case.id}")
        by_document_id = {d.id: d for d in case.documents}
        selections = {
            "baseline": [by_document_id[i] for i in case.baseline_ids]
            if case.baseline_ids is not None
            else case.documents[:top_n]
        }
        for mode in policies:
            request = SelectRequest(
                query=case.query,
                documents=case.documents,
                mode=mode,
                min_relevance=None if mode == "rerank" else min_relevance,
                top_n=top_n,
                relevance_guidance=trace.relevance_guidance,
                scoring_strategy=trace.scoring_strategy,
            )
            selections[mode] = apply_policy(
                request, trace.judgments, usage=trace.scoring_usage
            ).documents
        metrics = {}
        for name, documents in selections.items():
            item = measure(case, documents, cutoff=top_n)
            if answer_evaluator:
                correct = await answer_evaluator(case, documents)
                if not isinstance(correct, bool):
                    raise ValueError("answer_evaluator must return a boolean")
                item = item.model_copy(update={"answer_correct": correct})
            metrics[name] = item.model_dump()
        rows.append({"case_id": case.id, "metrics": metrics, "elapsed_ms": trace.elapsed_ms})
    summary = {}
    for name in ("baseline", *policies):
        group = [r["metrics"][name] for r in rows]
        summary[name] = {
            metric: mean(v[metric] for v in group if v[metric] is not None)
            if any(v[metric] is not None for v in group)
            else None
            for metric in Metrics.model_fields
        }
    elapsed = sorted(t.elapsed_ms for t in traces)
    return {
        "cases": len(cases),
        "sources": sorted({t.source for t in traces}),
        "models": sorted({j.model for t in traces for j in t.judgments}),
        "answer_quality": "measured_by_callback" if answer_evaluator else "not_evaluated",
        "context_size_unit": "characters (not model tokens)",
        "recall_scope": "relevant evidence already present in retrieved candidates",
        "ndcg_cutoff": top_n or "candidate count per query; shorter outputs padded with zeros",
        "usage": {
            "input_tokens": sum(
                t.scoring_usage.input_tokens
                if t.scoring_usage
                else sum(j.input_tokens for j in t.judgments)
                for t in traces
            ),
            "output_tokens": sum(
                t.scoring_usage.output_tokens
                if t.scoring_usage
                else sum(j.output_tokens for j in t.judgments)
                for t in traces
            ),
            "note": "scoring performed once; policy comparisons reuse judgments",
        },
        "scoring_p95_ms": elapsed[math.ceil(0.95 * len(elapsed)) - 1],
        "summary": summary,
        "rows": rows,
    }
