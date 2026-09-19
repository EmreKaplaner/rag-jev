"""Portable scored runs and repeatable comparisons. No server-side document storage."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from typing import Literal, Self

import tiktoken
from pydantic import Field, model_validator

from rag_jev.generation import Answer, Generator
from rag_jev.models import (
    PROMPT_VERSION,
    Contract,
    Judgment,
    Policy,
    SelectionResult,
    SelectRequest,
    Usage,
)
from rag_jev.provider import ProviderError
from rag_jev.selector import ContextSelector, apply_policy


def input_hash(request: SelectRequest) -> str:
    payload = request.model_dump(
        include={"query", "retrieval_query", "documents", "relevance_guidance"}
    )
    if request.scoring_strategy != "independent":
        payload["scoring_strategy"] = request.scoring_strategy
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class StudyRequest(Contract):
    request: SelectRequest
    relevant_ids: list[str] | None = None
    baseline_ids: list[str] | None = None

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        known = {d.id for d in self.request.documents}
        for ids in (self.relevant_ids, self.baseline_ids):
            if ids is not None and (len(set(ids)) != len(ids) or not set(ids) <= known):
                raise ValueError("labels and baseline IDs must be unique candidate IDs")
        return self


class ScoredRun(StudyRequest):
    schema_version: Literal[1] = 1
    created_at: datetime
    source: Literal["live", "fixture"]
    prompt_version: str = PROMPT_VERSION
    input_hash: str
    judgments: list[Judgment] = Field(max_length=256)
    scoring_elapsed_ms: float = Field(ge=0, allow_inf_nan=False)
    scoring_usage: Usage | None = None

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        if self.input_hash != input_hash(self.request):
            raise ValueError("run inputs changed; score the edited inputs again")
        if len(self.judgments) != len(self.request.documents):
            raise ValueError("run must contain exactly one judgment per candidate")
        if self.prompt_version != self.request.prompt_version:
            raise ValueError("run prompt version is not supported; rescore inputs")
        if self.request.scoring_strategy == "contextual":
            if (
                self.scoring_usage is None
                or not self.scoring_usage.complete
                or self.scoring_usage.completed_documents != len(self.request.documents)
            ):
                raise ValueError("contextual run requires complete shared usage")
        elif self.scoring_usage is not None:
            raise ValueError("independent run uses per-document usage")
        return self


class ReplayRequest(Contract):
    record: ScoredRun
    policy: Policy

    @model_validator(mode="after")
    def validate_guidance(self) -> Self:
        if self.policy.relevance_guidance != self.record.request.relevance_guidance:
            raise ValueError("changing relevance guidance requires new scoring")
        return self


class RunView(Contract):
    record: ScoredRun
    policy: Policy
    selection: SelectionResult
    baseline_ids: list[str]
    baseline_context_tokens: int
    selected_context_tokens: int
    token_estimator: str = "cl100k_base estimate; actual prompt usage appears after generation"
    evidence_recall: float | None
    relevant_dropped: list[str] | None
    baseline_evidence_recall: float | None


async def score_run(study: StudyRequest, selector: ContextSelector) -> ScoredRun:
    result = await selector.select_request(study.request)
    if result.status == "bypassed":
        raise ProviderError(result.error_code or "scoring_bypassed")
    return ScoredRun(
        **study.model_dump(),
        created_at=datetime.now(UTC),
        source="live",
        prompt_version=result.prompt_version,
        input_hash=input_hash(study.request),
        scoring_elapsed_ms=result.elapsed_ms,
        scoring_usage=result.usage if study.request.scoring_strategy == "contextual" else None,
        judgments=[
            Judgment.model_validate(
                {
                    "relevance": d.relevance,
                    "model": d.model,
                    "input_tokens": d.input_tokens or 0,
                    "output_tokens": d.output_tokens or 0,
                }
            )
            for d in result.decisions
        ],
    )


def replay_run(body: ReplayRequest) -> RunView:
    record = body.record
    request = SelectRequest.model_validate(
        {**record.request.model_dump(), **body.policy.model_dump()}
    )
    result = apply_policy(
        request, record.judgments, elapsed_ms=record.scoring_elapsed_ms, usage=record.scoring_usage
    )
    baseline = (
        record.baseline_ids
        if record.baseline_ids is not None
        else [d.id for d in request.documents]
    )
    selected = result.selected_ids or []
    by_id = {d.id: d for d in request.documents}
    encoding = tiktoken.get_encoding("cl100k_base")

    def count(ids: list[str]) -> int:
        return sum(len(encoding.encode(by_id[i].text, disallowed_special=())) for i in ids)

    gold = record.relevant_ids
    return RunView(
        record=record,
        policy=body.policy,
        selection=result,
        baseline_ids=baseline,
        baseline_context_tokens=count(baseline),
        selected_context_tokens=count(selected),
        evidence_recall=len(set(gold) & set(selected)) / len(gold) if gold else None,
        baseline_evidence_recall=len(set(gold) & set(baseline)) / len(gold) if gold else None,
        relevant_dropped=[i for i in gold if i not in selected] if gold is not None else None,
    )


class Price(Contract):
    input_per_million: float = Field(ge=0, allow_inf_nan=False)
    output_per_million: float = Field(ge=0, allow_inf_nan=False)

    def cost(self, inputs: int | None, outputs: int | None) -> float | None:
        if inputs is None or outputs is None:
            return None
        return (inputs * self.input_per_million + outputs * self.output_per_million) / 1_000_000

    @classmethod
    def from_env(cls, prefix: str) -> Price | None:
        inp, out = (
            os.getenv(prefix + "_INPUT_PER_MILLION"),
            os.getenv(prefix + "_OUTPUT_PER_MILLION"),
        )
        return (
            cls(input_per_million=float(inp), output_per_million=float(out))
            if inp and out
            else None
        )


class Comparison(Contract):
    input_hash: str
    policy_version: str
    baseline: Answer
    selected: Answer
    baseline_ids: list[str]
    selected_ids: list[str]
    selection_source: Literal["live", "fixture"]
    baseline_stage_ms: float
    selected_stage_ms: float
    selection_original_ms: float
    baseline_cost_usd: float | None
    selected_cost_usd: float | None
    selection_cost_usd: float | None
    cost_basis: str = (
        "Estimate from configured per-million rates; excludes retrieval and caching discounts"
    )
    timing_basis: str = (
        "Generation measured now; selected stage adds original scoring time. Retrieval excluded."
    )
    citation_check: str = (
        "Citation IDs checked; factual support and answer correctness are not graded"
    )


class ReplayBundle(ReplayRequest):
    comparison: Comparison | None = None

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        if self.comparison is not None:
            view = replay_run(self)
            c = self.comparison
            if (
                c.input_hash != self.record.input_hash
                or c.policy_version != view.selection.policy_version
                or c.baseline_ids != view.baseline_ids
                or c.selected_ids != view.selection.selected_ids
                or c.selection_source != self.record.source
            ):
                raise ValueError("recorded answers do not match this run and policy")
        return self


class ImportedRun(Contract):
    view: RunView
    comparison: Comparison | None = None


async def compare_run(
    view: RunView,
    generator: Generator,
    *,
    generation_price: Price | None = None,
    selection_price: Price | None = None,
) -> Comparison:
    by_id = {d.id: d for d in view.record.request.documents}
    selected_ids = view.selection.selected_ids or []
    baseline = await generator.generate(
        view.record.request.query, [by_id[i] for i in view.baseline_ids]
    )
    selected = await generator.generate(view.record.request.query, [by_id[i] for i in selected_ids])
    scoring_cost = (
        selection_price.cost(view.selection.usage.input_tokens, view.selection.usage.output_tokens)
        if selection_price and view.record.source == "live"
        else None
    )

    def answer_cost(answer: Answer) -> float | None:
        if answer.status == "no_context":
            return 0
        return (
            generation_price.cost(answer.input_tokens, answer.output_tokens)
            if generation_price
            else None
        )

    base_cost, selected_cost = answer_cost(baseline), answer_cost(selected)
    return Comparison(
        input_hash=view.record.input_hash,
        policy_version=view.selection.policy_version,
        baseline=baseline,
        selected=selected,
        baseline_ids=view.baseline_ids,
        selected_ids=selected_ids,
        selection_source=view.record.source,
        baseline_stage_ms=baseline.elapsed_ms,
        selected_stage_ms=view.record.scoring_elapsed_ms + selected.elapsed_ms,
        selection_original_ms=view.record.scoring_elapsed_ms,
        baseline_cost_usd=base_cost,
        selected_cost_usd=selected_cost + scoring_cost
        if selected_cost is not None and scoring_cost is not None
        else None,
        selection_cost_usd=scoring_cost,
    )
