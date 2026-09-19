"""Three-way comparisons from one scored run; scoring is never repeated here."""

from math import log2
from time import perf_counter
from typing import Literal

from pydantic import Field

from rag_jev.generation import Answer, Generator
from rag_jev.models import Contract, Policy, SelectionResult
from rag_jev.workbench import Price, ReplayRequest, ScoredRun, replay_run

Arm = Literal["original", "jev", "fusion"]


class RankingRequest(Contract):
    record: ScoredRun
    top_n: int = Field(default=10, ge=1, le=256, strict=True)
    max_context_tokens: int | None = Field(default=None, ge=1, le=1_000_000, strict=True)


class RankingArm(Contract):
    name: Arm
    selection: SelectionResult
    evidence_recall: float | None
    ndcg: float | None
    policy_elapsed_ms: float
    stage_elapsed_ms: float
    selector_cost_usd: float | None
    total_cost_usd: float | None
    answer: Answer | None = None
    answer_reused_from: Arm | None = None


class RankingReport(Contract):
    input_hash: str
    source: Literal["live", "fixture"]
    top_n: int
    arms: list[RankingArm]
    scoring_calls: int = 0
    generation_calls: int = 0
    quality_basis: str = (
        "Binary relevance labels supplied by the caller; candidate evidence recall and nDCG "
        "at the shared top-N cutoff (short outputs padded with zero gain). "
        "Answer correctness and factual support require review."
    )
    cost_basis: str = (
        "Configured API rates: Jev plus generation per deployed arm. Unknown rates/usage stay "
        "null. Retrieval, local compute, cache discounts excluded. Shared Jev scoring paid "
        "once in this comparison, not once per arm."
    )
    timing_basis: str = (
        "Recorded Jev time plus measured policy and generation time; retrieval excluded. "
        "This replay is not a fresh latency benchmark. Identical contexts reuse one answer."
    )
    selection_basis: str = (
        "Original input order versus Jev ordering versus equal-weight RRF (k=60), same "
        "candidate set and top-N/token limits. Pins/groups preserved in all arms. "
        "Recorded baseline_ids are not used; original means input-order selection here."
    )


def compare_rankings(
    body: RankingRequest, *, selection_price: Price | None = None
) -> RankingReport:
    rows = []
    record = body.record
    for name in ("original", "jev", "fusion"):
        policy = Policy(
            mode="filter" if name == "original" else "rerank" if name == "jev" else "fusion",
            min_relevance=0 if name == "original" else None,
            top_n=body.top_n,
            max_context_tokens=body.max_context_tokens,
            relevance_guidance=record.request.relevance_guidance,
        )
        started = perf_counter()
        view = replay_run(ReplayRequest(record=record, policy=policy))
        elapsed = (perf_counter() - started) * 1000
        selection = view.selection
        gold = set(record.relevant_ids or [])
        ids = selection.selected_ids or []
        ideal = sum(1 / log2(i + 2) for i in range(min(len(gold), body.top_n)))
        dcg = sum(1 / log2(i + 2) for i, doc in enumerate(ids[: body.top_n]) if doc in gold)
        cost = (
            0.0
            if name == "original"
            else selection_price.cost(selection.usage.input_tokens, selection.usage.output_tokens)
            if selection_price and record.source == "live"
            else None
        )
        rows.append(
            RankingArm(
                name=name,
                selection=selection,
                evidence_recall=view.evidence_recall,
                ndcg=(dcg / ideal if ideal else 0.0) if gold else None,
                policy_elapsed_ms=elapsed,
                stage_elapsed_ms=elapsed + (record.scoring_elapsed_ms if name != "original" else 0),
                selector_cost_usd=cost,
                total_cost_usd=cost,
            )
        )
    return RankingReport(
        input_hash=record.input_hash, source=record.source, top_n=body.top_n, arms=rows
    )


async def generate_rankings(
    body: RankingRequest,
    generator: Generator,
    *,
    generation_price: Price | None = None,
    selection_price: Price | None = None,
) -> RankingReport:
    report = compare_rankings(body, selection_price=selection_price)
    cache: dict[tuple[str, ...], tuple[Arm, Answer]] = {}
    rows = []
    calls = 0
    for arm in report.arms:
        key = tuple(arm.selection.selected_ids or [])
        reused = cache.get(key)
        if reused:
            source, answer = reused
        else:
            source = None
            answer = await generator.generate(body.record.request.query, arm.selection.documents)
            calls += bool(key)
            cache[key] = (arm.name, answer)
        cost = (
            0.0
            if answer.status == "no_context"
            else generation_price.cost(answer.input_tokens, answer.output_tokens)
            if generation_price
            else None
        )
        rows.append(
            arm.model_copy(
                update={
                    "answer": answer,
                    "answer_reused_from": source,
                    "stage_elapsed_ms": arm.stage_elapsed_ms + answer.elapsed_ms,
                    "total_cost_usd": cost + arm.selector_cost_usd
                    if cost is not None and arm.selector_cost_usd is not None
                    else None,
                }
            )
        )
    return report.model_copy(update={"arms": rows, "generation_calls": calls})
