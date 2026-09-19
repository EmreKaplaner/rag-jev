from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any, Literal, Required, TypedDict, Unpack

from rag_jev.fusion import ranking_signals
from rag_jev.models import (
    Decision,
    Document,
    Judgment,
    Mode,
    ScoringStrategy,
    SelectionResult,
    SelectRequest,
    Usage,
)
from rag_jev.provider import BatchScorer, ProviderError, Scorer
from rag_jev.tokens import count_tokens


class SelectOptions(TypedDict, total=False):
    scoring_strategy: ScoringStrategy
    query: Required[str]
    documents: Required[Sequence[Document | Mapping[str, Any]]]
    mode: Mode
    min_relevance: float | None
    top_n: int | None
    max_context_tokens: int | None
    shadow: bool
    relevance_guidance: str | None
    retrieval_query: str | None


def apply_policy(
    request: SelectRequest,
    judgments: list[Judgment],
    *,
    elapsed_ms: float = 0,
    usage: Usage | None = None,
) -> SelectionResult:
    """Pure selection, reusable in evaluation without paying for new judgments."""
    if len(judgments) != len(request.documents):
        raise ValueError("expected exactly one judgment per document")
    if request.scoring_strategy == "contextual" and usage is None:
        raise ValueError("contextual judgments require shared usage")
    jev_ranks, fused_scores = ranking_signals([j.relevance for j in judgments])
    direct = {
        i
        for i, j in enumerate(judgments)
        if request.mode in {"rerank", "fusion"} or j.relevance >= request.min_relevance  # type: ignore[operator]
    }
    groups: dict[tuple[str, str], list[int]] = {}
    for i, document in enumerate(request.documents):
        key = ("group", document.group_id) if document.group_id else ("document", document.id)
        groups.setdefault(key, []).append(i)
    units = list(groups.values())
    pinned = {i for unit in units if any(request.documents[j].pinned for j in unit) for i in unit}
    eligible_set = {
        i for unit in units if any(j in direct or j in pinned for j in unit) for i in unit
    }
    if request.mode == "fusion":
        units.sort(key=lambda unit: -max(fused_scores[i] for i in unit))
    elif request.mode != "filter":
        units.sort(key=lambda unit: -max(judgments[i].relevance for i in unit))
    chosen_set = set(pinned)
    tokens = [count_tokens(d.text) for d in request.documents]
    selected_tokens = sum(tokens[i] for i in pinned)
    budget_rejected: set[int] = set()
    for unit in units:
        if not any(i in eligible_set for i in unit) or all(i in chosen_set for i in unit):
            continue
        if request.top_n is None or len(chosen_set) < request.top_n:
            unit_tokens = sum(tokens[i] for i in unit)
            if (
                request.max_context_tokens is not None
                and selected_tokens + unit_tokens > request.max_context_tokens
            ):
                budget_rejected.update(unit)
                continue
            chosen_set.update(unit)
            selected_tokens += unit_tokens
    chosen = [i for unit in units for i in unit if i in chosen_set]
    if request.mode == "filter":
        chosen.sort()
    selected = [request.documents[i] for i in chosen]
    returned = request.documents if request.shadow else selected
    return SelectionResult(
        documents=returned,
        selected_ids=[d.id for d in selected],
        decisions=[
            Decision(
                id=d.id,
                input_index=i,
                relevance=judgments[i].relevance,
                model=judgments[i].model,
                input_tokens=judgments[i].input_tokens if usage is None else None,
                output_tokens=judgments[i].output_tokens if usage is None else None,
                selected=i in chosen_set,
                returned=request.shadow or i in chosen_set,
                original_rank=i + 1,
                jev_rank=jev_ranks[i],
                fusion_score=fused_scores[i] if request.mode == "fusion" else None,
                reason=(
                    "pinned"
                    if i in chosen_set and d.pinned
                    else "group_retained"
                    if i in chosen_set and i not in direct
                    else "retained"
                    if i in chosen_set
                    else "beyond_token_budget"
                    if i in budget_rejected
                    else "beyond_top_n"
                    if i in eligible_set
                    else "below_threshold"
                ),
            )
            for i, d in enumerate(request.documents)
        ],
        status="shadow" if request.shadow else "applied",
        prompt_version=request.prompt_version,
        models=sorted({j.model for j in judgments}),
        policy_version=request.policy.version,
        usage=usage
        if usage is not None
        else Usage(
            input_tokens=sum(j.input_tokens for j in judgments),
            output_tokens=sum(j.output_tokens for j in judgments),
            completed_documents=len(judgments),
        ),
        elapsed_ms=round(elapsed_ms, 3),
        input_characters=sum(len(d.text) for d in request.documents),
        selected_characters=sum(len(d.text) for d in selected),
        returned_characters=sum(len(d.text) for d in returned),
        outcome="context_selected" if selected else "no_context_selected",
        top_n_exceeded=request.top_n is not None and len(selected) > request.top_n,
        selected_context_tokens=selected_tokens,
        context_token_budget_exceeded=(
            request.max_context_tokens is not None and selected_tokens > request.max_context_tokens
        ),
    )


class ContextSelector:
    """Reuse within one event loop. The concurrency bound is shared across select calls."""

    def __init__(
        self,
        provider: Scorer,
        *,
        timeout_ms: float = 5000,
        max_concurrency: int = 8,
        on_error: Literal["passthrough", "raise"] = "passthrough",
    ) -> None:
        if not math.isfinite(timeout_ms) or timeout_ms <= 0:
            raise ValueError("timeout_ms must be finite and positive")
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
            raise ValueError("max_concurrency must be an integer")
        if not 1 <= max_concurrency <= 256:
            raise ValueError("max_concurrency must be between 1 and 256")
        if on_error not in ("passthrough", "raise"):
            raise ValueError("on_error must be passthrough or raise")
        self.provider = provider
        self.timeout_ms = timeout_ms
        self.on_error = on_error
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def select(self, **kwargs: Unpack[SelectOptions]) -> SelectionResult:
        return await self.select_request(SelectRequest.model_validate(kwargs))

    async def select_request(self, request: SelectRequest) -> SelectionResult:
        started = perf_counter()
        completed: dict[int, Judgment] = {}

        async def score_one(i: int) -> Judgment:
            async with self._semaphore:
                judgment = await self.provider.score(
                    request.retrieval_query or request.query,
                    request.documents[i],
                    request.relevance_guidance,
                )
                completed[i] = judgment
                return judgment

        tasks: list[asyncio.Task[Judgment]] = []
        error: ProviderError | None = None
        batch_usage: Usage | None = None
        try:
            async with asyncio.timeout(self.timeout_ms / 1000):
                if request.scoring_strategy == "contextual":
                    if not isinstance(self.provider, BatchScorer):
                        raise ProviderError("contextual_scoring_unsupported")
                    async with self._semaphore:
                        batch = await self.provider.score_context(
                            request.retrieval_query or request.query,
                            request.documents,
                            request.relevance_guidance,
                        )
                    if (
                        len(batch.judgments) != len(request.documents)
                        or not batch.usage.complete
                        or batch.usage.completed_documents != len(request.documents)
                    ):
                        raise ProviderError("invalid_provider_response")
                    judgments, batch_usage = batch.judgments, batch.usage
                else:
                    tasks = [
                        asyncio.create_task(score_one(i)) for i in range(len(request.documents))
                    ]
                    judgments = await asyncio.gather(*tasks)
        except TimeoutError:
            error = ProviderError("deadline_exceeded")
        except ProviderError as exc:
            error = exc
        finally:
            # A failed sibling or caller cancellation must not leave chargeable calls running.
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = (perf_counter() - started) * 1000
        if error is None:
            return apply_policy(request, judgments, elapsed_ms=elapsed, usage=batch_usage)
        if self.on_error == "raise":
            raise error
        characters = sum(len(d.text) for d in request.documents)
        return SelectionResult(
            documents=request.documents,
            selected_ids=None,
            decisions=[
                Decision(
                    id=d.id,
                    input_index=i,
                    relevance=completed[i].relevance if i in completed else None,
                    model=completed[i].model if i in completed else None,
                    input_tokens=completed[i].input_tokens if i in completed else None,
                    output_tokens=completed[i].output_tokens if i in completed else None,
                    selected=None,
                    returned=True,
                    original_rank=i + 1,
                    reason="bypassed",
                )
                for i, d in enumerate(request.documents)
            ],
            status="bypassed",
            prompt_version=request.prompt_version,
            error_code=error.code,
            models=sorted({j.model for j in completed.values()}),
            policy_version=request.policy.version,
            usage=Usage(
                input_tokens=sum(j.input_tokens for j in completed.values()),
                output_tokens=sum(j.output_tokens for j in completed.values()),
                completed_documents=len(completed),
                complete=False,
            ),
            elapsed_ms=round(elapsed, 3),
            input_characters=characters,
            selected_characters=None,
            returned_characters=characters,
        )
