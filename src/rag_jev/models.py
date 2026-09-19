"""The shared Python / HTTP contract. Unknown arguments are errors, not ignored knobs."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Mode = Literal["filter", "rerank", "filter_and_rerank", "fusion"]
PROMPT_VERSION = "useful-evidence-v1"
CONTEXTUAL_PROMPT_VERSION = "contextual-evidence-v1"
ScoringStrategy = Literal["independent", "contextual"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Document(Contract):
    id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=60_000)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    group_id: str | None = Field(default=None, min_length=1, max_length=256)
    pinned: bool = False


class Policy(Contract):
    mode: Mode = "filter"
    min_relevance: Probability | None = None
    top_n: int | None = Field(default=None, ge=1, le=256, strict=True)
    max_context_tokens: int | None = Field(default=None, ge=1, le=1_000_000, strict=True)
    shadow: bool = False
    relevance_guidance: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_threshold(self) -> Self:
        if self.mode not in {"rerank", "fusion"} and self.min_relevance is None:
            raise ValueError("min_relevance is required for filtering; calibrate on your data")
        if self.mode == "rerank" and self.min_relevance is not None:
            raise ValueError("use filter_and_rerank when combining a threshold with reranking")
        if self.mode == "fusion" and self.min_relevance is not None:
            raise ValueError("fusion combines ranks without a relevance threshold")
        return self

    @property
    def version(self) -> str:
        options = self.model_dump(exclude={"shadow"})
        # Preserve policy IDs and recorded comparisons from before token budgets existed.
        if self.max_context_tokens is None:
            options.pop("max_context_tokens")
        payload = json.dumps({"prompt": PROMPT_VERSION, **options}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class SelectRequest(Policy):
    scoring_strategy: ScoringStrategy = "independent"
    query: str = Field(min_length=1, max_length=8000)
    retrieval_query: str | None = Field(default=None, min_length=1, max_length=8000)
    documents: list[Document] = Field(max_length=256)

    @model_validator(mode="after")
    def validate_documents(self) -> Self:
        if not self.query.strip():
            raise ValueError("query must contain non-whitespace text")
        if self.retrieval_query is not None and not self.retrieval_query.strip():
            raise ValueError("retrieval_query must contain non-whitespace text")
        ids = [d.id for d in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("document IDs must be unique within a request")
        if any(not d.text.strip() or not d.id.strip() for d in self.documents):
            raise ValueError("document IDs and text must contain non-whitespace characters")
        if sum(len(d.text) for d in self.documents) > 1_000_000:
            raise ValueError("total document text exceeds 1,000,000 characters")
        return self

    @property
    def policy(self) -> Policy:
        return Policy.model_validate(self.model_dump(include=set(Policy.model_fields)))

    @property
    def prompt_version(self) -> str:
        return (
            CONTEXTUAL_PROMPT_VERSION if self.scoring_strategy == "contextual" else PROMPT_VERSION
        )


class Usage(Contract):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    completed_documents: int = Field(default=0, ge=0)
    complete: bool = True


class Judgment(Contract):
    relevance: Probability
    model: str = Field(min_length=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class Decision(Contract):
    id: str
    input_index: int
    relevance: Probability | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    selected: bool | None
    returned: bool
    original_rank: int | None = Field(default=None, ge=1)
    jev_rank: int | None = Field(default=None, ge=1)
    fusion_score: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    reason: Literal[
        "retained",
        "below_threshold",
        "beyond_top_n",
        "beyond_token_budget",
        "bypassed",
        "pinned",
        "group_retained",
    ]


class SelectionResult(Contract):
    documents: list[Document]
    selected_ids: list[str] | None
    decisions: list[Decision]
    status: Literal["applied", "shadow", "bypassed"]
    error_code: str | None = None
    models: list[str]
    prompt_version: str = PROMPT_VERSION
    policy_version: str
    usage: Usage
    elapsed_ms: float
    input_characters: int
    selected_characters: int | None
    returned_characters: int
    outcome: Literal["context_selected", "no_context_selected", "not_evaluated"] = "not_evaluated"
    top_n_exceeded: bool = False
    selected_context_tokens: int | None = None
    context_token_budget_exceeded: bool | None = None
    token_estimator: str = (
        "cl100k_base passage-text estimate; excludes prompts, separators and output"
    )
