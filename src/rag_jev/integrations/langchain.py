"""LangChain compressor using the HTTP service, with lossless original-document mapping."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.runnables import RunnableLambda
from pydantic import Field, PrivateAttr, SecretStr, field_validator

from rag_jev.models import Policy, ScoringStrategy, SelectionResult, SelectRequest


class JevDocumentCompressor(BaseDocumentCompressor):
    """Pass retrieved documents to Jev. Both synchronous and asynchronous calls work.

    Internal numeric IDs preserve duplicate source IDs, non-JSON metadata, and object identity.
    Only text and explicitly configured grouping/pinning metadata are sent to the service.
    """

    base_url: str = "http://127.0.0.1:8000"
    api_token: SecretStr = SecretStr("")
    policy: Policy
    scoring_strategy: ScoringStrategy = "independent"
    timeout: float = Field(default=15, gt=0)
    group_metadata_key: str | None = None
    pin_metadata_key: str | None = None
    _last_result: SelectionResult | None = PrivateAttr(default=None)

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        url = httpx.URL(value)
        if (
            url.scheme not in ("http", "https")
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("base_url must be HTTP(S), without credentials/query/fragment")
        return value.rstrip("/")

    @property
    def last_result(self) -> SelectionResult | None:
        """Diagnostics for the most recent completed call; avoid in concurrent request logic."""
        return self._last_result

    def _request(self, documents: Sequence[Document], query: str) -> SelectRequest:
        docs = []
        for i, d in enumerate(documents):
            group = d.metadata.get(self.group_metadata_key) if self.group_metadata_key else None
            pinned = (
                d.metadata.get(self.pin_metadata_key, False) if self.pin_metadata_key else False
            )
            docs.append(
                {
                    "id": str(i),
                    "text": d.page_content,
                    "group_id": str(group) if group is not None else None,
                    "pinned": pinned is True,
                }
            )
        return SelectRequest.model_validate(
            {
                "query": query,
                "documents": docs,
                "scoring_strategy": self.scoring_strategy,
                **self.policy.model_dump(),
            }
        )

    def _headers(self) -> dict[str, str]:
        token = self.api_token.get_secret_value()
        return {"Authorization": "Bearer " + token} if token else {}

    def _finish(
        self, response: httpx.Response, request: SelectRequest, documents: Sequence[Document]
    ) -> Sequence[Document]:
        if response.status_code != 200:
            raise RuntimeError(f"rag-jev service returned HTTP {response.status_code}")
        result = SelectionResult.model_validate(response.json())
        original = {d.id: d for d in request.documents}
        if any(d.id not in original or d != original[d.id] for d in result.documents):
            raise ValueError("service changed source content or IDs")
        ids = [d.id for d in result.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("service returned duplicate IDs")
        self._last_result = result
        return [documents[int(i)] for i in ids]

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks | None = None
    ) -> Sequence[Document]:
        request = self._request(documents, query)
        with httpx.Client(timeout=self.timeout, follow_redirects=False) as client:
            response = client.post(
                self.base_url.rstrip("/") + "/v1/select",
                json=request.model_dump(),
                headers=self._headers(),
            )
        return self._finish(response, request, documents)

    async def acompress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks | None = None
    ) -> Sequence[Document]:
        request = self._request(documents, query)
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
            response = await client.post(
                self.base_url.rstrip("/") + "/v1/select",
                json=request.model_dump(),
                headers=self._headers(),
            )
        return self._finish(response, request, documents)

    def as_runnable(self) -> RunnableLambda[dict[str, Any], Sequence[Document]]:
        """Compose with LCEL: input is {query: str, documents: list[Document]}."""
        return RunnableLambda(
            lambda value: self.compress_documents(value["documents"], value["query"]),
            afunc=lambda value: self.acompress_documents(value["documents"], value["query"]),
        )
