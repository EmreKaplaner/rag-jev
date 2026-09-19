"""Dify's native reranker contract mapped to rag-jev, preserving input indices."""

import httpx
from dify_plugin.entities.model.rerank import RerankDocument, RerankResult
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeServerUnavailableError
from dify_plugin.interfaces.model.rerank_model import RerankModel


class RagJevRerankModel(RerankModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        query: str,
        docs: list[str],
        score_threshold: float | None = None,
        top_n: int | None = None,
        user: str | None = None,
    ) -> RerankResult:
        if not docs:
            return RerankResult(model=model, docs=[])
        base = credentials["base_url"].rstrip("/")
        parsed = httpx.URL(base)
        if (
            parsed.scheme not in ("http", "https")
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise InvokeServerUnavailableError("Invalid rag-jev service URL")
        token = credentials.get("api_token", "")
        fusion = credentials.get("ranking_mode") == "fusion"
        if fusion and score_threshold is not None:
            raise InvokeServerUnavailableError(
                "Disable the score threshold for fusion; RRF scores are not relevance probabilities"
            )
        payload = {
            "query": query,
            "scoring_strategy": credentials.get("scoring_strategy") or "independent",
            "documents": [{"id": str(i), "text": text} for i, text in enumerate(docs)],
            "mode": "fusion"
            if fusion
            else "rerank"
            if score_threshold is None
            else "filter_and_rerank",
            "min_relevance": score_threshold,
            "top_n": top_n,
        }
        try:
            budget = credentials.get("max_context_tokens")
            if budget:
                payload["max_context_tokens"] = int(budget)
            with httpx.Client(timeout=60, follow_redirects=False) as client:
                response = client.post(
                    base + "/v1/select",
                    json=payload,
                    headers={"Authorization": "Bearer " + token} if token else {},
                )
            if response.status_code != 200:
                raise ValueError("service unavailable")
            result = response.json()
            if result["status"] != "applied":
                raise ValueError("service bypassed; no trustworthy rerank scores")
            scores = {
                x["id"]: x["fusion_score" if fusion else "relevance"] for x in result["decisions"]
            }
            output = []
            seen = set()
            for document in result["documents"]:
                index = int(document["id"])
                score = scores[document["id"]]
                if (
                    index in seen
                    or not 0 <= index < len(docs)
                    or document["text"] != docs[index]
                    or not 0 <= score <= 1
                ):
                    raise ValueError("invalid document mapping")
                seen.add(index)
                output.append(RerankDocument(index=index, text=docs[index], score=score))
            return RerankResult(model=model, docs=output)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise InvokeServerUnavailableError(
                "rag-jev selection failed; check service configuration"
            ) from None

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            self._invoke(
                model,
                credentials,
                "What is the return period?",
                ["Returns are accepted within thirty days."],
                top_n=1,
            )
        except Exception:
            raise CredentialsValidateFailedError(
                "Could not score a passage through rag-jev"
            ) from None

    @property
    def _invoke_error_mapping(self) -> dict:
        return {}
