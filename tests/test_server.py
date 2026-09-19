from fastapi.testclient import TestClient

from rag_jev import ContextSelector, ProviderError
from rag_jev.demo import CASES, FixtureScorer
from rag_jev.server import create_app


def request_body(**changes):
    return {
        "query": CASES[0].query,
        "documents": [d.model_dump() for d in CASES[0].documents],
        "min_relevance": 0.2,
        **changes,
    }


def test_service_auth_selection_shadow_and_schema():
    with TestClient(create_app(ContextSelector(FixtureScorer()), api_token="service-secret")) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.post("/v1/select", json=request_body()).status_code == 401
        headers = {"Authorization": "Bearer service-secret"}
        response = c.post("/v1/select", json=request_body(), headers=headers)
        assert response.status_code == 200
        assert response.json()["selected_ids"] == ["refund", "receipt"]
        shadow = c.post("/v1/select", json=request_body(shadow=True), headers=headers).json()
        assert shadow["status"] == "shadow" and len(shadow["documents"]) == 3
        assert "/v1/select" in c.get("/openapi.json").json()["paths"]


def test_invalid_requests_and_body_limit():
    with TestClient(create_app(ContextSelector(FixtureScorer()), api_token="")) as c:
        response = c.post("/v1/select", json=request_body(min_relevance=9, extra="PRIVATE"))
        assert response.status_code == 422
        assert "PRIVATE" not in response.text
        assert c.post("/v1/select", content=b"x" * 2_000_001).status_code == 413


def test_server_bypass_and_raise():
    class Failing:
        async def score(self, *args):
            raise ProviderError("provider_auth")

    with TestClient(create_app(ContextSelector(Failing()), api_token="")) as c:
        response = c.post("/v1/select", json=request_body(top_n=1))
        assert response.status_code == 200
        assert response.json()["status"] == "bypassed"
        assert len(response.json()["documents"]) == 3
    with TestClient(create_app(ContextSelector(Failing(), on_error="raise"), api_token="")) as c:
        response = c.post("/v1/select", json=request_body())
        assert response.status_code == 502
