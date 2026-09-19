import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from rag_jev.demo import CASES


def body(**kwargs):
    return {
        "query": CASES[0].query,
        "documents": [d.model_dump() for d in CASES[0].documents],
        "min_relevance": 0.2,
        **kwargs,
    }


def test_real_http_python_service_official_sdk_roundtrip(network_service):
    with httpx.Client(
        base_url=network_service, headers={"Authorization": "Bearer integration-test-token"}
    ) as client:
        result = client.post("/v1/select", json=body()).json()
        assert result["selected_ids"] == ["refund", "receipt"]
        assert result["models"] == ["fixture-http-not-jev"]
        assert result["usage"]["input_tokens"] == 75
        assert result["documents"][0]["metadata"]["source"] == "refunds.md"
        for query, code in [("__fail__", "provider_error"), ("__timeout__", "deadline_exceeded")]:
            result = client.post("/v1/select", json=body(query=query, top_n=1)).json()
            assert result["status"] == "bypassed"
            assert result["error_code"] == code
            assert len(result["documents"]) == 3


def test_typescript_client_through_real_http_and_sdk(network_service):
    if not Path("clients/typescript/dist/index.js").exists():
        pytest.fail("Build the TypeScript client first: npm --prefix clients/typescript run build")
    result = subprocess.run(
        [
            "node",
            "--test",
            "clients/typescript/test/network.e2e.mjs",
        ],
        env={**os.environ, "RAG_JEV_TEST_URL": network_service},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_uses_official_sdk_against_http_endpoint(network_service):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_jev.cli",
            "select",
            "examples/request.json",
        ],
        env={**os.environ, "TYPESAFE_API_KEY": "fake", "TYPESAFE_BASE_URL": network_service},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["selected_ids"] == ["refund", "receipt"]


def test_langchain_real_documents_lcel_and_metadata(network_service):
    from langchain_core.documents import Document

    from rag_jev.integrations.langchain import JevDocumentCompressor
    from rag_jev.models import Policy

    sentinel = object()
    docs = [
        Document(id="same-source", page_content=d.text, metadata={"original": sentinel})
        for d in CASES[0].documents
    ]
    compressor = JevDocumentCompressor(
        base_url=network_service,
        api_token="integration-test-token",
        policy=Policy(min_relevance=0.2),
    )
    result = compressor.as_runnable().invoke({"query": CASES[0].query, "documents": docs})
    assert result[0] is docs[1] and result[1] is docs[2]
    assert result[0].metadata["original"] is sentinel
    assert compressor.last_result.status == "applied"


async def test_langchain_async_shadow_and_grouping(network_service):
    from langchain_core.documents import Document

    from rag_jev.integrations.langchain import JevDocumentCompressor
    from rag_jev.models import Policy

    docs = [Document(page_content=d.text, metadata={"parent": "one"}) for d in CASES[0].documents]
    compressor = JevDocumentCompressor(
        base_url=network_service,
        api_token="integration-test-token",
        policy=Policy(min_relevance=0.9),
        group_metadata_key="parent",
    )
    result = await compressor.as_runnable().ainvoke({"query": CASES[0].query, "documents": docs})
    assert len(result) == 3 and all(a is b for a, b in zip(result, docs, strict=True))


def test_dify_sdk_against_real_http(network_service):
    python = Path("integrations/dify/.venv/bin/python")
    if not python.exists():
        pytest.skip("Run make setup-integrations to install the isolated Dify SDK")
    result = subprocess.run(
        [str(python), "scripts/check_dify.py"],
        env={
            **os.environ,
            "RAG_JEV_TEST_URL": network_service,
            "RAG_JEV_TEST_TOKEN": "integration-test-token",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_comparison_http_export_import_and_policy_staleness(network_service):
    with httpx.Client(
        base_url=network_service,
        headers={"Authorization": "Bearer integration-test-token"},
        timeout=20,
    ) as client:
        view = client.post(
            "/v1/runs",
            json={
                "request": body(),
                "relevant_ids": CASES[0].relevant_ids,
                "baseline_ids": ["refund"],
            },
        ).json()
        replay = {"record": view["record"], "policy": view["policy"]}
        response = client.post("/v1/compare", json=replay)
        assert response.status_code == 200, response.text
        comparison = response.json()
        assert comparison["baseline"]["model"] == "controlled-http-model"
        assert comparison["baseline"]["input_tokens"] == 70
        assert comparison["selected"]["input_tokens"] == 90
        assert comparison["selected"]["citations"] == ["refund", "receipt"]
        bundle = {**replay, "comparison": comparison}
        assert client.post("/v1/import", json=bundle).status_code == 200
        bundle["policy"]["min_relevance"] = 0.9
        assert client.post("/v1/import", json=bundle).status_code == 422
