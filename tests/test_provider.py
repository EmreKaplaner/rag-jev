import json

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from rag_jev import Document, Jev, ProviderError


def payload(probability=0.9):
    return {
        "model": "jev-test-pinned",
        "usage": {"input_tokens": 42, "output_tokens": 3},
        "answers": {"relevant": {"type": "noul", "noul": probability}},
    }


async def test_official_sdk_contract_and_private_metadata():
    seen = []

    async def handler(request):
        seen.append(json.loads(request.content))
        assert request.url.path == "/v1/systemone"
        return httpx2.Response(200, json=payload())

    async with AsyncTypeSafeClient(
        api_key="fake", transport=httpx2.MockTransport(handler)
    ) as client:
        provider = Jev(client=client, model="jev-pinned")
        result = await provider.score(
            "Refund window?",
            Document(id="a", text="30 days", metadata={"secret": "private"}),
            "Keep partial evidence",
        )
        assert result.relevance == 0.9 and result.input_tokens == 42
        assert result.model == "jev-test-pinned"
        await provider.aclose()  # Borrowed clients remain open.
        assert not client._http_client.is_closed
    sent = seen[0]
    assert sent["model"] == "jev-pinned"
    assert sent["state"] == {"query": "Refund window?", "passage": {"text": "30 days"}}
    assert "private" not in json.dumps(sent)
    assert "Keep partial evidence" in sent["questions"]["relevant"]["instructions"]
    assert sent["questions"]["relevant"]["type"] == "noul"


@pytest.mark.parametrize(
    "body",
    [
        {"model": "x", "usage": {}, "answers": {}},
        payload(2.5),
        payload("not-a-number"),
        {"model": "x", "answers": {"relevant": {"type": "choice", "choice": "yes"}}},
    ],
)
async def test_malformed_response_is_a_provider_failure(body):
    async with AsyncTypeSafeClient(
        api_key="fake",
        transport=httpx2.MockTransport(lambda r: httpx2.Response(200, json=body)),
        retry=RetryPolicy(max_retries=0),
    ) as client:
        with pytest.raises(ProviderError):
            await Jev(client=client).score("q", Document(id="a", text="doc"), None)


@pytest.mark.parametrize(
    "status,code", [(401, "provider_auth"), (429, "provider_rate_limit"), (503, "provider_error")]
)
async def test_upstream_error_body_is_not_exposed(status, code):
    async with AsyncTypeSafeClient(
        api_key="fake",
        transport=httpx2.MockTransport(lambda r: httpx2.Response(status, text="SECRET input")),
        retry=RetryPolicy(max_retries=0),
    ) as client:
        with pytest.raises(ProviderError) as exc:
            await Jev(client=client).score("q", Document(id="a", text="doc"), None)
        assert str(exc.value) == code
        assert "SECRET" not in str(exc.value)


async def test_owned_client_is_closed():
    provider = Jev(api_key="fake")
    await provider.aclose()
    assert provider.client._http_client.is_closed
