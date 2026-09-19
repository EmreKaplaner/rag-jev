"""Exercise the production image across isolated containers; no real API calls."""

import argparse
import json
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLIENT = r"""
import os, time
import httpx
from rag_jev.demo import CASES
from rag_jev.models import SelectionResult

assert os.getuid() == 10001
assert not os.path.exists('/app/.env')
base = 'http://selector:8080'
with httpx.Client(base_url=base, timeout=10) as client:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            ready = client.get('/healthz', timeout=.5).status_code == 200
            upstream = httpx.get('http://upstream:9000/healthz', timeout=.5)
            if ready and upstream.status_code == 200:
                break
        except httpx.TransportError:
            pass
        time.sleep(.2)
    else:
        raise AssertionError('service failed to become ready')
    body = {'query': CASES[0].query,
            'documents': [d.model_dump() for d in CASES[0].documents],
            'min_relevance': .2}
    assert client.post('/v1/select', json=body).status_code == 401
    client.headers['Authorization'] = 'Bearer container-test-token'
    selected = client.post('/v1/select', json=body)
    selected.raise_for_status()
    result = SelectionResult.model_validate(selected.json())
    assert result.status == 'applied', result
    assert result.selected_ids == ['refund', 'receipt'], result
    assert result.documents[0].metadata['source'] == 'refunds.md'
    body['scoring_strategy'] = 'contextual'
    contextual = client.post('/v1/select', json=body).json()
    assert contextual['status'] == 'applied'
    assert contextual['selected_ids'] == ['refund', 'receipt']
    assert contextual['usage']['input_tokens'] == 60
    fusion = {**body, 'mode': 'fusion', 'min_relevance': None, 'top_n': 1}
    fused = client.post('/v1/select', json=fusion).json()
    assert fused['selected_ids'] == ['refund']
    assert fused['decisions'][1]['jev_rank'] == 1
    run = client.post('/v1/runs', json={'request': fusion}).json()
    comparison = client.post('/v1/rankings', json={'record': run['record'], 'top_n': 1}).json()
    assert [a['selection']['selected_ids'] for a in comparison['arms']] == [
        ['shipping'], ['refund'], ['refund']]
    assert comparison['scoring_calls'] == 0
    body['shadow'] = True
    shadow = client.post('/v1/select', json=body).json()
    assert shadow['status'] == 'shadow'
    assert [d['id'] for d in shadow['documents']] == [d['id'] for d in body['documents']]
    # Simulate HTTPS ingress forwarding to HTTP inside the private test network.
    headers = {'Host': 'rag.example.test', 'X-Forwarded-Proto': 'https',
               'Origin': 'https://rag.example.test'}
    assert client.post('/v1/select', json=body, headers=headers).status_code == 200
    headers['Origin'] = 'https://unrelated.example.test'
    assert client.post('/v1/select', json=body, headers=headers).status_code == 403
    assert client.get('/research').status_code == 200
    assert client.get('/assets/brand/favicon.svg').status_code == 200
print('PASS: remote DNS, auth, selection, metadata, shadow, HTTPS forwarding, research assets')
"""


def docker(*args, check=True):
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="rag-jev:local")
    args = parser.parse_args()
    suffix = uuid.uuid4().hex[:10]
    network, upstream, service = (f"rag-jev-{s}-{suffix}" for s in ["net", "upstream", "service"])
    try:
        docker("network", "create", "--internal", network)
        docker(
            "run",
            "-d",
            "--name",
            upstream,
            "--network",
            network,
            "--network-alias",
            "upstream",
            "--read-only",
            "--tmpfs",
            "/tmp",
            "--cap-drop",
            "ALL",
            "-v",
            f"{ROOT / 'tests/network_app.py'}:/fixtures/network_app.py:ro",
            "-e",
            "TEST_UPSTREAM_URL=http://upstream:9000",
            "--entrypoint",
            "uvicorn",
            args.image,
            "network_app:create_network_app",
            "--factory",
            "--app-dir",
            "/fixtures",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--no-access-log",
        )
        docker(
            "run",
            "-d",
            "--name",
            service,
            "--network",
            network,
            "--network-alias",
            "selector",
            "--read-only",
            "--tmpfs",
            "/tmp",
            "--cap-drop",
            "ALL",
            "-e",
            "TYPESAFE_API_KEY=container-test-only",
            "-e",
            "TYPESAFE_BASE_URL=http://upstream:9000",
            "-e",
            "RAG_JEV_API_TOKEN=container-test-token",
            "-e",
            "PORT=8080",
            # The entire test network is isolated. Real deployments trust only their ingress.
            "-e",
            "FORWARDED_ALLOW_IPS=*",
            args.image,
        )
        result = docker(
            "run",
            "--rm",
            "--network",
            network,
            "--entrypoint",
            "python",
            args.image,
            "-c",
            CLIENT,
            check=False,
        )
        if result.returncode:
            print(result.stdout + result.stderr)
            print(docker("logs", service, check=False).stderr)
            raise SystemExit(result.returncode)
        settings = json.loads(docker("inspect", service).stdout)[0]
        assert settings["HostConfig"]["ReadonlyRootfs"]
        assert settings["Config"]["User"] == "app"
        print(result.stdout.strip())
    finally:
        docker("rm", "-f", service, upstream, check=False)
        docker("network", "rm", network, check=False)


if __name__ == "__main__":
    main()
