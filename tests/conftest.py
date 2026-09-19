import os
import socket
import subprocess
import sys
import time

import httpx
import pytest


@pytest.fixture(scope="session")
def network_service():
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tests.network_app:create_network_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "error",
            "--no-access-log",
        ],
        env={**os.environ, "TEST_UPSTREAM_URL": base_url},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail("test service failed: " + process.stdout.read())
            try:
                if httpx.get(base_url + "/healthz", timeout=0.2).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.05)
        else:
            pytest.fail("test service did not become ready")
        yield base_url
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
