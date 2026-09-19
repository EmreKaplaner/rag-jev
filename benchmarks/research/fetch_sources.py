"""Fetch checksum-pinned data/author metric sources; no credentials or prior runs needed."""

import hashlib
from pathlib import Path

import httpx

from benchmarks.public.common import read_json, write_json
from benchmarks.public.download import main as download_datasets


def main():
    download_datasets()
    for source in read_json("benchmarks/research/metric-sources.json").values():
        target = Path(source["local_path"])
        if target.exists():
            contents = target.read_bytes()
        else:
            response = httpx.get(source["url"], timeout=60, follow_redirects=True)
            response.raise_for_status()
            contents = response.content
        if hashlib.sha256(contents).hexdigest() != source["sha256"]:
            raise ValueError("Author source checksum mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
    write_json(
        "artifacts/research-v3/reranker-source.json",
        read_json("benchmarks/research/reranker-source.json"),
    )
    source = read_json("benchmarks/research/retrieval-protocol.json")["source"]
    target = Path("artifacts/retrieval-v1/scifact.zip")
    if target.exists():
        contents = target.read_bytes()
    else:
        response = httpx.get(source["url"], timeout=60, follow_redirects=True)
        response.raise_for_status()
        contents = response.content
    if hashlib.sha256(contents).hexdigest() != source["sha256"]:
        raise ValueError("SciFact source checksum mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(contents)
    write_json(target.with_name("source.json"), source)
    print("Pinned dataset, metric, and reranker sources ready")


if __name__ == "__main__":
    main()
