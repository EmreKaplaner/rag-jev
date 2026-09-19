"""Real local retrieval -> Jev selection -> optional answer comparison.

Uses SQLite FTS5/BM25, so trying your own Markdown/text files needs no embedding key.
The retrieval algorithm is intentionally replaceable by your production retriever.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sqlite3
from pathlib import Path

import httpx
from dotenv import load_dotenv

from rag_jev.models import Document, SelectRequest
from rag_jev.workbench import ReplayBundle, StudyRequest


def retrieve(directory: Path, query: str, top_k: int) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        if path.stat().st_size > 5_000_000:
            raise ValueError(f"File exceeds example's 5 MB limit: {path.name}")
        relative = str(path.relative_to(directory))
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", path.read_text()) if p.strip()]
        for i, paragraph in enumerate(paragraphs):
            for j in range(0, len(paragraph), 2000):
                doc_id = hashlib.sha256(f"{relative}:{i}:{j}".encode()).hexdigest()[:16]
                documents.append(
                    Document(
                        id=doc_id,
                        text=paragraph[j : j + 2000],
                        metadata={"source": relative, "paragraph": i + 1},
                    )
                )
    terms = list(dict.fromkeys(re.findall(r"\w+", query, re.UNICODE)))[:64]
    if not terms:
        return []
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE VIRTUAL TABLE passages USING fts5(id UNINDEXED, text)")
        db.executemany("INSERT INTO passages VALUES (?, ?)", [(d.id, d.text) for d in documents])
        rows = db.execute(
            "SELECT id FROM passages WHERE passages MATCH ? ORDER BY bm25(passages) LIMIT ?",
            (" OR ".join(f'"{term}"' for term in terms), top_k),
        ).fetchall()
    by_id = {d.id: d for d in documents}
    return [by_id[row[0]] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--mode", choices=["filter", "rerank", "filter_and_rerank", "fusion"], default="filter"
    )
    parser.add_argument("--top-n", type=int)
    parser.add_argument(
        "--scoring-strategy", choices=["independent", "contextual"], default="independent"
    )
    parser.add_argument(
        "--min-relevance", type=float, help="Required for filtering; omit for rerank/fusion"
    )
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Call your configured answer model")
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-rag-run.json"))
    args = parser.parse_args()
    if not args.documents.is_dir() or not 1 <= args.top_k <= 256:
        parser.error("provide a document directory and top-k between 1 and 256")
    load_dotenv(override=False)
    candidates = retrieve(args.documents, args.query, args.top_k)
    request = SelectRequest(
        query=args.query,
        documents=candidates,
        min_relevance=args.min_relevance,
        shadow=args.shadow,
        mode=args.mode,
        top_n=args.top_n,
        scoring_strategy=args.scoring_strategy,
    )
    token = os.getenv("RAG_JEV_API_TOKEN", "")
    with httpx.Client(
        base_url=args.base_url,
        headers={"Authorization": "Bearer " + token} if token else {},
        timeout=150,
        follow_redirects=False,
    ) as client:
        response = client.post("/v1/runs", json=StudyRequest(request=request).model_dump())
        if response.status_code != 200:
            raise SystemExit(
                f"Selection failed (HTTP {response.status_code}); check service logs/configuration."
            )
        view = response.json()
        bundle = {"record": view["record"], "policy": view["policy"], "comparison": None}
        if args.compare:
            response = client.post(
                "/v1/compare", json={"record": view["record"], "policy": view["policy"]}
            )
            if response.status_code != 200:
                raise SystemExit(
                    f"Comparison failed (HTTP {response.status_code}); "
                    "configure the generation endpoint."
                )
            bundle["comparison"] = response.json()
    result = ReplayBundle.model_validate(bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2) + "\n")
    selected = view["selection"]["selected_ids"]
    print(
        f"Retrieved {len(candidates)} passages with SQLite FTS5/BM25; Jev selected {len(selected)}."
    )
    print("Shadow mode returns all original passages." if args.shadow else "Selection is applied.")
    if not selected:
        print(
            "No context selected: retrieve more evidence or clarify the question before answering."
        )
    print(f"Import {args.output} in the playground to inspect decisions and tune the cutoff.")
    if result.comparison and any(
        a.status == "error" for a in (result.comparison.baseline, result.comparison.selected)
    ):
        raise SystemExit(
            "Generation failed in at least one branch; the saved replay contains details."
        )


if __name__ == "__main__":
    main()
