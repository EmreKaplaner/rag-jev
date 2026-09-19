"""Publish the separate exploratory corpus-retrieval result after metric verification."""

import hashlib
import html
from pathlib import Path

from benchmarks.public.common import read_json, write_json

ROOT = Path("artifacts/retrieval-v1")
ASSETS = Path("src/rag_jev/static")


def main():
    report = read_json(ROOT / "report.json")
    verified = read_json(ROOT / "verification.json")
    assert verified["status"] == "passed"
    for name, checksum in verified["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == checksum
    headings = ["Pipeline", "NDCG@10", "Recall@10", "MRR@10"]
    rows = [
        [arm, *[f"{values[k] * 100:.2f}" for k in ["ndcg10", "recall10", "mrr10"]]]
        for arm, values in report["summary"].items()
    ]
    paragraphs = [
        "Exploratory retrieval-only result: all 300 BEIR/SciFact test queries and all 5,183 "
        "corpus documents. Custom BM25 retrieves 20 candidates; BM25, Jev and Ettin each "
        "return 10 from that identical pool. No gold passages are inserted. "
        "All table values are percentages.",
        f"First-stage Recall@20: {report['first_stage_recall20'] * 100:.2f}%. "
        "Queries with missing relevant papers remain in every denominator.",
    ]
    for name, result in report["ndcg10_comparisons_descriptive"].items():
        lo, hi = result["ci95"]
        paragraphs.append(
            f"{name.replace('_', ' ')}: NDCG@10 difference {result['delta'] * 100:+.2f} points; "
            f"descriptive 95% interval [{lo * 100:+.2f}, {hi * 100:+.2f}]. "
            f"Bootstrap units: {result['query_clusters']} connected groups of queries sharing "
            "relevant papers; the estimate remains query-weighted."
        )
    paragraphs.extend(
        [
            f"Jev scoring API usage estimate: ${report['jev_total_api_usd']:.5f} for 300 queries "
            f"(${report['jev_total_api_usd'] / 300 * 1000:.4f} per 1,000). "
            f"Ettin inference: {report['ettin_total_cpu_seconds']:.1f} process CPU seconds; "
            "its hosting cost is unknown. Common retrieval cost is excluded; these figures "
            "do not establish a total-cost advantage over Ettin.",
            f"All {verified['metric_values_checked']:,} per-query metric values match "
            f"pytrec-eval-terrier {verified['version']} within 1e-12. The exported run contains "
            "only ten results per query, so reciprocal rank is truncated at ten.",
            "This is one retrieval dataset with a custom BM25 first stage, not the full BEIR "
            "suite, an answer-accuracy result, or a state-of-the-art claim. The separate "
            "HotpotQA/MuSiQue answer study keeps its original primary endpoints.",
            "Ettin's published training recipe selects checkpoints on NanoBEIR, including "
            "SciFact and HotpotQA; overlap with those selection examples has not been ruled "
            "out. Jev pretraining exposure is unknown. These are not demonstrably "
            "benchmark-naive models. No dense retriever or query expansion is evaluated.",
        ]
    )
    markdown = "# Exploratory BEIR/SciFact corpus retrieval\n\n"
    markdown += "| " + " | ".join(headings) + " |\n| " + " | ".join(["---"] * 4) + " |\n"
    markdown += "\n".join("| " + " | ".join(row) + " |" for row in rows)
    markdown += "\n\n" + "\n\n".join(paragraphs)
    markdown += (
        "\n\nSources: [BEIR data](https://github.com/beir-cellar/beir), "
        "[Ettin training recipe](https://huggingface.co/blog/ettin-reranker), "
        "[trec_eval wrapper](https://github.com/cvangysel/pytrec_eval). "
        "[Methods and reproduction](README.md).\n"
    )
    Path("benchmarks/research/RETRIEVAL_RESULTS.md").write_text(markdown)
    table = "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headings)
    table += "</tr></thead><tbody>"
    table += "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    table += "</tbody></table>"
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>rag-jev · SciFact retrieval study</title><link rel="stylesheet" href="/assets/style.css">
<style>table{width:100%;max-width:650px;text-align:left}th,td{padding:12px}main>p{max-width:85ch;line-height:1.7}</style>
</head><body><header class="masthead"><a class="wordmark" href="/">rag—jev</a>
<span>EXPLORATORY / CORPUS RETRIEVAL</span><a href="/assets/research-v3.html">Answer study ↗</a>
</header><main><section class="intro"><div><p class="eyebrow">BEIR / SciFact</p>
<h1>Retrieve first.<br><em>Rerank the same candidates.</em></h1></div></section>"""
    page += table + "".join("<p>" + html.escape(p) + "</p>" for p in paragraphs)
    page += """<p><a href="/assets/retrieval-v1-results.json" download>Results</a> ·
<a href="/assets/retrieval-v1-protocol.json" download>Frozen protocol</a> ·
<a href="/assets/retrieval-v1-verification.json" download>Metric verification</a> ·
<a href="https://huggingface.co/blog/ettin-reranker">Baseline training recipe</a></p>
<footer>Locally frozen exploratory study; not peer reviewed.</footer></main></body></html>"""
    (ASSETS / "retrieval-v1.html").write_text(page)
    for name, value in [
        ("results", report),
        ("verification", verified),
        ("protocol", read_json(ROOT / "protocol.json")),
    ]:
        write_json(ASSETS / f"retrieval-v1-{name}.json", value)
    print("Published separate SciFact retrieval study")


if __name__ == "__main__":
    main()
