"""Build an evidence-linked pilot report and figure from audited, complete artifacts."""

import shutil
from pathlib import Path

from benchmarks.ecosystem.common import ARMS, ROOT, Ledger
from benchmarks.public.common import read_json, write_json

LABELS = {
    "baseline": "Retriever top10",
    "bge": "BGE top10",
    "jev_rerank": "Jev rerank top10",
    "jev_filter": "Jev filter + rerank",
    "bge_jev_filter": "BGE + Jev filter",
}


def main():
    report = read_json(ROOT / "report.json")
    verification = read_json(ROOT / "verification.json")
    assert verification["complete"], "Audit must pass before publishing completed results"
    bergen = read_json(ROOT / "bergen-verification.json")
    semantic = read_json(ROOT / "crag-semantic-report.json")
    checker = read_json(ROOT / "ragchecker-report.json")
    assert all(g["failed"] == 0 for g in semantic["summary"].values())
    live = read_json(ROOT / "live-flashrag-integration.json")
    assert live["complete"]
    budget = Ledger().summary()
    destination = Path("benchmarks/ecosystem")
    write_json(destination / "results.json", report)
    write_json(
        destination / "live-integration-summary.json",
        {
            "complete": True,
            "runs": len(live["rows"]),
            "protocol_hash": live["protocol_hash"],
            "api_cost_usd": sum(r["api_cost_usd"] for r in live["rows"]),
            "scope": live["scope"],
            "metadata_preserved": True,
        },
    )
    for name in [
        "verification.json",
        "bergen-verification.json",
        "ragchecker-report.json",
        "crag-semantic-report.json",
        "wiki-resource-probe.json",
        "ragchecker-empty-extraction-amendment.json",
    ]:
        shutil.copyfile(ROOT / name, destination / name)
    # Standalone source-data table: no HTML/corpus redistribution and no API credentials.
    shutil.copyfile(ROOT / "per-query.json", destination / "per-query.json")
    lines = [
        "# Ecosystem pilot results",
        "",
        "This is a small exploratory pilot, not a SOTA or universal-improvement claim. "
        "All five policies, datasets and query samples were fixed before paid evaluation.",
        "",
        f"Recorded API usage cost: **${budget['known_usage_cost_usd']:.4f}** of the $25 ceiling. "
        f"**${budget['remaining_after_reservations_usd']:.4f} remains**; "
        f"{budget['unreconciled_requests']} unreconciled requests. "
        "This is usage-based accounting, not a vendor invoice. Local compute cost is unknown.",
        "",
        "## Full-corpus BEIR retrieval",
        "",
        "nDCG@10, percent. 100 test queries per dataset, identical queries across retrievers. "
        "The corpora are complete; the query sets are sampled. SciFact is a previously exposed "
        "replication. These are retrieval metrics, not generated-answer accuracy.",
        "",
        "| Dataset / retrieval | Original | BGE | Jev rerank | Jev filter | BGE + Jev filter |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    retrieval_groups = [
        (name, group)
        for name, group in report["groups"].items()
        if name.endswith("/evaluation") and not name.startswith("crag/")
    ]
    for name, group in retrieval_groups:
        scores = [f"{group['summary'][arm]['ndcg10'] * 100:.2f}" for arm in ARMS]
        lines.append("| " + name.removesuffix("/evaluation") + " | " + " | ".join(scores) + " |")
    lines += [
        "",
        "Paired, descriptive cluster-bootstrap intervals are in [results.json](results.json). "
        "NFCorpus has only ten query clusters, including one containing 91 of the 100 "
        "queries; its interval is particularly unstable. Do not treat the 100 queries as "
        "100 independent observations or infer a confirmatory win.",
        "",
        "## CRAG through native FlashRAG",
        "",
        "40 official public-test questions, five supplied search pages each, one generated "
        "answer per arm. The additional 16 validation questions are reported separately in "
        "the JSON, not mixed into this table. Semantic accuracy uses pinned CRAG prompts "
        "with a Luna judge and corrected evaluation loop; it is **not an official CRAG "
        "leaderboard result**. Generator/judge model overlap introduces bias.",
        "",
        "| Pipeline | Answer F1 | Semantic accuracy | Utility | Mean passages | "
        "Normalized API $/query |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    qa = report["groups"]["crag/bm25/evaluation"]["summary"]
    for arm in ARMS:
        row, grade = qa[arm], semantic["summary"][arm]
        lines.append(
            f"| {LABELS[arm]} | {row['f1'] * 100:.2f} | {grade['accuracy'] * 100:.1f} | "
            f"{grade['utility_failures_minus_one']:.3f} | {row['selected_documents']:.2f} | "
            f"{row['normalized_api_cost_usd']:.6f} |"
        )
    lines += [
        "",
        "API costs include the selector and generation, normalized to uncached input "
        "prices. BGE's local computation is not free and is excluded from this monetary "
        "comparison. CRAG utility scores correct/abstained/incorrect responses as +1/0/−1. "
        "Short-answer F1 can undercount valid aliases; the two metrics should not be "
        "interchanged. No multiple-comparison-corrected superiority claim is made.",
        "",
        "![Pilot overview](pilot.png)",
        "",
        "## Verification and diagnostics",
        "",
        f"- {verification['checks']:,} artifact checks passed, including equality with "
        "the production selection policies and reconciliation of returned usage to the ledger.",
        "- Two additional native FlashRAG runs executed real Jev selection and Luna generation "
        "on the same managed adapter, preserving metadata and closing its connections. "
        "These integration checks are excluded from benchmark accuracy metrics.",
        f"- Native BERGEN `Rerank.eval` reproduced {bergen['queries']} complete rankings "
        f"({bergen['pairs']:,} query/document pairs). This is a rerank-stage check, not "
        "the full BERGEN generation benchmark.",
        f"- Native RAGChecker evaluated {checker['coverage']['cases']} responses from eight "
        "paired test questions. [Metrics and extraction coverage](ragchecker-report.json) "
        "are diagnostics, not independent human judgments.",
        f"- {len(list((ROOT / 'empty-extraction-interpretations').glob('*.json')))} completed "
        "empty extractor responses were retained as zero claims using RefChecker's native "
        "parser. Raw traces and charges remain unchanged; no retry or invented claims. "
        "The adapter amendment and empty-extraction coverage are published explicitly.",
        "- Model revisions, input checksums, truncation counts, local runtime and all "
        "development results are retained in the protocol and JSON reports.",
        "- A blinded 40-question/five-arm human review packet is available locally as "
        "`artifacts/ecosystem-v1/human-review-blinded.json`. No human adjudication has "
        "been completed or claimed.",
        "",
        "## What remains",
        "",
        "1. Full-Wikipedia NQ, TriviaQA and HotpotQA on a larger machine. The E5 index "
        "alone expands to 64.56 GB, plus the corpus; the compressed index is 37.28 GB.",
        "2. Full BERGEN generation replication, additional retrievers/rerankers, and "
        "repeated generation with simultaneous confidence intervals.",
        "3. Independent human review and a shadow trial using real production queries. "
        "Calibrate thresholds on separate development data before deployment.",
        "",
        "[Reproduction commands and limitations](README.md) · [Frozen protocol](protocol.json) "
        "· [Per-query metrics](per-query.json)",
        "",
    ]
    (destination / "RESULTS.md").write_text("\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout="constrained")
    colors = ["#64748b", "#e09c26", "#2874a6", "#15856d", "#8756aa"]
    x = np.arange(len(retrieval_groups))
    for i, arm in enumerate(ARMS):
        axes[0].bar(
            x + (i - 2) * 0.16,
            [g["summary"][arm]["ndcg10"] * 100 for _, g in retrieval_groups],
            width=0.15,
            color=colors[i],
            label=LABELS[arm],
        )
    axes[0].set_xticks(
        x,
        [name.replace("/evaluation", "").replace("/", "\n") for name, _ in retrieval_groups],
        fontsize=8,
    )
    axes[0].set_ylabel("nDCG@10 (%)")
    axes[0].set_title("BEIR: full corpora, 100 test queries each")
    axes[0].set_ylim(0, 100)
    for i, arm in enumerate(ARMS):
        axes[1].scatter(
            qa[arm]["normalized_api_cost_usd"] * 1000,
            semantic["summary"][arm]["accuracy"] * 100,
            s=70,
            color=colors[i],
            label=LABELS[arm],
        )
    axes[1].set_xlabel("Normalized selector + generation API $ / 1,000 queries")
    axes[1].set_ylabel("CRAG-style semantic accuracy (%)")
    axes[1].set_title("CRAG: 40 test questions, same-model judge")
    axes[1].legend(fontsize=8, loc="best")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Exploratory pilot • no SOTA claim • local compute excluded from API cost", fontsize=11
    )
    fig.savefig(destination / "pilot.png", dpi=200)
    plt.close(fig)
    print("Published audited pilot report and figure")


if __name__ == "__main__":
    main()
