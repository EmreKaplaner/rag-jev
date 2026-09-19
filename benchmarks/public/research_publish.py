"""Publish completed round-2 observations and validated production-compatible replays."""

import html
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.public.common import digest, read_json, write_json
from benchmarks.public.research import ROOT, inputs, path
from rag_jev.generation import Answer
from rag_jev.models import Judgment, SelectRequest, Usage
from rag_jev.workbench import Comparison, ReplayBundle, ScoredRun, input_hash

LABELS = {"hotpotqa": "HotpotQA", "musique": "MuSiQue-Ans"}


def export_replays(datasets, policy):
    if policy["scorer"] != "contextual":
        raise ValueError("Only the production contextual scorer can be exported")
    manifest = {dataset: {"wins_vs_old": [], "regressions_vs_old": []} for dataset in datasets}
    for dataset, splits in datasets.items():
        for case in splits["evaluation"]:
            trace = read_json(path("evaluation", case, "contextual"))
            base = read_json(path("evaluation", case, "answer_all_context"))
            old = read_json(path("evaluation", case, "answer_contextual_filter_0.35"))
            selected = read_json(path("evaluation", case, "answer_" + policy["name"]))
            request = SelectRequest(
                query=case["query"],
                documents=case["documents"],
                scoring_strategy="contextual",
                mode="filter_and_rerank" if policy.get("order") == "relevance" else "filter",
                min_relevance=policy["threshold"],
            )
            record = ScoredRun(
                request=request,
                relevant_ids=case["relevant_ids"],
                source="live",
                created_at=datetime.now(UTC),
                input_hash=input_hash(request),
                prompt_version=request.prompt_version,
                judgments=[
                    Judgment(relevance=s, model=trace["models"][0]) for s in trace["scores"]
                ],
                scoring_elapsed_ms=trace["elapsed_ms"],
                scoring_usage=Usage(
                    input_tokens=trace["input_tokens"],
                    output_tokens=trace["output_tokens"],
                    completed_documents=len(request.documents),
                ),
            )
            comparison = Comparison(
                input_hash=record.input_hash,
                policy_version=request.policy.version,
                baseline=Answer.model_validate(base["answer"]),
                selected=Answer.model_validate(selected["answer"]),
                baseline_ids=base["selected_ids"],
                selected_ids=selected["selected_ids"],
                selection_source="live",
                baseline_stage_ms=base["stage_ms"],
                selected_stage_ms=selected["stage_ms"],
                selection_original_ms=trace["elapsed_ms"],
                baseline_cost_usd=base["cost_usd"],
                selected_cost_usd=selected["cost_usd"],
                selection_cost_usd=selected["scoring_cost_usd"],
                citation_check=(
                    "Research short-answer prompt; citations not requested. See RESEARCH.md."
                ),
                cost_basis=(
                    "Published-rate estimate including scoring and reported cached input; "
                    "retrieval excluded."
                ),
            )
            bundle = ReplayBundle(record=record, policy=request.policy, comparison=comparison)
            destination = ROOT / "replays" / dataset / (case["id"] + ".json")
            write_json(destination, bundle.model_dump(mode="json"))
            delta = selected["metrics"]["f1"] - old["metrics"]["f1"]
            if delta:
                manifest[dataset]["wins_vs_old" if delta > 0 else "regressions_vs_old"].append(
                    case["id"]
                )
    write_json(ROOT / "demo-cases.json", manifest)


def main():
    protocol, datasets = inputs()
    report = read_json(ROOT / "evaluation-report.json")
    frozen = read_json(ROOT / "frozen-policy.json")
    if report["protocol_hash"] != digest(protocol) or frozen["protocol_hash"] != digest(protocol):
        raise ValueError("Report does not match frozen protocol")
    policy = frozen["policy"]
    branches = [
        ("all_context", "All context"),
        ("contextual_filter_0.35", "Old Jev profile"),
        (policy["name"], "Research v2"),
    ]
    headers = [
        "Dataset (200 questions)",
        "Pipeline",
        "Answer EM",
        "Answer F1",
        "API cost / 100",
        "Support recall",
        "All support retained",
    ]
    rows, comparisons = [], []
    for dataset, result in report["datasets"].items():
        for branch, label in branches:
            s = result["summary"][branch]
            if s["cases"] != len(datasets[dataset]["evaluation"]):
                raise ValueError("Incomplete evaluation")
            rows.append(
                [
                    LABELS[dataset],
                    label,
                    f"{s['answer_em'] * 100:.2f}",
                    f"{s['answer_f1'] * 100:.2f}",
                    f"${s['api_cost_usd'] * 100 / s['cases']:.6f}",
                    f"{s['support_recall'] * 100:.2f}%",
                    f"{s['all_support_retained'] * 100:.2f}%",
                ]
            )
        for baseline, label in branches[:2]:
            c = result["comparisons"][policy["name"] + "_vs_" + baseline]
            comparisons.append(
                f"{LABELS[dataset]}, research v2 versus {label.lower()}: "
                f"F1 {c['delta'] * 100:+.2f} points; 95% paired interval "
                f"[{c['ci95'][0] * 100:+.2f}, {c['ci95'][1] * 100:+.2f}]. "
                f"Estimated API cost {abs(c['cost_savings_fraction']) * 100:.1f}% "
                f"{'lower' if c['cost_savings_fraction'] >= 0 else 'higher'}."
            )
    table = "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
    table += "\n".join("| " + " | ".join(row) + " |" for row in rows)
    summaries = [s for d in report["datasets"].values() for s in d["summary"].values()]
    completeness = "; ".join(
        f"{sum(s[k] for s in summaries)} {label}"
        for k, label in [
            ("errors", "errors"),
            ("unknown_costs", "unknown costs"),
            ("unknown_cache_usage", "unknown cache-usage records"),
            ("truncated_answers", "truncations"),
        ]
    )
    gate = all(
        d["comparisons"][policy["name"] + "_vs_all_context"]["accuracy_and_cost_gate"]
        for d in report["datasets"].values()
    )
    verdict = "The predeclared accuracy-and-cost gate versus all context is " + (
        "met." if gate else "not met."
    )
    method = (
        "400 fresh validation questions, 200 per dataset, disjoint from all 240 development "
        "questions. Supplied candidate passages; Jev 1.13.0 and GPT-5.6 Luna, low reasoning. "
        "At most one generation per branch; 1,200 evaluation branches. "
        "Costs include Jev and generation with reported cache discounts; common retrieval, "
        "hosting and networking excluded. These are subsets, not full benchmark "
        "or leaderboard results."
    )
    limits = (
        "Answer F1 measures normalized token overlap, so wording changes can affect scores. "
        "Supporting-passage retention is a separate diagnostic. Model training overlap is unknown. "
        "No query expansion, full-corpus retrieval, or modern neural reranker comparison "
        "was tested. "
        "MuSiQue uses the pinned community mirror described in the protocol."
    )
    assets = Path("src/rag_jev/static")
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for j, (branch, label) in enumerate(branches):
        x = [i + (j - 1) * 0.25 for i in range(2)]
        summaries_j = [d["summary"][branch] for d in report["datasets"].values()]
        values = [
            [s["answer_f1"] * 100 for s in summaries_j],
            [s["api_cost_usd"] * 1000 / s["cases"] for s in summaries_j],
            [s["all_support_retained"] * 100 for s in summaries_j],
        ]
        for axis, metric in zip(axes, values, strict=True):
            bars = axis.bar(
                x, metric, width=0.23, color=["#797c76", "#b68b64", "#236248"][j], label=label
            )
            axis.bar_label(bars, fmt="%.2f", fontsize=8)
    for axis, label in zip(
        axes,
        [
            "Answer F1 (%)",
            "Estimated API cost / 1,000 (USD)",
            "Questions retaining all support (%)",
        ],
        strict=True,
    ):
        axis.set_xticks([0, 1], ["HotpotQA", "MuSiQue-Ans"])
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, 108)
    axes[2].set_ylim(0, 108)
    axes[0].legend(fontsize=8, loc="upper left")
    figure.suptitle("Preserve the evidence. Measure the answer.", fontsize=19)
    figure.supxlabel(
        "200 fresh questions per dataset · fixed candidates · Luna low reasoning\n"
        "Answer-accuracy gains remain unproven; v2 costs more than the old filter.",
        fontsize=9,
    )
    figure.savefig(assets / "research-v2-results.svg")
    figure.savefig(ROOT / "research-v2-results.png", dpi=200)
    plt.close(figure)
    html_table = (
        "<div style='overflow-x:auto'>"
        "<table style='width:100%;text-align:left;line-height:2'><thead><tr>"
    )
    html_table += (
        "".join("<th>" + html.escape(h) + "</th>" for h in headers) + "</tr></thead><tbody>"
    )
    html_table += "".join(
        "<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in row) + "</tr>" for row in rows
    )
    html_table += "</tbody></table></div>"
    paragraphs = "".join(
        "<p>" + html.escape(s) + "</p>"
        for s in [verdict, *comparisons, method, completeness, limits]
    )
    dashboard = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>rag-jev · Research round 2</title><link rel="stylesheet" href="/assets/style.css">
</head><body><header class="masthead"><a href="/" class="wordmark">rag—jev</a>
<span>RESEARCH / 400 NEW QUESTIONS</span><a href="/benchmarks">Round 1 results ↗</a></header>
<main><section class="intro"><div><p class="eyebrow">Evidence retention, tested again</p>
<h1>Keep the links.<br><em>Measure the tradeoff.</em></h1></div>
<p class="intro-note">A lower cutoff and relevance ordering, chosen on development data
and evaluated once on fresh questions.</p></section>
<img src="/assets/research-v2-results.svg" width="1500" height="500"
style="width:100%;height:auto" alt="Answer F1, total API cost and complete evidence retention">
"""
        + html_table
        + paragraphs
        + """
<p><a href="/assets/research-v2-results.json" download>Download measured results</a> ·
<a href="/assets/research-v2-protocol.json" download>Download frozen protocol</a> ·
<a href="/">Open workbench</a></p>
<p>Profile: passages together, filter &amp; rerank, cutoff 0.20, no top-N cap.
Test on your own retrieval results before adopting it.</p>
<footer>Reproducible experiments; all wins and regressions retained.
<span>September 18, 2026</span></footer></main></body></html>"""
    )
    (assets / "research-v2.html").write_text(dashboard)
    write_json(assets / "research-v2-results.json", report)
    write_json(assets / "research-v2-protocol.json", protocol)
    markdown = "# Research round 2 results — September 18, 2026\n\n" + verdict + "\n\n" + table
    markdown += "\n\n" + "\n\n".join(comparisons + [method, completeness + ".", limits])
    markdown += (
        "\n\n[Protocol and reproduction](RESEARCH.md). Raw records: `artifacts/research-v2/`.\n"
    )
    (Path("benchmarks/public") / "RESEARCH_RESULTS.md").write_text(markdown)
    export_replays(datasets, policy)
    print("Published research dashboard, report, plot, and 400 validated replays.")


if __name__ == "__main__":
    main()
