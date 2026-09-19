"""Build an auditable Markdown report, public dashboard, plot, and inspectable replays."""

import html
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.public.common import digest, read_json
from benchmarks.public.run import ROOT, get_inputs, trace_path
from rag_jev.generation import Answer
from rag_jev.models import Judgment, SelectRequest, Usage
from rag_jev.workbench import Comparison, ReplayBundle, ScoredRun, input_hash

LABELS = {"hotpotqa": "HotpotQA", "musique": "MuSiQue-Ans"}


def main():
    protocol, datasets = get_inputs()
    report = read_json(ROOT / "evaluation-report.json")
    frozen = read_json(ROOT / "frozen-policy.json")
    if report["protocol_hash"] != digest(protocol) or frozen["protocol_hash"] != digest(protocol):
        raise ValueError("Report/protocol mismatch")
    policy = frozen["policy"]
    name = policy["name"]
    if policy["scorer"] != "contextual":
        raise ValueError("This dashboard/export profile requires a contextual policy")
    assert all(d["summary"][name]["cases"] == 100 for d in report["datasets"].values())
    rows, deltas = [], []
    for dataset, result in report["datasets"].items():
        for branch, label in [
            ("all_context", "All context"),
            ("bm25_top5", "BM25 top 5"),
            (name, "Jev contextual"),
        ]:
            r = result["summary"][branch]
            rows.append(
                [
                    LABELS[dataset],
                    label,
                    f"{r['answer_em'] * 100:.2f}",
                    f"{r['answer_f1'] * 100:.2f}",
                    f"${r['api_cost_usd']:.6f}",
                    f"{r['support_recall'] * 100:.2f}%",
                ]
            )
        delta = result["comparisons"][name + "_vs_all_context"]
        deltas.append(
            f"{LABELS[dataset]}: F1 difference **{delta['delta'] * 100:+.2f} points**, "
            f"95% paired bootstrap interval **[{delta['ci95'][0] * 100:+.2f}, "
            f"{delta['ci95'][1] * 100:+.2f}]**; estimated API cost "
            f"**{delta['cost_savings_fraction'] * 100:.1f}% lower**."
        )
    headings = [
        "Dataset (100 questions)",
        "Pipeline",
        "Answer EM",
        "Answer F1",
        "API cost / 100",
        "Support recall",
    ]
    table = "| " + " | ".join(headings) + " |\n| " + " | ".join(["---"] * len(headings)) + " |\n"
    table += "\n".join("| " + " | ".join(row) + " |" for row in rows)
    assets = Path("src/rag_jev/static")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    colors = ["#797c76", "#b68b64", "#236248"]
    for j, (branch, label) in enumerate(
        [("all_context", "All context"), ("bm25_top5", "BM25 top 5"), (name, "Jev contextual")]
    ):
        x = [i + (j - 1) * 0.25 for i in range(2)]
        f1s = [r["summary"][branch]["answer_f1"] * 100 for r in report["datasets"].values()]
        costs = [r["summary"][branch]["api_cost_usd"] * 10 for r in report["datasets"].values()]
        for ax, values in zip(axes, [f1s, costs], strict=True):
            bars = ax.bar(x, values, width=0.23, color=colors[j], label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=9)
    for ax in axes:
        ax.set_xticks([0, 1], ["HotpotQA", "MuSiQue-Ans"])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Answer F1 (%)")
    axes[1].set_ylabel("Estimated API cost / 1,000 questions (USD)")
    axes[1].set_ylim(0, 0.8)
    axes[0].legend(loc="upper left", fontsize=8)
    figure.suptitle("Less context, measured outcomes", fontsize=19)
    figure.supxlabel(
        "100 questions per dataset · fixed candidates · Luna low reasoning\n"
        "Accuracy-difference 95% intervals include zero; no conclusive accuracy gain.",
        fontsize=9,
    )
    figure.savefig(assets / "benchmark-results.svg")
    figure.savefig(ROOT / "benchmark-results.png", dpi=200)
    plt.close(figure)
    html_rows = "".join(
        "<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in row) + "</tr>" for row in rows
    )
    dashboard = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>rag-jev · Public benchmark results</title><link rel="stylesheet" href="/assets/style.css">
</head><body><header class="masthead"><a href="/" class="wordmark">rag—jev</a>
<span>PUBLIC BENCHMARKS / 200 QUESTIONS</span><a href="/">Open workbench ↗</a></header>
<main><section class="intro"><div><p class="eyebrow">Measured, with the tradeoffs visible</p>
<h1>Less context.<br><em>Check the outcome.</em></h1></div><p class="intro-note">
Compare answer scores, estimated total API costs, and supporting evidence.
The differences and uncertainty below come from the frozen evaluation records.</p></section>
<p><a href="/assets/research-v2.html">Research round 2: 400 new questions ↗</a>
— better evidence retention, mixed answer results, and the measured cost tradeoff.</p>
<img src="/assets/benchmark-results.svg" width="1200" height="480"
style="width:100%;height:auto"
alt="Answer F1 and total API cost comparisons, 100 questions per dataset">
<div style="overflow-x:auto"><table style="width:100%;text-align:left;line-height:2"><thead><tr>"""
    dashboard += "".join("<th>" + html.escape(v) + "</th>" for v in headings)
    dashboard += "</tr></thead><tbody>" + html_rows + "</tbody></table></div>"
    dashboard += """<p>Fixed validation-set subsets; dataset-provided candidates; Jev 1.13.0;
GPT-5.6 Luna with low reasoning; one response per branch. Policy selected on 40 separate
development questions. Not full benchmark or leaderboard results.</p>
{{DELTAS}}
<p>Neither query expansion nor a modern neural reranker was evaluated.</p>
<details><summary>Method, sources, and what the costs include</summary>
<p>Official answer normalization and token F1, checked against the authors' metric functions
on all 760 development/evaluation predictions. Paired bootstrap: 2,000 resamples.
{{COMPLETENESS}}</p>
<p>Estimated API cost includes Jev and generation with reported cache discounts. Excludes
common retrieval, hosting and networking. Rates: Luna $0.20 input/$0.02 cached/$1.20 output
per million tokens; Jev $0.042 input/free output.</p>
<p><a href="https://github.com/hotpotqa/hotpot">HotpotQA</a> ·
<a href="https://github.com/StonyBrookNLP/musique">MuSiQue (CC BY 4.0)</a> ·
<a href="https://developers.openai.com/api/docs/models/gpt-5.6-luna">Luna pricing</a> ·
<a href="https://typesafe.ai/blog/introducing-system-one-models-and-jev">Jev pricing</a></p>
<p>MuSiQue came from the pinned dgslibisey/MuSiQue mirror. Model training overlap is unknown.
No modern neural reranker baseline was evaluated. Full protocol, source hashes, per-case
records and reproduction commands are in benchmarks/public in the project.</p></details>
<footer>Reproducible evaluation, not a promise for every RAG pipeline.
<span>September 18, 2026</span></footer>
</main></body></html>"""
    summaries = [
        value for result in report["datasets"].values() for value in result["summary"].values()
    ]
    errors = sum(value["errors"] for value in summaries)
    missing = sum(value["unknown_costs"] for value in summaries)
    truncated = sum(value["truncated_answers"] for value in summaries)
    gate = all(
        result["comparisons"][name + "_vs_all_context"]["success_gate"]
        for result in report["datasets"].values()
    )
    difference_html = "".join(
        "<p>" + html.escape(text.replace("**", "")) + "</p>" for text in deltas
    )
    difference_html += (
        "<p>Predeclared accuracy-and-cost gate: " + ("met" if gate else "not met") + ".</p>"
    )
    dashboard = dashboard.replace("{{DELTAS}}", difference_html).replace(
        "{{COMPLETENESS}}",
        f"Errors: {errors}; missing cost observations: {missing}; truncated answers: {truncated}.",
    )
    (assets / "benchmarks.html").write_text(dashboard)
    examples = {d: {"wins": [], "regressions": []} for d in datasets}
    for dataset, splits in datasets.items():
        for case in splits["evaluation"]:
            trace = read_json(trace_path("evaluation", case, "contextual"))
            base = read_json(trace_path("evaluation", case, "answer_all_context"))
            selected = read_json(trace_path("evaluation", case, "answer_" + name))
            req = SelectRequest(
                query=case["query"],
                documents=case["documents"],
                scoring_strategy="contextual",
                min_relevance=policy["threshold"],
            )
            record = ScoredRun(
                request=req,
                relevant_ids=case["relevant_ids"],
                source="live",
                created_at=datetime.now(UTC),
                input_hash=input_hash(req),
                prompt_version=req.prompt_version,
                judgments=[
                    Judgment(relevance=s, model=trace["models"][0]) for s in trace["scores"]
                ],
                scoring_elapsed_ms=trace["elapsed_ms"],
                scoring_usage=Usage(
                    input_tokens=trace["input_tokens"],
                    output_tokens=trace["output_tokens"],
                    completed_documents=len(req.documents),
                ),
            )
            comparison = Comparison(
                input_hash=record.input_hash,
                policy_version=req.policy.version,
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
                    "Benchmark short-answer prompt; citations not requested. "
                    "See benchmarks/public/protocol.json"
                ),
                cost_basis=(
                    "Published-rate estimate including Jev and reported cached-input "
                    "discounts; retrieval excluded"
                ),
            )
            bundle = ReplayBundle(record=record, policy=req.policy, comparison=comparison)
            path = ROOT / "replays" / dataset / (case["id"] + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(bundle.model_dump_json(indent=2) + "\n")
            delta = selected["metrics"]["f1"] - base["metrics"]["f1"]
            if delta:
                examples[dataset]["wins" if delta > 0 else "regressions"].append(case["id"])
    from benchmarks.public.common import write_json

    write_json(ROOT / "demo-cases.json", examples)
    print("Published dashboard, plot, and 200 validated replays.")
    print(examples)


if __name__ == "__main__":
    main()
