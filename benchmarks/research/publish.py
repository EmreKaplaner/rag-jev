"""Publish audited observations, uncertainty, blind review packet, and a portable bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.public.common import digest, read_json, write_json
from benchmarks.research.design import ROOT, answer_path, inputs, policies

LABELS = {"hotpotqa": "HotpotQA", "musique": "MuSiQue-Ans"}


def review_packet(protocol, datasets, frozen):
    label_file = ROOT / "review-labels.csv"
    if label_file.exists():
        existing = list(csv.DictReader(io.StringIO(label_file.read_text())))
        if any(
            any(row[k] for k in row if k not in {"case_id", "anonymous_answer"}) for row in existing
        ):
            raise ValueError("Preserve completed review labels; do not overwrite the audit packet")
    packet, key = [], {}
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "case_id",
            "anonymous_answer",
            "reviewer",
            "semantically_correct",
            "ambiguous_question_or_reference",
            "notes",
        ]
    )
    for dataset, splits in datasets.items():
        sample = sorted(
            splits["evaluation"], key=lambda c: digest([protocol["seed"], "audit", c["id"]])
        )[:20]
        for case in sample:
            arms = [
                p
                for p in policies("evaluation", protocol)
                if p["name"] in {"all_context", frozen["candidate"]["name"]}
            ]
            arms.sort(key=lambda p: digest([protocol["seed"], case["id"], p["name"], "blind"]))
            answers = {}
            for label, arm in zip(["A", "B"], arms, strict=True):
                answer = read_json(answer_path("evaluation", case, arm, 0))["answer"]
                answers[label] = {"text": answer["text"], "status": answer["status"]}
                key[case["id"] + "/" + label] = arm["name"]
                writer.writerow([case["id"], label, "", "", "", ""])
            packet.append(
                {
                    "case_id": case["id"],
                    "dataset": dataset,
                    "query": case["query"],
                    "reference_answers": case["answers"],
                    "full_candidate_evidence": case["documents"],
                    "answers": answers,
                }
            )
    write_json(
        ROOT / "blind-review.json",
        {
            "status": "unreviewed",
            "sample_rule": "20 hash-selected questions per dataset; always repetition 0",
            "instructions": "Judge semantic correctness against the question, references and "
            "full evidence. Mark ambiguous cases. This packet does not establish faithfulness "
            "to selected context. "
            "Keep review-key.json away from reviewers until labels are complete.",
            "cases": packet,
        },
    )
    write_json(ROOT / "review-key.json", key)
    (ROOT / "review-labels.csv").write_text(output.getvalue())


def main(bundle=False):
    protocol, datasets = inputs()
    frozen, report = read_json(ROOT / "frozen.json"), read_json(ROOT / "evaluation-report.json")
    verified = read_json(ROOT / "verification.json")
    assert report["protocol_hash"] == verified["protocol_hash"] == digest(protocol)
    assert verified["status"] == "passed"
    assert verified["evaluation_report_hash"] == digest(report)
    names = [p["name"] for p in policies("evaluation", protocol)]
    winner = frozen["candidate"]["name"]
    headings = [
        "Dataset",
        "Pipeline",
        "Answer F1",
        "EM",
        "All support retained",
        "Uncached API $ / 1,000",
        "CPU seconds / question",
        "Answer disagreement",
    ]
    rows, paragraphs = [], []
    for dataset, results in report["datasets"].items():
        for name in names:
            s = results["summary"][name]
            label = name + (" (frozen candidate)" if name == winner else "")
            rows.append(
                [
                    LABELS[dataset],
                    label,
                    f"{s['f1'] * 100:.2f}",
                    f"{s['em'] * 100:.2f}",
                    f"{s['all_support'] * 100:.1f}%",
                    f"{s['normalized_cost_per_1000_usd']:.4f}",
                    f"{s['local_cpu_seconds']:.3f}",
                    f"{s['answer_disagreement_fraction'] * 100:.1f}%",
                ]
            )
        p = results["primary"]
        paragraphs.append(
            f"{LABELS[dataset]}: frozen candidate versus all context, F1 difference "
            f"{p['delta'] * 100:+.2f} points, simultaneous 97.5% interval "
            f"[{p['ci'][0] * 100:+.2f}, {p['ci'][1] * 100:+.2f}]; Holm-adjusted "
            f"one-sided p={p['holm_p']:.4f}. Normalized API cost "
            f"{abs(p['normalized_savings_fraction']) * 100:.1f}% "
            f"{'lower' if p['normalized_savings_fraction'] >= 0 else 'higher'}."
        )
        ne = results["neural_descriptive"]
        paragraphs.append(
            f"{LABELS[dataset]}: candidate versus Ettin, descriptive F1 difference "
            f"{ne['delta'] * 100:+.2f} points, unadjusted 95% interval "
            f"[{ne['ci'][0] * 100:+.2f}, {ne['ci'][1] * 100:+.2f}]. "
            f"CPU-cost break-even: ${results['neural_cpu_break_even_usd_per_hour']:.4f} "
            "per CPU-hour, excluding other hosting costs. A negative break-even means "
            "Ettin generation alone already costs more than the candidate's API bill."
        )
        for effect, label in [
            ("lower_cutoff_main_effect", "lowering the cutoff, averaged over both orders"),
            ("ranking_main_effect", "relevance ordering, averaged over both cutoffs"),
        ]:
            contrast = results["factorial_descriptive"][effect]
            paragraphs.append(
                f"{LABELS[dataset]}: {label}, descriptive F1 difference "
                f"{contrast['delta'] * 100:+.2f} points; unadjusted 95% interval "
                f"[{contrast['ci'][0] * 100:+.2f}, {contrast['ci'][1] * 100:+.2f}]."
            )
        mix = ", ".join(
            f"{name}: {group['questions']}"
            for name, group in results["subgroups_descriptive"].items()
        )
        paragraphs.append(f"{LABELS[dataset]} evaluation composition — {mix}.")
    table = "| " + " | ".join(headings) + " |\n| " + " | ".join(["---"] * len(headings)) + " |\n"
    table += "\n".join("| " + " | ".join(r) + " |" for r in rows)
    verdict = "The predeclared accuracy-and-cost superiority gate is " + (
        "met." if report["superiority_gate"] else "not met."
    )
    total = sum(s["branches"] for d in report["datasets"].values() for s in d["summary"].values())
    errors = sum(s["errors"] for d in report["datasets"].values() for s in d["summary"].values())
    truncations = sum(
        s["truncated"] for d in report["datasets"].values() for s in d["summary"].values()
    )
    unknown = sum(
        s["unknown_usage"] for d in report["datasets"].values() for s in d["summary"].values()
    )
    method = (
        f"400 isolated evaluation questions, three samples per pipeline/question, {total:,} "
        "evaluation branches. The independent sample size is 200 questions per dataset, "
        "not the number of generations. All-context, a full cutoff/order factorial, and "
        "development-tuned Ettin are evaluated on the same questions. "
        f"Errors: {errors}; truncations: {truncations}; unknown-usage records: {unknown}."
    )
    cost_note = (
        "Costs are returned-usage estimates at uncached token rates, including Jev. "
        "Ettin's listed dollars cover GENERATION ONLY; CPU cost is separate and its total "
        "monetary cost is unknown. Observed cache-discounted API costs are in the JSON report. "
        "Common retrieval, application hosting and network costs are excluded."
    )
    limits = (
        "Strict source-component isolation changes the validation population. These are not "
        "full benchmark or corpus-retrieval results. Answer F1 is token overlap, not semantic "
        "or citation correctness. Human adjudication has not been completed. Proprietary-model "
        "training contamination and weights cannot be independently verified. Ettin's published "
        "checkpoint-selection procedure uses NanoBEIR, including HotpotQA and SciFact; "
        "disjointness from those selection examples is not established. Scoring is fixed "
        "per question; repeats measure conditional generation variance. No query expansion claim."
        " Secondary findings cannot be used to replace the frozen candidate and claim a new "
        "held-out win. In particular, the small higher-hop MuSiQue subgroups cannot support "
        "precise higher-hop accuracy claims."
    )
    inference = (
        "Primary intervals use question-paired bootstrap (10,000 draws) with simultaneous "
        "Bonferroni coverage for two endpoints. One-sided sign-flip tests use 20,000 draws "
        "and Holm correction. Secondary factorial, neural and subgroup intervals are "
        "descriptive. A nonsignificant difference does not establish equivalence."
    )
    assets = Path("src/rag_jev/static")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for i, (_dataset, result) in enumerate(report["datasets"].items()):
        p = result["primary"]
        point, low, high = p["delta"] * 100, p["ci"][0] * 100, p["ci"][1] * 100
        axes[0].errorbar(
            point, i, xerr=[[point - low], [high - point]], fmt="o", capsize=5, color="#236248"
        )
        axes[0].annotate(
            f"{point:+.2f} [{low:+.2f}, {high:+.2f}]",
            (point, i),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
        for j, (key, _label) in enumerate(
            [
                ("lower_cutoff_main_effect", "Lower cutoff"),
                ("ranking_main_effect", "Relevance order"),
                ("interaction", "Interaction"),
            ]
        ):
            p = result["factorial_descriptive"][key]
            point, low, high = p["delta"] * 100, p["ci"][0] * 100, p["ci"][1] * 100
            axes[1].errorbar(
                point,
                i * 3 + j,
                xerr=[[point - low], [high - point]],
                fmt="o",
                capsize=4,
                color="#236248" if i == 0 else "#b68b64",
            )
    axes[0].set_yticks([0, 1], list(LABELS.values()))
    axes[0].set_ylim(-0.5, 1.5)
    axes[0].set_title("Primary: frozen candidate versus all context\nSimultaneous 97.5% intervals")
    axes[1].set_yticks(
        range(6),
        [
            f"{d} · {effect}"
            for d in LABELS.values()
            for effect in ["Lower cutoff", "Relevance order", "Interaction"]
        ],
    )
    axes[1].set_title("Secondary factorial contrasts\nDescriptive 95% intervals")
    for axis in axes:
        axis.axvline(0, color="#797c76", linestyle="--", linewidth=1)
        axis.set_xlabel("Answer F1 difference (percentage points)")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Separate the effects. Show the uncertainty.", fontsize=18)
    figure.supxlabel(
        "200 isolated questions per dataset · 3 answer samples per arm/question\n" + verdict,
        fontsize=9,
    )
    figure.savefig(assets / "research-v3-results.svg")
    figure.savefig(ROOT / "research-v3-results.png", dpi=200)
    plt.close(figure)
    paragraphs_all = [verdict, method, *paragraphs, inference, cost_note, limits]
    markdown = "# Controlled research results — round 3\n\n" + verdict + "\n\n" + table + "\n\n"
    markdown += "\n\n".join([method, *paragraphs, inference, cost_note, limits])
    markdown += "\n\n[Design, sources and reproduction](README.md). "
    markdown += "Raw records, audit packets and integrity manifest: `artifacts/research-v3/`.\n"
    Path("benchmarks/research/RESULTS.md").write_text(markdown)
    html_table = (
        "<div style='overflow-x:auto'>"
        "<table style='width:100%;text-align:left;line-height:2'><thead><tr>"
    )
    html_table += (
        "".join("<th>" + html.escape(h) + "</th>" for h in headings) + "</tr></thead><tbody>"
    )
    html_table += "".join(
        "<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in r) + "</tr>" for r in rows
    )
    html_table += "</tbody></table></div>"
    page = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>rag-jev · Controlled research</title><link rel="stylesheet" href="/assets/style.css">
</head><body><header class="masthead"><a class="wordmark" href="/">rag—jev</a>
<span>CONTROLLED STUDY / ROUND 3</span>
<a href="/assets/research-v2.html">Earlier results ↗</a></header>
<main><section class="intro"><div><p class="eyebrow">A frozen test with stronger controls</p>
<h1>Separate the effects.<br><em>Show the uncertainty.</em></h1></div>
<p class="intro-note">400 isolated questions, repeated answers, a neural baseline,
and every pipeline measured against the same evidence.</p></section>
<img src="/assets/research-v3-results.svg" width="1300" height="500" style="width:100%;height:auto"
alt="Primary paired F1 confidence intervals and secondary factorial contrasts">
"""
        + html_table
        + "".join("<p>" + html.escape(p) + "</p>" for p in paragraphs_all)
        + """
<p><a href="/assets/research-v3-results.json" download>Download measured results</a> ·
<a href="/assets/research-v3-protocol.json" download>Download frozen protocol</a> ·
<a href="/assets/research-v3-verification.json" download>Download integrity checks</a></p>
<p><a href="/assets/retrieval-v1.html">Separate corpus-retrieval check:
all 300 SciFact test queries ↗</a></p>
<footer>Locally predeclared and reproducible; not externally preregistered or peer reviewed.
<span>September 18, 2026</span></footer></main></body></html>"""
    )
    (assets / "research-v3.html").write_text(page)
    for name, data in [("results", report), ("protocol", protocol), ("verification", verified)]:
        write_json(assets / f"research-v3-{name}.json", data)
    review_packet(protocol, datasets, frozen)
    write_json("benchmarks/research/verification.json", verified)
    if bundle:
        destination = ROOT / "reproduction.tar.gz"
        with tarfile.open(destination, "w:gz") as archive:
            # Enumerate allowed research/product files. Never traverse .env, caches or keys.
            files = set()
            for directory in [
                ROOT,
                Path("artifacts/retrieval-v1"),
                Path("benchmarks/research"),
                Path("benchmarks/public"),
                Path("src/rag_jev"),
                Path("examples"),
                Path("docs"),
                Path("tests"),
                Path("scripts"),
                Path("clients/typescript"),
                Path("integrations/dify"),
            ]:
                for file in directory.rglob("*"):
                    if (
                        file.is_file()
                        and file.suffix
                        in {
                            ".py",
                            ".json",
                            ".md",
                            ".csv",
                            ".html",
                            ".css",
                            ".js",
                            ".mjs",
                            ".svg",
                            ".ts",
                            ".toml",
                            ".yaml",
                            ".yml",
                            ".lock",
                            ".woff2",
                            ".png",
                            ".txt",
                        }
                        and not set(file.parts).intersection(
                            {"__pycache__", ".venv", "node_modules", "dist"}
                        )
                    ):
                        files.add(file)
            files.update(
                Path(f)
                for f in ["pyproject.toml", "uv.lock", "README.md", "MVP_SCOPE.md", "Makefile"]
            )
            files.update(
                Path(s["local_path"])
                for s in read_json("benchmarks/research/metric-sources.json").values()
            )
            for file in sorted(files):
                archive.add(file, arcname=str(file), recursive=False)
        checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
        (ROOT / "reproduction.sha256").write_text(checksum + "  reproduction.tar.gz\n")
        print("Bundle:", destination, "SHA256", checksum)
    print("Published audited results, uncertainty chart, and UNREVIEWED blind audit packet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="store_true")
    main(parser.parse_args().bundle)
