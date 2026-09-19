"""Publish a study index and evidence-linked visuals without running inference."""

import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/rag_jev/static"
REPO = "https://github.com/EmreKaplaner/rag-jev/blob/main/"


def charts():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.hashsalt": "rag-jev",
            "figure.facecolor": "#f5f1e7",
            "axes.facecolor": "#f5f1e7",
            "text.color": "#253b36",
            "axes.labelcolor": "#253b36",
            "xtick.color": "#253b36",
            "ytick.color": "#253b36",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(fig, name):
        target = ASSETS / name
        fig.savefig(target, metadata={"Date": None}, bbox_inches="tight")
        target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
        plt.close(fig)

    retrieval = json.loads((ASSETS / "retrieval-v1-results.json").read_text())
    fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
    names = ["bm25", "ettin", "jev"]
    values = [retrieval["summary"][n]["ndcg10"] * 100 for n in names]
    bars = ax.barh(["BM25", "Ettin", "Jev"], values, color=["#87998e", "#253b36", "#b84f2d"])
    ax.bar_label(bars, fmt="%.2f", padding=6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("nDCG@10 (%) · same top20 candidate pool")
    ax.set_title("SciFact / full corpus, 300 test queries", loc="left", pad=18)
    ax.invert_yaxis()
    save(fig, "retrieval-overview.svg")

    robustness = json.loads((ROOT / "benchmarks/robustness/results.json").read_text())
    rows = robustness["conditions"]
    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r["p50_ms"] for r in rows], width=0.35, color="#253b36", label="Median")
    ax.bar(x + 0.18, [r["p95_ms"] for r in rows], width=0.35, color="#b84f2d", label="p95")
    ax.set_xticks(x, [f"{r['strategy']}\nconcurrency {r['concurrency']}" for r in rows], fontsize=9)
    ax.set_ylabel("Observed selection latency (ms)")
    ax.set_title("Robustness / 18 synthetic cases per condition, not a production SLA", loc="left")
    ax.legend(frameon=False)
    save(fig, "robustness-overview.svg")

    # Parse the published source table, rather than maintain a second set of measurements.
    source = (ROOT / "benchmarks/pilot/RESULTS.md").read_text()
    fields = ["Estimated passage tokens (`cl100k_base`)", "Estimated total stage cost"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, field, title in zip(
        axes, fields, ["Estimated passage tokens", "Estimated stage cost (USD)"], strict=True
    ):
        row = next(line for line in source.splitlines() if line.startswith(f"| {field} |"))
        values = [
            float(value.strip().replace(",", "").replace("$", "")) for value in row.split("|")[2:4]
        ]
        bars = ax.bar(["Original", "Jev filter"], values, color=["#253b36", "#b84f2d"])
        ax.bar_label(bars, fmt="%.0f" if "tokens" in title else "%.6f", padding=4)
        ax.set_ylim(0, max(values) * 1.25)
        ax.set_title(title, loc="left")
    fig.suptitle(
        "First live pilot / ten questions · less context did not mean lower cost", fontsize=12
    )
    save(fig, "pilot-overview.svg")
    shutil.copyfile(ROOT / "benchmarks/ecosystem/pilot.png", ASSETS / "ecosystem-pilot.png")


def main():
    studies = json.loads((ROOT / "docs/studies.json").read_text())
    charts()
    esc = html.escape
    entries, md = (
        [],
        [
            "# The rag-jev research notebook",
            "",
            "Every study, including regressions and inconclusive results. "
            "These studies use different "
            "questions and protocols; their sample counts and scores must not be pooled.",
            "",
            "Start the app and open `/research` for the visual notebook. "
            "[Deploy the service](DEPLOYMENT.md) · [Try your pipeline](REAL_RAG.md)",
            "",
        ],
    )
    for i, s in enumerate(studies):
        links = []
        for label, field in [
            ("Full report", "report"),
            ("Reproduce", "methods"),
            ("Protocol", "protocol"),
            ("Data", "data"),
        ]:
            if field in s:
                assert (ROOT / s[field]).is_file(), s[field]
                links.append(f'<a href="{REPO}{esc(s[field])}">{label} ↗</a>')
        if "page" in s:
            links.insert(0, f'<a href="{s["page"]}">Open dashboard →</a>')
        links.append(f'<a href="/assets/{s["visual"]}">Full-size chart ↗</a>')
        assert (ASSETS / s["visual"]).is_file()
        entries.append(f'''<article class="study" id="{s["id"]}">
          <div class="study-number">{i + 1:02d}</div><div class="study-body">
          <p class="eyebrow">{esc(s["scope"])}</p><h2>{esc(s["title"])}</h2>
          <p>{esc(s["summary"])}</p>
          <figure><img src="/assets/{s["visual"]}" alt="{esc(s["visual_alt"])}" loading="lazy">
          <figcaption>{esc(s["caveat"])}</figcaption></figure>
          <nav class="study-links" aria-label="{esc(s["title"])} resources">{" ".join(links)}</nav>
          </div></article>''')
        md += [
            f"## {i + 1:02d} · {s['title']}",
            "",
            f"**{s['scope']}**",
            "",
            s["summary"],
            "",
            f"![{s['visual_alt']}](../src/rag_jev/static/{s['visual']})",
            "",
            s["caveat"],
            "",
            " · ".join(
                f"[{label}](../{s[field]})"
                for label, field in [
                    ("Report", "report"),
                    ("Methods", "methods"),
                    ("Protocol", "protocol"),
                    ("Data", "data"),
                ]
                if field in s
            ),
            "",
        ]
    md += [
        "## What remains",
        "",
        "Full-Wikipedia NQ/TriviaQA/HotpotQA, full BERGEN generation, "
        "independent human adjudication and a production shadow trial remain pending. "
        "The supplied-page and supplied-candidate studies do not substitute for those evaluations.",
        "",
    ]
    (ROOT / "docs/RESEARCH.md").write_text("\n".join(md))
    (ASSETS / "research.html").write_text(
        """<!doctype html><html lang="en"><head>
      <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
      <title>rag-jev · Research notebook</title><link rel="stylesheet" href="/assets/style.css">
      <meta name="description"
        content="Every rag-jev study: retrieval, quality, cost, robustness and the tradeoffs.">
      </head><body><header class="masthead"><a class="wordmark" href="/">rag—jev</a>
      <span>THE RESEARCH NOTEBOOK</span><a href="/">Open workbench ↗</a></header>
      <main class="notebook"><section class="notebook-intro"><div>
      <p class="eyebrow">Evidence before enthusiasm</p>
      <h1>Keep the useful bits.<br><em>Show all the evidence.</em></h1>
      <p class="notebook-lede">A field notebook for context selection. Seven studies of retrieval,
      answer quality, cost, and where things go wrong. Open methods. Visible tradeoffs.</p>
      <a class="text-cta" href="#ecosystem">Start with the latest study ↓</a></div>
      <img class="notebook-mascot" src="/assets/brand/squirrel.svg" alt="" width="220" height="220">
      </section><div class="notebook-note"><strong>Read the scope with the score.</strong>
      These are distinct experiments, not a combined leaderboard. Small samples, model-judge bias
      and inconclusive intervals remain visible.</div>
      """
        + "\n".join(entries)
        + '''<section class="next-study"><p class="eyebrow">The next page</p>
      <h2>From public evidence to your pipeline.</h2><p>Full-Wikipedia runs,
      independent human review
      and production shadow trials are still pending. Start with your own retrieved passages and
      compare against your current reranker.</p><a href="'''
        + REPO
        + '''docs/DEPLOYMENT.md">Remote deployment guide ↗</a>
      <a href="'''
        + REPO
        + """docs/REAL_RAG.md">Integration guide ↗</a></section>
      <footer>rag-jev · MIT-licensed integration · Powered by the external TypeSafe Jev API.
      Studies are exploratory and not peer reviewed.</footer></main></body></html>"""
    )

    # Apply the same identity to the existing study dashboards, preserving their evidence.
    for path in ASSETS.glob("*.html"):
        text = path.read_text()
        if "/assets/notebook.css" not in text:
            text = text.replace(
                "</head>", '<link rel="stylesheet" href="/assets/notebook.css"></head>'
            )
        if 'rel="icon"' not in text:
            text = text.replace(
                "</head>",
                '<link rel="icon" type="image/svg+xml" href="/assets/brand/favicon.svg">'
                '<meta name="theme-color" content="#253b36"></head>',
            )
        if 'rel="apple-touch-icon"' not in text:
            text = text.replace(
                "</head>",
                '<link rel="apple-touch-icon" sizes="180x180" '
                'href="/assets/brand/apple-touch-icon.png"></head>',
            )
        text = re.sub(
            r'(<a\b[^>]*class="wordmark"[^>]*>).*?(</a\s*>)',
            r'\1<img src="/assets/brand/squirrel.svg" width="38" height="38" alt="">'
            r'<span class="brand-name">rag<span>—</span>jev</span>\2',
            text,
            flags=re.S,
        )
        if path.name == "index.html":
            text = text.replace(
                'href="/benchmarks">Public benchmarks ↗', 'href="/research">Research notebook ↗'
            )
        elif path.name != "research.html" and 'class="research-nav"' not in text:
            text = text.replace(
                "</header>",
                '</header><nav class="research-nav" aria-label="Research navigation">'
                '<a href="/research">← All seven studies</a><a href="/">Workbench</a></nav>',
            )
        path.write_text(text)


if __name__ == "__main__":
    main()
