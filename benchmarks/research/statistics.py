"""Question-clustered analysis: repeats never masquerade as extra independent questions."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from benchmarks.public.common import digest, normalize, read_json, write_json
from benchmarks.research.design import (
    CANDIDATES,
    NEURAL,
    ROOT,
    answer_path,
    inputs,
    policies,
)


def paired(left, right, *, confidence=0.95, draws=10000, seed=31918, permutations=20000):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2 or not np.isfinite(a + b).all():
        raise ValueError("At least two finite paired question observations required")
    differences = b - a
    rng = np.random.default_rng(seed)
    boot = differences[rng.integers(len(a), size=(draws, len(a)))].mean(axis=1)
    alpha = (1 - confidence) / 2
    interval = np.quantile(boot, [alpha, 1 - alpha]).tolist()
    # One-sided sign-flip randomization, matched by question; +1 Monte Carlo correction.
    signs = rng.choice([-1, 1], size=(permutations, len(a)))
    simulated = (signs * differences).mean(axis=1)
    p_value = (1 + int(np.sum(simulated >= differences.mean() - 1e-12))) / (permutations + 1)
    return {
        "delta": float(differences.mean()),
        "ci": interval,
        "confidence": confidence,
        "p_one_sided": p_value,
        "n_questions": len(a),
        "bootstrap_draws": draws,
        "randomization_draws": permutations,
    }


def holm(values):
    adjusted, floor = {}, 0.0
    for rank, (key, p_value) in enumerate(sorted(values.items(), key=lambda x: x[1])):
        floor = max(floor, min(1.0, (len(values) - rank) * p_value))
        adjusted[key] = floor
    return adjusted


def summarize(groups):
    """Each group is all repeats of one question in fixed order."""
    if not groups or any(len(g) != len(groups[0]) for g in groups):
        raise ValueError("Incomplete repeat groups")
    rows = [r for g in groups for r in g]
    repeats = len(groups[0])
    question = {
        key: [float(np.mean([r[key] for r in g])) for g in groups]
        for key in [
            "f1",
            "em",
            "api_cost_usd",
            "normalized_api_cost_usd",
            "support_recall",
            "support_precision",
            "all_support",
            "ndcg5",
            "ndcg10",
            "local_cpu_seconds",
        ]
    }
    summary = {key: float(np.mean(values)) for key, values in question.items()}
    summary.update(
        {
            "questions": len(groups),
            "repeats": repeats,
            "branches": len(rows),
            "errors": sum(r["answer"]["status"] == "error" for r in rows),
            "empty_contexts": sum(r["answer"]["status"] == "no_context" for r in rows),
            "truncated": sum(r["answer"].get("finish_reason") == "length" for r in rows),
            "unknown_usage": sum(r["unknown_usage"] for r in rows),
            "within_question_f1_sd": float(
                np.sqrt(np.mean([np.var([r["f1"] for r in g], ddof=1) for g in groups]))
            ),
            "answer_disagreement_fraction": float(
                np.mean([len({normalize(r["answer"]["text"]) for r in g}) > 1 for g in groups])
            ),
            "replicate_f1": [float(np.mean([g[i]["f1"] for g in groups])) for i in range(repeats)],
            "median_stage_ms": float(np.median([r["stage_ms"] for r in rows])),
            "p95_stage_ms": float(np.quantile([r["stage_ms"] for r in rows], 0.95)),
            "normalized_cost_per_1000_usd": summary["normalized_api_cost_usd"] * 1000,
            "observed_api_cost_per_1000_usd": summary["api_cost_usd"] * 1000,
        }
    )
    return summary, question


def load_rows(split, protocol, datasets, arms):
    from benchmarks.research.runner import fingerprint

    results = {}
    for dataset, splits in datasets.items():
        results[dataset] = {}
        for policy in arms:
            groups = []
            for case in splits[split]:
                group = []
                for rep in range(protocol["repetitions"][split]):
                    record = read_json(answer_path(split, case, policy, rep))
                    if record["fingerprint"] != fingerprint(
                        case, policy, rep, protocol, record["selected_ids"]
                    ):
                        raise ValueError("Answer provenance mismatch")
                    group.append(record)
                groups.append(group)
            results[dataset][policy["name"]] = groups
    return results


def analyze(split, freeze=False):
    protocol, datasets = inputs()
    arms = policies(split, protocol)
    all_rows = load_rows(split, protocol, datasets, arms)
    report = {
        "protocol_hash": digest(protocol),
        "split": split,
        "datasets": {},
        "inference_unit": "question mean across repetitions",
    }
    questions = {}
    stats = protocol["statistics"]

    def compare(a, b, confidence=0.95):
        return paired(
            a,
            b,
            confidence=confidence,
            draws=stats["bootstrap_draws"],
            permutations=stats["randomization_draws"],
            seed=stats["seed"],
        )

    for dataset, arms_rows in all_rows.items():
        summaries, questions[dataset] = {}, {}
        for name, groups in arms_rows.items():
            summaries[name], questions[dataset][name] = summarize(groups)
        report["datasets"][dataset] = {"summary": summaries}
    if split == "development":
        if any(
            s[k]
            for d in report["datasets"].values()
            for s in d["summary"].values()
            for k in ["errors", "truncated", "unknown_usage"]
        ):
            raise ValueError("Development incomplete or degraded; retain results, do not select")
        summaries = [d["summary"] for d in report["datasets"].values()]
        eligible = [
            p
            for p in CANDIDATES
            if all(
                d[p["name"]]["normalized_api_cost_usd"]
                < d["all_context"]["normalized_api_cost_usd"]
                for d in summaries
            )
        ]

        def rank(p):
            return (
                -float(np.mean([d[p["name"]]["f1"] for d in summaries])),
                float(np.mean([d[p["name"]]["normalized_api_cost_usd"] for d in summaries])),
                p["name"],
            )

        winner = min(eligible or CANDIDATES, key=rank)
        neural = min(NEURAL, key=lambda p: (rank(p)[0], p["top_k"]))
        report["selection"] = {"candidate": winner, "neural": neural, "eligible": bool(eligible)}
        power = {}
        for dataset, q in questions.items():
            difference = np.asarray(q[winner["name"]]["f1"]) - q["all_context"]["f1"]
            n = protocol["sample"]["evaluation_per_dataset"]
            power[dataset] = {
                "planned_questions": n,
                "approx_80pct_power_mde_f1": float(
                    (NormalDist().inv_cdf(0.9875) + NormalDist().inv_cdf(0.8))
                    * np.std(difference, ddof=1)
                    / np.sqrt(n)
                ),
                "note": "Normal approximation using development paired SD; not a post-hoc "
                "power claim.",
            }
        report["power_planning"] = power
        write_json(ROOT / "development-report.json", report)
        if freeze:
            if (ROOT / "frozen.json").exists():
                raise ValueError("Do not overwrite a frozen policy")
            frozen = {
                **report["selection"],
                "protocol_hash": digest(protocol),
                "development_report_hash": digest(report),
                "power_planning": power,
            }
            write_json(ROOT / "frozen.json", frozen)
            write_json("benchmarks/research/frozen.json", frozen)
            print("FROZEN", winner, "NEURAL", neural, "POWER", power)
        return report
    frozen = read_json(ROOT / "frozen.json")
    winner, neural = frozen["candidate"]["name"], frozen["neural"]["name"]
    p_values = {}
    for dataset, q in questions.items():
        primary = compare(q["all_context"]["f1"], q[winner]["f1"], confidence=stats["primary_ci"])
        primary["normalized_cost"] = compare(
            q["all_context"]["normalized_api_cost_usd"],
            q[winner]["normalized_api_cost_usd"],
            confidence=stats["primary_ci"],
        )
        primary["normalized_savings_fraction"] = 1 - float(
            np.mean(q[winner]["normalized_api_cost_usd"])
        ) / float(np.mean(q["all_context"]["normalized_api_cost_usd"]))
        p_values[dataset] = primary["p_one_sided"]
        report["datasets"][dataset]["primary"] = primary
        # Orthogonal factorial contrasts. These are descriptive secondary endpoints.
        arrays = {name: np.asarray(values["f1"]) for name, values in q.items()}
        a, b = arrays["jev_0.2_original"], arrays["jev_0.2_rank"]
        c, d = arrays["jev_0.35_original"], arrays["jev_0.35_rank"]
        factorial = {
            "lower_cutoff_main_effect": compare((c + d) / 2, (a + b) / 2),
            "ranking_main_effect": compare((a + c) / 2, (b + d) / 2),
            "interaction": compare(d - c, b - a),
        }
        report["datasets"][dataset]["factorial_descriptive"] = factorial
        report["datasets"][dataset]["neural_descriptive"] = compare(
            q[neural]["f1"], q[winner]["f1"]
        )
        # Local compute price where neural generation + CPU compute matches the Jev API bill.
        s = report["datasets"][dataset]["summary"]
        cpu = s[neural]["local_cpu_seconds"]
        report["datasets"][dataset]["neural_cpu_break_even_usd_per_hour"] = (
            3600
            * (s[winner]["normalized_api_cost_usd"] - s[neural]["normalized_api_cost_usd"])
            / cpu
            if cpu
            else None
        )
        subgroups = {}
        for stratum in sorted({c["stratum"] for c in datasets[dataset]["evaluation"]}):
            ids = [
                i for i, c in enumerate(datasets[dataset]["evaluation"]) if c["stratum"] == stratum
            ]
            subgroups[stratum] = {
                "questions": len(ids),
                "f1": {arm: float(np.mean([q[arm]["f1"][i] for i in ids])) for arm in q},
            }
        report["datasets"][dataset]["subgroups_descriptive"] = subgroups
    adjusted = holm(p_values)
    for dataset, result in report["datasets"].items():
        p = result["primary"]
        p["holm_p"] = adjusted[dataset]
        clean = not any(
            result["summary"][name][k]
            for name in ["all_context", winner]
            for k in ["errors", "truncated", "unknown_usage"]
        )
        p["gate"] = (
            clean and p["ci"][0] > 0 and p["holm_p"] < 0.05 and p["normalized_cost"]["ci"][1] < 0
        )
    report["superiority_gate"] = all(d["primary"]["gate"] for d in report["datasets"].values())
    report["frozen"] = frozen
    write_json(ROOT / "evaluation-report.json", report)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["development", "evaluation"])
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    result = analyze(args.split, args.freeze)
    for dataset, values in result["datasets"].items():
        print(dataset, values)
