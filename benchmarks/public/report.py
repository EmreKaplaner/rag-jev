"""Aggregate every frozen case, optionally freeze development's selected policy."""

import argparse
from pathlib import Path
from statistics import mean, median

from benchmarks.public.common import bootstrap_delta, digest, read_json, write_json
from benchmarks.public.run import ROOT, get_inputs, trace_path


def report(split, freeze=False):
    protocol, datasets = get_inputs()
    if split == "development":
        candidates = read_json(ROOT / "development-shortlist.json")["picked"]
    else:
        candidates = [read_json(ROOT / "frozen-policy.json")["policy"]]
    policies = [{"name": "all_context"}, {"name": "bm25_top5"}, *candidates]
    report_data = {"split": split, "protocol_hash": digest(protocol), "datasets": {}}
    for dataset, splits in datasets.items():
        rows = {}
        for policy in policies:
            rows[policy["name"]] = [
                read_json(trace_path(split, c, "answer_" + policy["name"])) for c in splits[split]
            ]
        summaries = {}
        for name, group in rows.items():
            assert len({r["case_id"] for r in group}) == len(splits[split])
            errors = sum(r["answer"]["status"] == "error" for r in group)
            unknown_costs = sum(r["cost_usd"] is None for r in group)
            summaries[name] = {
                "cases": len(group),
                "answer_em": mean(r["metrics"]["em"] for r in group),
                "answer_f1": mean(r["metrics"]["f1"] for r in group),
                "support_recall": mean(r["support_recall"] for r in group),
                "all_support_retained": mean(r["all_support_retained"] for r in group),
                "api_cost_usd": sum(r["cost_usd"] for r in group) if not unknown_costs else None,
                "errors": errors,
                "unknown_costs": unknown_costs,
                "unknown_cache_usage": sum(not r["cache_usage_known"] for r in group),
                "truncated_answers": sum(r["answer"]["finish_reason"] == "length" for r in group),
                "prompt_tokens": sum(r["answer"]["input_tokens"] or 0 for r in group),
                "cached_prompt_tokens": sum(r["answer"]["cached_input_tokens"] or 0 for r in group),
                "output_tokens": sum(r["answer"]["output_tokens"] or 0 for r in group),
                "median_stage_ms": median(r["stage_ms"] for r in group),
            }
        comparisons = {}
        for candidate in candidates:
            for baseline in ["all_context", "bm25_top5"]:
                name = candidate["name"]
                interval = bootstrap_delta(
                    [r["metrics"]["f1"] for r in rows[baseline]],
                    [r["metrics"]["f1"] for r in rows[name]],
                )
                costs = [summaries[x]["api_cost_usd"] for x in [baseline, name]]
                savings = 1 - costs[1] / costs[0] if all(x is not None for x in costs) else None
                interval["cost_savings_fraction"] = savings
                interval["success_gate"] = (
                    interval["ci95"][0] > 0
                    and savings is not None
                    and savings > 0
                    and not any(
                        summaries[x]["errors"]
                        or summaries[x]["unknown_costs"]
                        or summaries[x]["unknown_cache_usage"]
                        or summaries[x]["truncated_answers"]
                        for x in [baseline, name]
                    )
                )
                comparisons[name + "_vs_" + baseline] = interval
        report_data["datasets"][dataset] = {"summary": summaries, "comparisons": comparisons}
    write_json(ROOT / (split + "-report.json"), report_data)
    if freeze:
        if split != "development" or (ROOT / "frozen-policy.json").exists():
            raise ValueError("Policy can only be frozen once from development")
        summaries = [v["summary"] for v in report_data["datasets"].values()]
        if any(s["errors"] or s["unknown_costs"] for d in summaries for s in d.values()):
            raise ValueError("Resolve development failures before selecting policy")
        base_cost = sum(d["all_context"]["api_cost_usd"] for d in summaries)
        eligible = [
            p
            for p in candidates
            if sum(d[p["name"]]["api_cost_usd"] for d in summaries) < base_cost
        ]
        chosen = max(
            eligible or candidates,
            key=lambda p: (
                mean(d[p["name"]]["answer_f1"] for d in summaries),
                -sum(d[p["name"]]["api_cost_usd"] for d in summaries),
            ),
        )
        value = {
            "policy": chosen,
            "protocol_hash": digest(protocol),
            "development_report_hash": digest(report_data),
            "development_report": report_data,
            "reason": (
                "Highest development F1 among lower-cost candidates; fallback to best F1 if none"
            ),
        }
        write_json(ROOT / "frozen-policy.json", value)
        write_json(Path("benchmarks/public/frozen-policy.json"), value)
        print("FROZEN", chosen)
    for name, data in report_data["datasets"].items():
        print(name)
        for policy, summary in data["summary"].items():
            print(policy, summary)
        print(data["comparisons"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["development", "evaluation"])
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    report(args.split, args.freeze)
