"""Verify split isolation, author-metric parity and production-policy parity, offline.

Uses the author metric source snapshots downloaded during round 1 under
artifacts/benchmark-research. Only their standalone metric functions are executed.
"""

import ast
import collections
import hashlib
import re
import string
from pathlib import Path

from benchmarks.public.common import digest, normalize, read_json, write_json
from benchmarks.public.research import BASELINES, ROOT, inputs, path
from rag_jev.models import Judgment, SelectRequest, Usage
from rag_jev.selector import apply_policy


def author_metrics():
    functions, sources = {}, {}
    for dataset in ["hotpotqa", "musique"]:
        source = Path("artifacts/benchmark-research") / (dataset + "-official-metrics.py")
        module = ast.parse(source.read_text())
        names = (
            {"normalize_answer", "f1_score", "exact_match_score"}
            if dataset == "hotpotqa"
            else {"normalize_answer", "get_tokens", "compute_exact", "compute_f1"}
        )
        body = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in body} != names:
            raise ValueError("Unexpected author metric source")
        namespace = {
            "re": re,
            "string": string,
            "Counter": collections.Counter,
            "collections": collections,
        }
        exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
        functions[dataset] = namespace
        sources[dataset] = {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    return functions, sources


def main():
    protocol, datasets = inputs()
    functions, sources = author_metrics()
    development = {normalize(c["query"]) for d in datasets.values() for c in d["development"]}
    evaluation = [normalize(c["query"]) for d in datasets.values() for c in d["evaluation"]]
    assert not development.intersection(evaluation)
    assert len(evaluation) == len(set(evaluation)) == 400
    frozen = read_json(ROOT / "frozen-policy.json")
    assert frozen["protocol_hash"] == digest(protocol)
    assert frozen["development_report_hash"] == digest(read_json(ROOT / "development-report.json"))
    metrics_checked, selections_checked = 0, 0
    for dataset, splits in datasets.items():
        author = functions[dataset]
        for split, cases in splits.items():
            policies = BASELINES + (
                read_json(ROOT / "shortlist.json")["picked"]
                if split == "development"
                else [frozen["policy"]]
            )
            for case in cases:
                for policy in policies:
                    result = read_json(path(split, case, "answer_" + policy["name"]))
                    prediction = (
                        result["answer"]["text"]
                        if result["answer"]["status"] == "generated"
                        else ""
                    )
                    if dataset == "hotpotqa":
                        em = max(
                            author["exact_match_score"](prediction, a) for a in case["answers"]
                        )
                        f1 = max(author["f1_score"](prediction, a)[0] for a in case["answers"])
                    else:
                        em = max(author["compute_exact"](a, prediction) for a in case["answers"])
                        f1 = max(author["compute_f1"](a, prediction) for a in case["answers"])
                    assert result["metrics"]["em"] == em
                    assert abs(result["metrics"]["f1"] - f1) < 1e-12
                    metrics_checked += 1
                    if policy.get("scorer") != "contextual":
                        continue
                    trace = read_json(path(split, case, "contextual"))
                    request = SelectRequest(
                        query=case["query"],
                        documents=case["documents"],
                        scoring_strategy="contextual",
                        mode="filter_and_rerank"
                        if policy.get("order") == "relevance"
                        else "filter",
                        min_relevance=policy["threshold"],
                    )
                    usage = Usage(
                        input_tokens=trace["input_tokens"],
                        output_tokens=trace["output_tokens"],
                        completed_documents=len(request.documents),
                    )
                    selected = apply_policy(
                        request,
                        [Judgment(relevance=s, model=trace["models"][0]) for s in trace["scores"]],
                        usage=usage,
                    )
                    assert selected.selected_ids == result["selected_ids"]
                    assert selected.usage == usage
                    selections_checked += 1
    evidence = {
        "metrics_checked": metrics_checked,
        "production_selections_checked": selections_checked,
        "disjoint_evaluation_questions": len(evaluation),
        "author_sources": sources,
        "protocol_hash": digest(protocol),
        "status": "passed",
    }
    write_json(ROOT / "verification.json", evidence)
    print(evidence)


if __name__ == "__main__":
    main()
