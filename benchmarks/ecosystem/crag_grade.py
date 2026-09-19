"""CRAG-style semantic diagnosis using pinned upstream prompts/parser and a Luna judge.

Not an official leaderboard result: judge model differs and the upstream evaluation
loop has defects. Preserve its prompt/parser/75-token trim, explicitly document the
corrected loop and our abstention mapping, and retain every failed grade.
"""

import ast
import hashlib
import json
import re
from pathlib import Path

from benchmarks.ecosystem.common import ARMS, ROOT, protocol
from benchmarks.ecosystem.judge import AuditedJudge
from benchmarks.public.common import bootstrap_delta, digest, read_json, write_json


def upstream_helpers():
    from transformers import LlamaTokenizerFast

    upstream = ROOT / "upstream/crag"
    namespace = {
        "re": re,
        "tokenizer": LlamaTokenizerFast.from_pretrained(str(upstream / "tokenizer")),
    }
    for node in ast.parse((upstream / "prompts/templates.py").read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "INSTRUCTIONS",
                    "IN_CONTEXT_EXAMPLES",
                }:
                    namespace[target.id] = ast.literal_eval(node.value)
    names = {"get_system_message", "parse_response", "trim_predictions_to_max_token_length"}
    functions = [
        node
        for node in ast.parse((upstream / "local_evaluation.py").read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(functions) == len(names)
    # Execute only these inspected pure upstream helpers, avoiding CUDA-only imports
    # and the buggy evaluation loop. Their source hashes are pinned in the protocol.
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "crag-pinned-helpers", "exec"),
        namespace,
    )
    return namespace


def main():
    p = protocol()
    cases = [
        c
        for c in read_json(ROOT / "retrieval/crag-bm25.json")["cases"]
        if c["split"] == "evaluation"
    ]
    spec = {
        "base_protocol_hash": digest(p),
        "case_ids": [c["id"] for c in cases],
        "arms": ARMS,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_sha": p["upstream"]["crag"],
        "judge": p["generator"],
        "scope": (
            "Exploratory CRAG-style semantic accuracy, hallucination, missing and "
            "utility. Not official leaderboard scores; same-model judge bias. Upstream "
            "prompts, response parser and 75-token Llama trimming retained. Corrected "
            "upstream undefined prediction_lowercase and string/list ground-truth loop. "
            "Map exact 'unanswerable' to missing, like 'I don't know'. Alternative gold "
            "answers accepted. Arm names never enter judge prompts. Identical prompts "
            "reuse their paid judge result."
        ),
    }
    target = ROOT / "crag-grade-protocol.json"
    if target.exists():
        assert read_json(target) == spec
    else:
        write_json(target, spec)
    helpers, judge = upstream_helpers(), AuditedJudge(p)
    rows = []
    for case in sorted(cases, key=lambda c: digest(["grade", c["id"]])):
        for arm in sorted(ARMS, key=lambda a: digest(["grade", case["id"], a])):
            answer = read_json(ROOT / "answers" / f"{case['id']}-{arm}.json")
            prediction = helpers["trim_predictions_to_max_token_length"](
                answer["prediction"]
            ).strip()
            low = prediction.lower()
            grade, explanations = "incorrect", []
            if answer["answer"]["status"] == "error":
                grade = "failed"
            elif "i don't know" in low or low == "unanswerable":
                grade = "missing"
            else:
                for reference in case["references"]:
                    reference = reference.strip()
                    if low == reference.lower() or (
                        "invalid" in low and "invalid" in reference.lower()
                    ):
                        grade = "correct"
                        break
                    if ("invalid" in low) != ("invalid" in reference.lower()):
                        continue
                    messages = [
                        {"role": "system", "content": helpers["get_system_message"]()},
                        {
                            "role": "user",
                            "content": (
                                f"Question: {case['question']}\n Ground truth: {reference}\n "
                                f"Prediction: {prediction}\n"
                            ),
                        },
                    ]
                    response = judge([messages])[0]
                    explanation, score = helpers["parse_response"](response)
                    explanations.append(explanation)
                    if score == 1:
                        grade = "correct"
                        break
                    if score == -1:
                        grade = "failed"
                        break
            rows.append(
                {
                    "id": case["id"],
                    "arm": arm,
                    "grade": grade,
                    "prediction_after_trim": prediction,
                    "explanations": explanations,
                }
            )
            write_json(ROOT / "crag-grades.json", rows)
    summary = {}
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        counts = {
            g: sum(r["grade"] == g for r in group)
            for g in ["correct", "incorrect", "missing", "failed"]
        }
        n = len(group)
        summary[arm] = {
            **counts,
            "n": n,
            "accuracy": counts["correct"] / n,
            "hallucination": counts["incorrect"] / n,
            "missing_rate": counts["missing"] / n,
            "utility_failures_minus_one": (2 * counts["correct"] + counts["missing"]) / n - 1,
        }
    by_key = {(r["id"], r["arm"]): float(r["grade"] == "correct") for r in rows}
    intervals = {
        arm: bootstrap_delta(
            [by_key[c["id"], "baseline"] for c in cases],
            [by_key[c["id"], arm] for c in cases],
            samples=10000,
        )
        for arm in ARMS
        if arm != "baseline"
    }
    result = {
        "protocol": spec,
        "summary": summary,
        "accuracy_delta_vs_baseline_descriptive": intervals,
    }
    write_json(ROOT / "crag-semantic-report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
