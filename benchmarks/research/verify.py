"""Offline integrity audit: sources, split isolation, real requests, metrics and costs."""

import hashlib
import json
import math
from pathlib import Path

from benchmarks.public.common import digest, normalize, read_json, write_json
from benchmarks.public.research_verify import author_metrics
from benchmarks.research.design import (
    ROOT,
    answer_path,
    choose,
    inputs,
    isolation_keys,
    policies,
    raw_rows,
    score_path,
)
from benchmarks.research.runner import evidence_metrics, fingerprint, generation_cost
from rag_jev.generation import Answer
from rag_jev.models import Judgment, SelectRequest, Usage
from rag_jev.selector import apply_policy


def close(a, b):
    if abs(a - b) > 1e-10:
        raise ValueError("Numeric integrity mismatch")


def main():
    protocol, datasets = inputs()
    manifest_path = ROOT / "artifact-manifest.json"
    if manifest_path.exists():
        for name, checksum in read_json(manifest_path).items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == checksum
    source_rows = raw_rows()
    previous = read_json("benchmarks/research/prior-exposure.json")
    used_keys, used_questions = set(), set()
    for dataset, rows in source_rows.items():
        for row in rows:
            if row["id"] in previous[dataset]:
                used_keys.update(isolation_keys(dataset, row))
                used_questions.add(normalize(row["question"]))
    isolation = read_json(ROOT / "isolation.json")
    assert digest(isolation) == protocol["isolation_hash"]
    for dataset, splits in datasets.items():
        by_id = {r["id"]: r for r in source_rows[dataset]}
        for case in splits["evaluation"]:
            keys = isolation_keys(dataset, by_id[case["id"]])
            assert not keys & used_keys
            assert normalize(case["query"]) not in used_questions
            assert sorted(keys) == isolation[dataset + "/" + case["id"]]
            used_keys.update(keys)
            used_questions.add(normalize(case["query"]))
    for source in protocol["sources"]:
        assert hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() == source["sha256"]
    for source in read_json("benchmarks/research/metric-sources.json").values():
        assert (
            hashlib.sha256(Path(source["local_path"]).read_bytes()).hexdigest() == source["sha256"]
        )
    functions, _ = author_metrics()
    frozen = read_json(ROOT / "frozen.json")
    assert digest(read_json(ROOT / "development-report.json")) == frozen["development_report_hash"]
    rates = protocol["pricing_usd_per_million"]
    counters = {
        "answer_records": 0,
        "captured_requests": 0,
        "empty_branches": 0,
        "production_policy_parity": 0,
        "neural_truncated_pairs": 0,
    }
    actual_models = set()
    artifact_hashes = {}
    for split in ["development", "evaluation"]:
        arms = policies(split, protocol)
        for dataset, splits in datasets.items():
            author = functions[dataset]
            for case in splits[split]:
                traces = {
                    kind: read_json(score_path(split, case, kind)) for kind in ["jev", "ettin"]
                }
                jev, neural = traces["jev"], traces["ettin"]
                assert jev["fingerprint"] == digest(
                    {
                        "query": case["query"],
                        "documents": case["documents"],
                        "scoring": protocol["scoring"],
                    }
                )
                assert neural["fingerprint"] == digest(
                    {
                        "pairs": [(case["query"], d["text"]) for d in case["documents"]],
                        "config": protocol["neural"],
                    }
                )
                assert neural["model"] == protocol["neural"]["model"]
                assert neural["revision"] == protocol["neural"]["revision"]
                assert len(neural["pair_token_lengths"]) == len(case["documents"])
                assert max(neural["pair_token_lengths"]) <= protocol["neural"]["max_length"]
                for score_trace in traces.values():
                    assert len(score_trace["scores"]) == len(case["documents"])
                    assert all(math.isfinite(s) for s in score_trace["scores"])
                assert all(0 <= s <= 1 for s in jev["scores"])
                close(
                    jev["api_cost_usd"],
                    (
                        jev["input_tokens"] * rates["jev_input"]
                        + jev["output_tokens"] * rates["jev_output"]
                    )
                    / 1e6,
                )
                counters["neural_truncated_pairs"] += traces["ettin"]["truncated_pairs"]
                assert traces["jev"]["models"] == [protocol["scoring"]["model"]]
                for policy in arms:
                    trace = traces.get(policy["kind"])
                    selected = (
                        [
                            i
                            for i, d in enumerate(case["documents"])
                            if d["id"] in case["relevant_ids"]
                        ]
                        if policy["kind"] == "oracle"
                        else choose(case["documents"], trace["scores"] if trace else [], policy)
                    )
                    docs = [case["documents"][i] for i in selected]
                    ids = [d["id"] for d in docs]
                    if policy["kind"] == "jev" and policy["order"] != "bookend":
                        req = SelectRequest(
                            query=case["query"],
                            documents=case["documents"],
                            scoring_strategy="contextual",
                            min_relevance=policy["threshold"],
                            mode="filter" if policy["order"] == "original" else "filter_and_rerank",
                        )
                        result = apply_policy(
                            req,
                            [
                                Judgment(relevance=s, model=trace["models"][0])
                                for s in trace["scores"]
                            ],
                            usage=Usage(
                                input_tokens=trace["input_tokens"],
                                output_tokens=trace["output_tokens"],
                                completed_documents=len(req.documents),
                            ),
                        )
                        assert result.selected_ids == ids
                        counters["production_policy_parity"] += 1
                    for rep in range(protocol["repetitions"][split]):
                        file = answer_path(split, case, policy, rep)
                        record = read_json(file)
                        assert record["selected_ids"] == ids
                        assert record["fingerprint"] == fingerprint(
                            case, policy, rep, protocol, ids
                        )
                        answer = Answer.model_validate(record["answer"])
                        prediction = answer.text if answer.status == "generated" else ""
                        if dataset == "hotpotqa":
                            em = max(
                                author["exact_match_score"](prediction, a) for a in case["answers"]
                            )
                            f1 = max(author["f1_score"](prediction, a)[0] for a in case["answers"])
                        else:
                            em = max(
                                author["compute_exact"](a, prediction) for a in case["answers"]
                            )
                            f1 = max(author["compute_f1"](a, prediction) for a in case["answers"])
                        close(record["f1"], f1)
                        close(record["em"], em)
                        for key, value in evidence_metrics(case, ids).items():
                            close(record[key], value)
                        actual, normalized, unknown = generation_cost(answer, rates)
                        scoring = trace["api_cost_usd"] if policy["kind"] == "jev" else 0
                        close(record["api_cost_usd"], actual + scoring)
                        close(record["normalized_api_cost_usd"], normalized + scoring)
                        assert unknown == record["unknown_usage"]
                        if answer.status == "no_context":
                            assert not docs and not file.with_suffix(".request.json").exists()
                            counters["empty_branches"] += 1
                        else:
                            request = read_json(file.with_suffix(".request.json"))
                            assert request["model"] == protocol["generation"]["model"]
                            assert (
                                request["max_completion_tokens"]
                                == protocol["generation"]["max_completion_tokens"]
                            )
                            assert (
                                request["reasoning_effort"]
                                == protocol["generation"]["reasoning_effort"]
                            )
                            assert request["messages"][0] == {
                                "role": "system",
                                "content": protocol["generation"]["system_prompt"],
                            }
                            evidence = [
                                {"label": f"S{i + 1}", "text": d["text"]}
                                for i, d in enumerate(docs)
                            ]
                            assert json.loads(request["messages"][1]["content"]) == {
                                "question": case["query"],
                                "evidence": evidence,
                            }
                            assert set(request) == {
                                "model",
                                "messages",
                                "max_completion_tokens",
                                "reasoning_effort",
                            }
                            counters["captured_requests"] += 1
                            response_file = file.with_suffix(".response.json")
                            if answer.status == "generated":
                                response = read_json(response_file)
                                assert response["model"] == answer.model
                                assert response["choices"][0]["message"]["content"] == answer.text
                                close(response["usage"]["prompt_tokens"], answer.input_tokens)
                                close(response["usage"]["completion_tokens"], answer.output_tokens)
                                close(
                                    response["usage"]["prompt_tokens_details"]["cached_tokens"],
                                    answer.cached_input_tokens,
                                )
                                actual_models.add(answer.model)
                        counters["answer_records"] += 1
    for folder in [ROOT / "development", ROOT / "evaluation"]:
        for file in sorted(folder.glob("*/*/*.json")):
            artifact_hashes[str(file.relative_to(ROOT))] = hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
    write_json(ROOT / "artifact-manifest.json", artifact_hashes)
    evidence = {
        **counters,
        "status": "passed",
        "protocol_hash": digest(protocol),
        "artifact_manifest_hash": digest(artifact_hashes),
        "evaluation_report_hash": digest(read_json(ROOT / "evaluation-report.json")),
        "actual_generation_models": sorted(actual_models),
        "evaluation_questions": 400,
        "shared_evaluation_isolation_keys": 0,
    }
    write_json(ROOT / "verification.json", evidence)
    print(evidence)


if __name__ == "__main__":
    main()
