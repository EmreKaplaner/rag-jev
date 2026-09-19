"""Independent artifact audit, including production-policy parity and spend reconciliation."""

import argparse
import math

from benchmarks.ecosystem.answer import generation_cost, selected
from benchmarks.ecosystem.common import ARMS, ROOT, Ledger, protocol
from benchmarks.ecosystem.retrieve import all_cases
from benchmarks.ecosystem.score import load_score, path
from benchmarks.public.common import digest, read_json, write_json
from rag_jev.generation import Answer
from rag_jev.models import Document, Judgment, SelectRequest, Usage
from rag_jev.selector import apply_policy


def main(partial=False):
    p = protocol()
    cases = all_cases()
    missing = []
    checks, costs, failed = 0, 0.0, []
    for case in cases:
        traces = {}
        for stage in ["bge", "jev", "jev_post_bge"]:
            if not path(case, stage).exists():
                missing.append(str(path(case, stage)))
                continue
            row = load_score(case, stage, p)
            if row["status"] != "ok":
                failed.append({"case": case["id"], "retriever": case["retriever"], "stage": stage})
                continue
            assert all(math.isfinite(v) for v in row["scores"])
            expected_count = (
                min(10, len(case["documents"]))
                if stage == "jev_post_bge"
                else len(case["documents"])
            )
            assert len(row["scores"]) == expected_count
            if stage != "bge":
                actual_cost = (
                    row["usage"]["input_tokens"] * p["pricing_usd_per_million"]["jev_input"] / 1e6
                )
                assert math.isclose(actual_cost, row["api_cost_usd"], abs_tol=1e-12)
                assert all(0 <= v <= 1 for v in row["scores"])
                costs += actual_cost
            traces[stage] = row
            checks += 1
        if len(traces) != 3:
            continue
        for arm in ["jev_rerank", "jev_filter", "bge_jev_filter"]:
            docs = selected(case, "bge", p) if arm == "bge_jev_filter" else case["documents"]
            trace = traces["jev_post_bge" if arm == "bge_jev_filter" else "jev"]
            assert trace["document_ids"] == [d["id"] for d in docs]
            mode = {
                "jev_rerank": "rerank",
                "jev_filter": "filter_and_rerank",
                "bge_jev_filter": "filter",
            }[arm]
            request = SelectRequest(
                query=case["query"],
                documents=[Document(id=d["id"], text=d["text"]) for d in docs],
                mode=mode,
                min_relevance=None if mode == "rerank" else p["threshold"],
                top_n=10,
                scoring_strategy="contextual",
            )
            result = apply_policy(
                request,
                [Judgment(relevance=v, model=p["jev"]) for v in trace["scores"]],
                usage=Usage.model_validate(trace["usage"]),
            )
            assert result.selected_ids == [d["id"] for d in selected(case, arm, p)]
            checks += 1
        if case["dataset"] == "crag":
            for arm in ARMS:
                target = ROOT / "answers" / f"{case['id']}-{arm}.json"
                if not target.exists():
                    missing.append(str(target))
                    continue
                row = read_json(target)
                answer = Answer.model_validate(row["answer"])
                assert row["prompt"]["query"] == case["query"]
                assert row["prompt"]["documents"] == [
                    {"id": d["id"], "text": d["text"]} for d in selected(case, arm, p)
                ]
                for field, normalized in [
                    ("api_cost_usd", False),
                    ("normalized_api_cost_usd", True),
                ]:
                    actual = generation_cost(answer, p, normalized)
                    assert row[field] == actual
                if row["api_cost_usd"] is not None:
                    costs += row["api_cost_usd"]
                if answer.status == "error":
                    failed.append({"case": case["id"], "stage": "generation", "arm": arm})
                checks += 1
    for target in (ROOT / "judge-requests").glob("*.json"):
        row = read_json(target)
        if row["cost_usd"] is not None:
            usage = row["usage"]
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            cost = (
                (usage["prompt_tokens"] - cached) * 0.2
                + cached * 0.02
                + usage["completion_tokens"] * 1.2
            ) / 1e6
            assert math.isclose(row["cost_usd"], cost, abs_tol=1e-12)
            costs += cost
        checks += 1
    integration_path = ROOT / "live-flashrag-integration.json"
    if integration_path.exists():
        integration = read_json(integration_path)
        assert integration["complete"]
        for row in integration["rows"]:
            expected = row["selector"]["usage"]["input_tokens"] * 0.042 / 1e6
            expected += generation_cost(Answer.model_validate(row["answer"]), p)
            assert math.isclose(row["api_cost_usd"], expected, abs_tol=1e-12)
            costs += expected
            checks += 1
    budget = Ledger().summary()
    assert math.isclose(costs, budget["known_usage_cost_usd"], abs_tol=1e-9)
    assert budget["remaining_after_reservations_usd"] >= 0
    out = {
        "protocol_hash": digest(p),
        "checks": checks,
        "missing_artifacts": missing,
        "failed_stages": failed,
        "known_cost_reconciles": True,
        "budget": budget,
        "complete": not missing and not failed and budget["unreconciled_requests"] == 0,
    }
    write_json(ROOT / "verification.json", out)
    print({**out, "missing_artifacts": len(missing)})
    if not partial and not out["complete"]:
        raise SystemExit("Incomplete audit: no clean-completion claim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    main(parser.parse_args().partial)
