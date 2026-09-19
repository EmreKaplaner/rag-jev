"""Explicit post-run reporting amendment for numeric CRAG alternative references."""

import math

from benchmarks.ecosystem import report
from benchmarks.ecosystem.common import ROOT, fingerprint_files
from benchmarks.ecosystem.retrieve import all_cases
from benchmarks.public.common import answer_metrics, write_json


def scalar_reference_metrics(prediction, references, dataset):
    normalized = []
    for reference in references:
        if isinstance(reference, str):
            normalized.append(reference)
        elif dataset == "crag" and type(reference) in (int, float) and math.isfinite(reference):
            normalized.append(str(reference))
        else:
            raise TypeError("Expected a string or finite CRAG numeric reference")
    return answer_metrics(prediction, normalized, dataset)


def main():
    from pathlib import Path

    affected = [
        {"id": case["id"], "split": case["split"], "references": case["references"]}
        for case in all_cases()
        if case["dataset"] == "crag" and any(not isinstance(r, str) for r in case["references"])
    ]
    write_json(
        ROOT / "report-reference-amendment.json",
        {
            "reason": "The frozen string normalizer rejects numeric CRAG alternative answers.",
            "action": (
                "Convert finite numeric CRAG references to their literal string only for "
                "diagnostic EM/F1. Preserve raw inputs, predictions, every arm and all cases. "
                "No semantic-judge, retrieval, cost or selection changes; no new API calls."
            ),
            "affected_cases": affected,
            "analysis_code": fingerprint_files([Path(__file__)]),
        },
    )
    original = report.answer_metrics
    try:
        report.answer_metrics = scalar_reference_metrics
        report.main()
    finally:
        report.answer_metrics = original


if __name__ == "__main__":
    main()
