"""Validate explicitly supplied reviewer labels; never substitute automated judgments."""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize_review(
    packet_path: str, labels_path: str, key_path: str | None = None
) -> dict[str, Any]:
    packet = json.loads(Path(packet_path).read_text())
    expected = {(c["case_id"], label) for c in packet["cases"] for label in c["answers"]}
    if not expected:
        raise ValueError("Review packet has no answers")
    key = json.loads(Path(key_path).read_text()) if key_path else {}
    seen: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    by_arm: dict[str, Counter[str]] = {}
    reviewers: set[str] = set()
    with Path(labels_path).open(newline="") as file:
        for row in csv.DictReader(file):
            pair = (row["case_id"], row["anonymous_answer"])
            if pair not in expected or pair in seen:
                raise ValueError("Unknown or duplicate review answer")
            seen.add(pair)
            label = row["semantically_correct"].strip().lower()
            reviewer = row["reviewer"].strip()
            ambiguity = row["ambiguous_question_or_reference"].strip().lower()
            if not label and not reviewer:
                continue
            if (
                label not in {"yes", "no", "unclear"}
                or not reviewer
                or ambiguity not in {"yes", "no", "unclear"}
            ):
                raise ValueError(
                    "Each labeled answer needs reviewer, correctness and ambiguity labels"
                )
            reviewers.add(reviewer)
            counts[label] += 1
            if key_path:
                arm = key[pair[0] + "/" + pair[1]]
                by_arm.setdefault(arm, Counter())[label] += 1
    completed = sum(counts.values())
    return {
        "status": "complete" if completed == len(expected) else "incomplete",
        "expected_answers": len(expected),
        "labeled_answers": completed,
        "unlabeled_answers": len(expected) - completed,
        "reviewers": sorted(reviewers),
        "correctness_counts": dict(counts),
        "by_arm": {k: dict(v) for k, v in by_arm.items()},
        "scope": "Supplied reviewer labels; reviewer identity and independence are self-reported. "
        "Unclear answers remain separate. This does not establish citation faithfulness.",
    }
