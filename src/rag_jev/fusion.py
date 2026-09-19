"""Equal-weight reciprocal rank fusion over the same ordered candidate list.

Input order is the retriever ranking. Tied Jev scores retain that order;
ties in the fused sum also retain input order. Scores are not probabilities.
"""

from collections.abc import Sequence
from math import isfinite

RRF_K = 60


def ranking_signals(scores: Sequence[float]) -> tuple[list[int], list[float]]:
    """Return one-based Jev ranks and RRF scores aligned with input positions."""
    if any(not isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("expected finite Jev scores in [0, 1]")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for rank, i in enumerate(order, 1):
        ranks[i] = rank
    return ranks, [1 / (RRF_K + i + 1) + 1 / (RRF_K + r) for i, r in enumerate(ranks)]
