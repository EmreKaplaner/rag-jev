"""Fixed examples with hand-authored scores. This is NOT a Jev benchmark or model."""

from rag_jev.evaluation import EvalCase, Trace, fingerprint
from rag_jev.models import Document, Judgment
from rag_jev.provider import ProviderError

CASES = [
    EvalCase(
        id="refund",
        query="How many days do I have to request a refund?",
        documents=[
            Document(
                id="shipping",
                text="Orders ship within 5 business days.",
                metadata={"source": "shipping.md"},
            ),
            Document(
                id="refund",
                text="You may request a refund within 30 days of purchase.",
                metadata={"source": "refunds.md", "page": 2},
            ),
            Document(
                id="receipt",
                text="Keep your receipt; it is required for refund requests.",
                metadata={"source": "refunds.md", "page": 3},
            ),
        ],
        relevant_ids=["refund", "receipt"],
        reference_answer="30 days",
    ),
    EvalCase(
        id="correction",
        query="How do I renew my subscription every 30 days?",
        documents=[
            Document(id="annual", text="Subscriptions renew annually, not every 30 days."),
            Document(id="renew", text="Annual subscriptions renew automatically unless canceled."),
            Document(id="returns", text="Return physical products within 30 days."),
        ],
        relevant_ids=["annual", "renew"],
        reference_answer="annually",
    ),
    EvalCase(
        id="no-answer",
        query="What is the office Wi-Fi password?",
        documents=[
            Document(id="hours", text="The office opens at 9 AM."),
            Document(id="wifi", text="Wi-Fi is available in the office lobby."),
        ],
        relevant_ids=[],
        reference_answer="insufficient evidence",
    ),
]
SCORES = [[0.03, 0.98, 0.60], [0.97, 0.85, 0.02], [0.01, 0.08]]


def fixture_traces() -> list[Trace]:
    return [
        Trace(
            case_id=case.id,
            input_hash=fingerprint(case),
            elapsed_ms=0,
            source="hand_authored_fixture_NOT_a_model_benchmark",
            judgments=[Judgment(relevance=p, model="fixture-not-jev") for p in scores],
        )
        for case, scores in zip(CASES, SCORES, strict=True)
    ]


class FixtureScorer:
    async def score(self, query: str, document: Document, guidance: str | None) -> Judgment:
        for case, trace in zip(CASES, fixture_traces(), strict=True):
            if query == case.query:
                for candidate, judgment in zip(case.documents, trace.judgments, strict=True):
                    if candidate.id == document.id and candidate.text == document.text:
                        return judgment
        raise ProviderError("unknown_demo_fixture")
