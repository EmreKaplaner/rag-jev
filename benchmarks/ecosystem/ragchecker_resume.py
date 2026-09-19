"""Preserve completed empty claim extractions using RefChecker's native zero-claim semantics.

The original generic judge bridge rejects empty text. Claim extraction can validly
return no triples for an abstention. This adapter reuses that exact paid response;
it never resubmits it, edits a trace, supplies invented claims, or accepts an empty
entailment judgment. Empty extraction coverage remains explicit in the report.
"""

import hashlib
from pathlib import Path

from benchmarks.ecosystem import judge
from benchmarks.ecosystem.common import ROOT, protocol
from benchmarks.public.common import digest, read_json, write_json

OriginalJudge = judge.AuditedJudge


def completed_empty_extraction(prompt, record, extraction_prefix):
    return (
        isinstance(prompt, str)
        and prompt.startswith(extraction_prefix)
        and record.get("http_status") == 200
        and record.get("cost_usd") is not None
        and record.get("choice", {}).get("finish_reason") == "stop"
        and record.get("choice", {}).get("message", {}).get("content") == ""
    )


class ClaimAwareJudge(OriginalJudge):
    def __call__(self, prompts):
        from refchecker.extractor.extractor_prompts import LLM_TRIPLET_EXTRACTION_PROMPT_Q

        prefix = LLM_TRIPLET_EXTRACTION_PROMPT_Q.split("{q}")[0]
        output = []
        for prompt in prompts:
            try:
                output.extend(super().__call__([prompt]))
            except RuntimeError:
                messages = (
                    [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
                )
                body = {
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": 2000,
                    "reasoning_effort": "low",
                }
                fp = digest({"body": body, "protocol": digest(self.p)})
                path = ROOT / "judge-requests" / f"{fp}.json"
                if not path.exists():
                    raise
                record = read_json(path)
                if not completed_empty_extraction(prompt, record, prefix):
                    raise
                output.append("")
                write_json(
                    ROOT / "empty-extraction-interpretations" / f"{fp}.json",
                    {
                        "request_fingerprint": fp,
                        "raw_trace_unchanged": True,
                        "interpretation": (
                            "HTTP200, finish=stop, empty extraction -> zero claims using native "
                            "RefChecker parser. Retain as empty extraction coverage; may be "
                            "abstention or missed extraction. No retry."
                        ),
                    },
                )
        return output


def main():
    from refchecker.extractor import LLMExtractor

    assert LLMExtractor().parse_claims("") == []
    amendment = {
        "base_protocol_hash": digest(protocol()),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trigger": (
            "First extraction of an 'unanswerable' response completed with "
            "HTTP200/stop and empty content. Native RefChecker parses this as zero "
            "triples, while the generic audited bridge rejected it."
        ),
        "action": (
            "Keep all raw traces and costs. Feed completed empty extraction responses "
            "into the native parser. No retry, answer regeneration, sampling change, "
            "or invented claims. Empty entailment/checker results still fail."
        ),
    }
    path = ROOT / "ragchecker-empty-extraction-amendment.json"
    if path.exists():
        assert read_json(path) == amendment
    else:
        write_json(path, amendment)
    judge.AuditedJudge = ClaimAwareJudge
    judge.main()


if __name__ == "__main__":
    main()
