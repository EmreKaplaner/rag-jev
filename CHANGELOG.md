# Changelog

## Unreleased

- Original squirrel identity, repository banner, SVG/ICO favicon and shared app branding.
- Visual research notebook linking all seven study tracks with methods and limitations.
- Stateless remote-deployment guide, non-root Docker image and Compose configuration.
- Platform-provided `PORT` and `RAG_JEV_HOST` configuration; explicit CLI flags take priority.
- CI coverage for service-to-service container requests, auth, shadow mode and TLS forwarding.
- Completed ecosystem pilot: full-corpus BEIR subsets, native framework integration checks,
  CRAG answer comparison, diagnostic judges, and reconciled API-spend report.

## 0.2.0 — 2026-09-19

- MIT license, contribution/security guidance and third-party notices.
- Key-free browser replay mode with recorded improvement and regression examples.
- Contextual evaluation and development/held-out policy calibration with frozen choices,
  shared usage, caller baselines and reusable shadow-mode configuration.
- Passage-token budgets that preserve pins/groups and explicitly report overruns.
- Browser-based blinded human review and label completeness validation.
- Synthetic long-passage, duplicate, missing-evidence, conversational and embedded-
  instruction diagnostics, with live scoring latency at three concurrency levels.
- Reproducible research reports and immutable historical evidence bundle retained.

### Compatibility

New policy fields are optional. Policies without a token budget keep their original IDs,
so existing recorded comparisons import unchanged. The token budget estimates passage
text only with cl100k_base; it is not a provider context-window guarantee. Fail-open and
shadow responses can return more text than the proposed selection budget.

Historical research fingerprints intentionally require their original source snapshot.
Reproduce those studies from the release's frozen research archive, not updated core code.
