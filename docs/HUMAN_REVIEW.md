# Independent blinded review

The round-3 audit packet contains 40 deterministically sampled questions and two anonymous
answers per question. The human review is **pending** until independent reviewers provide
labels. The tool does not manufacture reviewer identity or grade answers automatically.

```sh
rag-jev serve --replay-only
```

Open `/review`. Load `artifacts/research-v3/blind-review.json` from the reproducibility
bundle. Give the reviewer the packet only; withhold `review-key.json`. The browser shows
the question, reference answers, full candidate evidence, and anonymous A/B answers.
The reviewer records semantic correctness, ambiguity and optional notes, then exports CSV.
File contents stay in the browser tab; labels are not sent to an API. Export before closing.

```sh
rag-jev review-results artifacts/research-v3/blind-review.json review-labels.csv \
  --output artifacts/human-review.json
# Only after review is finished, reveal the pipeline names:
rag-jev review-results artifacts/research-v3/blind-review.json review-labels.csv \
  --key artifacts/research-v3/review-key.json --output artifacts/unblinded-review.json
```

Blank and missing rows remain incomplete. Unknown/duplicate answers and partial labels
are rejected. “Unclear” is reported separately from correct/incorrect. The report records
self-reported reviewer IDs; it cannot verify independence or human identity. This packet
does not establish faithfulness to the selected context or citation correctness.
