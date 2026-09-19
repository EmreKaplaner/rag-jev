# Calibrate on your own retrieved candidates

Use separate development and held-out JSON arrays. Each item contains `id`, `query`,
`documents` (`id`, `text`, optional metadata/group/pin), and `relevant_ids`. An empty
relevance list means no relevant evidence exists in those candidates, not “unlabeled”.
Optional `baseline_ids` records your current pipeline's chosen passages in their actual
order. Without it, the baseline is all candidates. Optional `retrieval_query` supplies
your application's standalone conversational rewrite. No rewriting is performed here.

Use `family_id` for related questions that must stay in one split. Duplicate normalized
questions and IDs are rejected across and within splits; family overlap across splits is
rejected. Authors must still prevent semantic/data-source leakage that IDs cannot detect.

```sh
rag-jev calibrate development.json held-out.json \
  --scoring-strategy contextual --min-recall 0.98 \
  --cutoffs 0.2 0.35 0.5 --top-ns 3 5 8 \
  --output artifacts/my-calibration.json
```

Contextual scoring accepts at most 32 passages and the documented token limits per
question. With a configured key, this command scores each case once, preserving shared
usage. It evaluates filtering, reranking, and their combination without rescoring each
policy. It chooses the lowest mean passage-token count meeting the requested observed
development evidence recall; ties favor higher NDCG, then the stable policy ID.

The choice is written to `my-calibration.frozen.json` **before** held-out scoring and
assessment. The held-out table also includes the other fixed candidates for descriptive
inspection; those results never change the selected policy. If no development candidate
qualifies, no reusable policy file is produced. Output files are never overwritten.

Artifacts:

- `my-calibration.json`: baseline, fixed candidate policies, selected-policy results,
  usage, per-question metrics and descriptive context-token difference interval.
  Supplied family IDs are resampled together; fewer than two families suppress the interval.
- `my-calibration.policy.json`: reusable selection arguments with `shadow: true`.
- `my-calibration.frozen.json`: selection, configuration and both input hashes.
- `my-calibration.traces.json`: live scores, queries hashed for matching, model IDs and
  usage; emitted after each successful case. Replay inputs themselves contain source text.

```python
import json
config = json.load(open("artifacts/my-calibration.policy.json"))
result = await selector.select(query=question, documents=retrieved_chunks, **config)
```

Start in shadow mode. Review lost evidence, then compare actual answers before enabling
selection. The recall target is an observed development statistic, **not a guarantee**.
This calibration measures candidate evidence, not semantic answer correctness or total
API cost. It cannot recover passages absent from retrieval. Use the workbench's answer
comparison or `evaluate(..., answer_evaluator=your_callback)` to evaluate application answers.

Replay a complete trace set with `--replay my-calibration.traces.json` and a **new** output
path; no scoring requests are made. Interrupted runs preserve successful traces, but do
not silently resume an unresolved paid call. Complete or reconcile traces before replay.

For the existing evaluation command, contextual scoring is also supported:

```sh
rag-jev eval cases.json --scoring-strategy contextual --min-relevance 0.2 \
  --save-traces artifacts/contextual-traces.json
rag-jev eval cases.json --scoring-strategy contextual --min-relevance 0.35 \
  --replay artifacts/contextual-traces.json
```

Use representative answerable, missing-evidence, conversational and multi-passage cases.
Keep labels, private inputs and trace files out of public issues and source control.
