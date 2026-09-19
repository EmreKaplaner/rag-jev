# MVP acceptance scope

- Python selector and HTTP API: filter, rerank, combined mode, explicit threshold, top-N
  target, shadow, concurrency bound, total scoring deadline, and declared error behavior.
- Evidence controls: preserved source metadata, optional standalone retrieval query,
  pinned documents, atomic groups, explicit empty-selection outcome.
- Browser workbench: custom query/chunks, caller-supplied baseline IDs, optional relevance
  labels, live scoring, clearly labeled fixtures, score replay, retained/dropped evidence,
  mobile layout, and JSON replay export/import. No server-side document persistence.
- Answer comparison: configurable server-side Chat Completions endpoint, same prompt/model,
  citation/source mapping, empty-context fallback, provider usage, measured stage latency,
  configured-rate cost estimates, and preserved errors for each branch.
- LangChain: native compressor plus LCEL runnable, sync/async, original-object mapping.
- Dify: native model-provider plugin, SDK discovery/invocation, index preservation, package.
- Runnable local-file retrieval example and real-pipeline adoption guide.
- Optional contextual scoring across Python/HTTP, TypeScript, LangChain, and Dify; shared
  usage accounting, bounded context, and strategy-aware replay/input validation.
- Reproducible HotpotQA/MuSiQue subset evaluation, fixed development/evaluation separation,
  official answer metrics, cost accounting, uncertainty, dashboard, and inspectable replays.

The configured GPT-5.6 Luna endpoint and Jev are tested live. Public-benchmark results are
scoped to the recorded subsets; they do not establish general factual-accuracy improvement.
The generation transport and browser comparison also have controlled HTTP tests. Dify SDK invocation
and packaging are tested locally; no remote Dify installation is provisioned by this project.

Not part of this MVP: a public hosted SaaS, public replay URLs, account/billing systems,
automated factual grading, a universal threshold, guaranteed quality gains, or demo-video
production. Exported replay files are the sharing mechanism.
