# Third-party material

The MIT license covers original rag-jev code and documentation. It does not relicense
upstream dependencies, model weights, API services, or benchmark source material.

- The two recorded examples in `src/rag_jev/static/replays` contain MuSiQue benchmark
  questions and Wikipedia-derived passages, from the
  [MuSiQue author project](https://github.com/StonyBrookNLP/musique), pinned source commit
  `8b8d8ce102f97f968b259ea587a4482163eed2ff`. The upstream CC BY 4.0 license, pinned at license commit
  `922ac98f19a201998dbdae6d7f2887a5258dbdeb`, is included as
  `MUSIQUE_LICENSE`. Passage text begins with the original article title; article history
  and attribution can be found at the corresponding English Wikipedia article. The JSON
  structure, Jev scores and model answers were added by this project. These are recorded,
  selected illustrations from the first public study, not representative samples.
- Benchmark source URLs, hashes, authors' metric implementations, and data-license links
  are recorded in `benchmarks/public` and `benchmarks/research`. Full source datasets and
  model weights are downloaded separately and are not included in normal packages.
- FastAPI documentation excerpts in `benchmarks/pilot/corpus` retain the accompanying
  `benchmarks/pilot/FASTAPI_LICENSE` and source metadata.
- TypeSafe Jev and configured answer models are external services; this project does not
  distribute their weights or grant access to those services. Ettin's model-card and pinned
  revision are recorded in the research protocol; its weights are not shipped.
- Python and npm dependencies retain their respective licenses. The browser uses native
  fonts and makes no third-party font requests.
