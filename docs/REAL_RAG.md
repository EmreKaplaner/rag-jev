# Try rag-jev in your own RAG pipeline

The service takes **already retrieved text**, selects context, and preserves the original
documents. Your current ingestion, embeddings, search, access checks, and answer model remain
in your application. Start with shadow mode and review a representative set of questions.

## 1. Start the playground

From the project root, with `TYPESAFE_API_KEY` in `.env`:

```sh
uv sync --locked
uv run rag-jev serve --port 8000
```

Open **http://127.0.0.1:8000**. The CLI loads `.env` without overwriting exported variables.

- Run an example with **Score with Jev** to make real scoring calls.
- **Explore hand-authored fixture** uses explicitly labeled synthetic scores.
- Paste your own query and a JSON array of `{id, text, metadata}` chunks.
- Optionally enter the IDs your current reranker would send to the answer model under
  **Evaluation & query context → Existing pipeline's selected IDs**. Order is preserved.
- Supply known relevant IDs to measure evidence retention. Leave blank for unlabeled data;
  `[]` means you have labeled the candidate set as containing no relevant evidence.
- Use **Shadow** to return every original candidate while inspecting the proposed selection.
- Adjust the cutoff, mode, or top-N target without making more Jev calls.
- Export a replay and import it later, or send the JSON file to a colleague using this app.
  Replays contain the documents and any generated answers. Imported provenance is supplied
  by the file author, not cryptographically authenticated.

Editing query/chunks clears the playground's old labels and baseline IDs. Add labels after
finishing edits. Changing text, query, or relevance guidance requires a new scoring run.

## 2. Retrieve from real files without an embedding API

This executable example uses **SQLite FTS5/BM25** to retrieve actual paragraphs from `.md`
and `.txt` files. It is a lexical retrieval example, not a vector-search benchmark.

```sh
uv run python examples/local_rag.py \
  --documents examples/knowledge \
  --query "How many days do I have to request a refund?" \
  --min-relevance 0.2 \
  --shadow \
  --base-url http://127.0.0.1:8000
```

Replace `examples/knowledge` with your document directory. Import the resulting
`artifacts/local-rag-run.json` in the playground. The example builds its search index in
memory, handles up to 256 retrieved candidates, and does not alter your files.

The `0.2` cutoff is an example setting. It dropped supporting evidence in our live smoke
test, so use it to explore the tradeoff rather than as a production recommendation.

## 3. Insert it into your existing retriever

For a Python application, call the library after retrieval. Reuse a selector within its
event loop so concurrent requests share the concurrency limit.

```python
from rag_jev import ContextSelector, Document, Jev

# Your application loads/exports TYPESAFE_API_KEY.
async with Jev() as provider:
    selector = ContextSelector(provider, on_error="passthrough")
    result = await selector.select(
        query=question,
        # retrieved_chunks is your current retriever's output, after authorization checks.
        documents=[
            Document(id=c.id, text=c.text, metadata=c.metadata)
            for c in retrieved_chunks
        ],
        min_relevance=calibrated_cutoff,
        shadow=True,
    )
    # In shadow mode result.documents still contains the original candidates.
    # result.selected_ids and result.decisions show the proposed selection.
    if not result.documents:
        # Retry retrieval, ask for clarification, or abstain according to your app policy.
        answer = "I need more evidence to answer this question."
    else:
        answer = await your_answer_model(question, result.documents)
```

Switch `shadow=False` only after reviewing evidence retention and answer quality. A
`bypassed` result means scoring failed and the original documents were returned; exclude
those runs from quality/savings measurements. `outcome="no_context_selected"` means no
passage passed the policy, not that the knowledge base has no answer.

### Score connected passages together

The optional `scoring_strategy="contextual"` sends the candidate set as one state, with
one relevance question per passage. Jev can then recognize intermediate facts that connect
passages. Choose **Passages together · connected evidence** in the playground, or pass
`scoring_strategy="contextual"` to the Python selector, HTTP request, or LangChain compressor.
The generated TypeScript client accepts the same field. Independent scoring remains the default.

This mode accepts at most 32 passages and enforces conservative local token estimates
(24k state tokens, 48k including questions); TypeSafe enforces its own tokenizer's limits.
Limit/provider errors follow the configured passthrough-or-raise policy and are never
silently retried as a different strategy. Titles and adjacent context should be included
in passage text. Arbitrary metadata and evaluation labels still remain local.

Contextual usage belongs to the whole request. Per-passage token counts are null; the
result's `usage` and replay's `scoring_usage` preserve the actual shared charge. Replaying
a cutoff preserves those costs and makes no scoring calls. Changing strategy requires
new scoring. Existing independent-scoring replays remain compatible.

Our [public-benchmark report](../benchmarks/public/RESULTS.md) evaluates a contextual 0.35
cutoff selected on development data. It is an experimental preset, not a universal cutoff.

The [second research round](../benchmarks/public/RESEARCH_RESULTS.md) tested a more
conservative evidence-selection profile on 400 new questions:

```python
result = await selector.select(
    query=question,
    documents=retrieved_chunks,
    scoring_strategy="contextual",
    mode="filter_and_rerank",
    min_relevance=0.20,
    # No top_n cap: it can discard necessary links in multi-hop evidence.
)
```

Use those same three fields in an HTTP request or the TypeScript client. For LangChain,
set `scoring_strategy="contextual"` and
`policy=Policy(mode="filter_and_rerank", min_relevance=0.20)` on the compressor.
The workbench's **Try research v2 settings** button sets them and clears the top-N cap;
it preserves your query, guidance, and shadow setting. Custom guidance and pinned/grouped
documents were not used in the benchmark. The benchmark pinned `Jev(model="jev-1.13.0")`.

This profile retained every supporting passage on 91% of MuSiQue questions versus 80%
for the old 0.35 filter, but increased API cost by 12.6% versus that filter. Its answer F1
was 1.68 points higher on MuSiQue and 0.70 lower on HotpotQA, with inconclusive confidence
intervals. Treat it as an option when missing evidence is costly, and test on your own
questions. The default remains unchanged.

Optional document controls:

```json
{
  "id": "refund-receipt",
  "text": "A purchase receipt is required.",
  "metadata": {"source": "refunds.md", "page": 3},
  "group_id": "refund-rule-and-conditions",
  "pinned": false
}
```

Documents sharing a `group_id` are kept together if any group member qualifies. `pinned`
forces retention of that document and its group. These controls can **exceed the top-N
target**, reported by `top_n_exceeded`. Filtering preserves original order; reranking sorts
groups by their best score and preserves order within each group. Grouping preserves
related evidence after independent scoring; it does not jointly reason across passages.
Use `retrieval_query` for a standalone rewrite of a conversational question. Include useful
headings in `text`; arbitrary metadata is preserved locally and is not sent to Jev.

### LangChain

Install the optional adapter in your application's environment:

```sh
uv pip install '/absolute/path/to/rag-jev[langchain]'
```

```python
from rag_jev.integrations.langchain import JevDocumentCompressor
from rag_jev.models import Policy

compressor = JevDocumentCompressor(
    base_url="http://127.0.0.1:8000",
    policy=Policy(min_relevance=calibrated_cutoff, shadow=True),
    # scoring_strategy="contextual", # optional; see limits above
    # api_token="your-service-token",  # if configured on the service
    # group_metadata_key="parent_id", # optional, explicit opt-in
    # pin_metadata_key="must_keep",
)
retrieved = await your_retriever.ainvoke(question)
selected = await compressor.acompress_documents(retrieved, question)
# Or compose compressor.as_runnable() with {"query": question, "documents": retrieved}.
answer = await your_answer_chain.ainvoke({"question": question, "context": selected})
```

Both synchronous `compress_documents` and async calls are supported. The adapter maps
back to original LangChain document objects, preserving duplicate source IDs and metadata
that cannot be JSON serialized. `last_result` exposes diagnostics for the last completed
call; don't use that property to coordinate concurrent requests.

### Dify

The native model-provider plugin is in `integrations/dify`. A locally built package is
available at `dist/rag-jev.difypkg`. To rebuild with the official Dify plugin CLI:

```sh
dify plugin package integrations/dify -o dist/rag-jev.difypkg
```

Install the package in a Dify installation that permits local plugins. Add the **rag-jev**
model provider, configure the service URL and service bearer token, and select its **Jev**
reranker in Knowledge Retrieval. Dify's score threshold and top-N map to the selector.
The plugin uses the model configured on the service, preserves original passage indices,
and reports provider errors instead of inventing rerank scores. Dify's reranker interface
passes text only, so pins, groups, and shadow analysis belong in the Python/HTTP integration.
The provider's optional **How to judge evidence** setting also exposes contextual scoring;
the same 32-passage/context limits apply. The default remains independent scoring.

For Dify in Docker on your Mac, use a reachable URL such as
`http://host.docker.internal:8000`. Bind the service to `0.0.0.0` and configure
`RAG_JEV_API_TOKEN` first; the CLI requires authentication for non-loopback binding.
On a remote Dify instance, use a service address reachable by its plugin worker.
Credential validation makes a small real scoring call. Full Dify application deployment
is separate from the SDK integration and package checks performed here.

## 4. Connect your answer model and evaluate

Configure the optional Chat Completions endpoint in `.env` and restart the service:

```dotenv
RAG_JEV_GENERATION_BASE_URL=https://your-provider.example/v1
RAG_JEV_GENERATION_MODEL=your-model-id
RAG_JEV_GENERATION_API_KEY=your-provider-key
RAG_JEV_GENERATION_TOKEN_LIMIT_FIELD=max_completion_tokens
RAG_JEV_GENERATION_MAX_OUTPUT_TOKENS=700
```

The service appends `/chat/completions`. Endpoints expecting `max_tokens` can set the
token-limit field accordingly. An endpoint that doesn't require authentication can leave
the key blank. Credentials stay on the server. Increase the output budget if your model
reports truncated or empty output.

For GPT-5.6 Luna, use `https://api.openai.com/v1` and model `gpt-5.6-luna`.
Set `RAG_JEV_GENERATION_REASONING_EFFORT=low` and, for the initial pilot,
`RAG_JEV_GENERATION_MAX_OUTPUT_TOKENS=2000`; this budget includes reasoning tokens.
The service accepts `OPENAI_API_KEY` as a fallback only for that exact OpenAI base URL;
an explicit `RAG_JEV_GENERATION_API_KEY` takes precedence. Other endpoints require their
own explicit credential. Replay answers record the requested budget and reasoning effort.
A quota error requires funding or quota on the API project; changing the model does not
resolve it. Empty responses retain reported token usage, so failed calls aren't counted
as free.

Use **Generate both answers**, or add `--compare` to the local-files command. Both branches
receive the same question and system prompt, with their respective context. Empty context
uses an explicit application fallback and makes no generation call. Citation IDs are
validated against each branch's sources; factual support still needs review.

The playground reports actual provider prompt/output usage when available and an explicitly
labeled `cl100k_base` estimate for passage tokens. Optional per-million USD rates in
`.env.example` enable cost estimates. Missing usage/rates remain unavailable. Costs exclude
retrieval and provider-specific discounts; selected-stage timing adds original scoring time
to newly measured generation time. Cached policy replay is not presented as live latency.

Try at least a representative set of answerable, ambiguous, multi-passage, false-premise,
and unanswerable questions. Compare against your existing reranker, inspect supporting
evidence that was lost, and review answer correctness/citations alongside total stage cost
and latency. Tune on one set and verify on a held-out set. The included toy examples and
successful HTTP calls do not establish a general quality improvement.
