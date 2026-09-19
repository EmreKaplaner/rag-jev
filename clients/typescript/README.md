# rag-jev TypeScript client

Node 20+ typed HTTP client. Types are generated from the Python service's OpenAPI
schema. Request deadlines, cancellation, response checks, and explicit HTTP errors
are included. There is no runtime dependency and no model key belongs in this client.

From the repository root:

```sh
npm --prefix clients/typescript ci
npm --prefix clients/typescript run build
uv run rag-jev serve
# In another terminal:
node examples/typescript-rag.mjs
```

```ts
import { ContextSelectorClient } from "rag-jev-client";

const client = new ContextSelectorClient({
  baseUrl: "http://127.0.0.1:8000",
  apiToken: process.env.RAG_JEV_API_TOKEN,
});
const result = await client.select({
  query: "What is the refund window?",
  documents: [{ id: "refunds", text: "Request a refund within 30 days." }],
  min_relevance: 0.2, // illustrative; validate on your queries
});
```

The package is not published. To install it in another project, build it and run
`npm pack` from this directory, then `npm install /path/to/rag-jev-client-0.1.0.tgz`.
The import above applies after installing that tarball.

Service `status` can be `applied`, `shadow`, or `bypassed`. A bypass is a successful
HTTP response containing unchanged input documents and a visible `error_code`.
HTTP failures throw `SelectionError` with a `status` field. This client does not
silently turn network/authentication errors into empty context.

The service currently has no browser CORS configuration; use this client server-side.
`apiToken` is the selection service token, never `TYPESAFE_API_KEY`.
