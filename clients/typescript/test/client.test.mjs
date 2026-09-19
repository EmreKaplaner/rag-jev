import { test } from "node:test";
import assert from "node:assert/strict";
import { ContextSelectorClient, SelectionError } from "../dist/index.js";

const request = { query: "q", documents: [{ id: "a", text: "evidence", metadata: {} }], min_relevance: 0.2 };
const valid = {
  documents: request.documents, selected_ids: ["a"],
  decisions: [{ id: "a", input_index: 0, relevance: 0.9, model: "test", input_tokens: 1,
    output_tokens: 1, selected: true, returned: true, reason: "retained" }],
  status: "applied", error_code: null, models: ["test"], prompt_version: "test", policy_version: "test",
  usage: { input_tokens: 1, output_tokens: 1, completed_documents: 1, complete: true },
  elapsed_ms: 10, input_characters: 8, selected_characters: 8, returned_characters: 8,
};

test("request mapping and service token", async () => {
  const client = new ContextSelectorClient({ baseUrl: "http://localhost:8000/", apiToken: "service",
    fetch: async (url, init) => {
      assert.equal(url, "http://localhost:8000/v1/select");
      assert.equal(init.headers.Authorization, "Bearer service");
      assert.deepEqual(JSON.parse(init.body), request);
      return Response.json(valid);
    },
  });
  assert.deepEqual(await client.select(request), valid);
});

test("contextual scoring accepts shared usage without inventing per-passage costs", async () => {
  const shared = { ...valid, prompt_version: "contextual-evidence-v1",
    decisions: valid.decisions.map(d => ({ ...d, input_tokens: null, output_tokens: null })),
    usage: { ...valid.usage, input_tokens: 999 } };
  const client = new ContextSelectorClient({ baseUrl: "http://localhost",
    fetch: async () => Response.json(shared) });
  const result = await client.select({ ...request, scoring_strategy: "contextual" });
  assert.equal(result.usage.input_tokens, 999);
  assert.equal(result.decisions[0].input_tokens, null);
  await assert.rejects(client.select(request), SelectionError);
});

test("invalid protocol and altered document mapping are rejected", async () => {
  for (const body of [{}, { ...valid, documents: [{ id: "wrong", text: "injected", metadata: {} }] },
    { ...valid, decisions: [] }, { ...valid, decisions: [{ ...valid.decisions[0], relevance: 9 }] },
    { ...valid, selected_ids: ["missing"] }, { ...valid, documents: [{ id: "a", text: "evidence", metadata: { source: "changed" } }] },
    { ...valid, usage: {} }, { ...valid, documents: [{ id: "missing", metadata: {} }] }]) {
    const client = new ContextSelectorClient({ baseUrl: "http://localhost", fetch: async () => Response.json(body) });
    await assert.rejects(client.select(request), SelectionError);
  }
});

test("HTTP errors do not reflect private response bodies", async () => {
  const client = new ContextSelectorClient({ baseUrl: "http://localhost",
    fetch: async () => new Response("SECRET upstream body", { status: 502 }) });
  await assert.rejects(client.select(request), e => e.status === 502 && !e.message.includes("SECRET"));
});

test("timeouts and caller abort cancel the HTTP request", async () => {
  const fetch = async (url, { signal }) => new Promise((resolve, reject) => {
    if (signal.aborted) return reject(new Error("aborted"));
    signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
  });
  const client = new ContextSelectorClient({ baseUrl: "http://localhost", fetch, timeoutMs: 10 });
  await assert.rejects(client.select(request), /aborted or timed out/);
  await assert.rejects(client.select(request, { signal: AbortSignal.abort() }), /aborted or timed out/);
});

test("invalid client configuration is rejected", () => {
  for (const baseUrl of ["file:///tmp/file", "http://user:pass@localhost", "http://localhost?key=x"]) {
    assert.throws(() => new ContextSelectorClient({ baseUrl }), SelectionError);
  }
  assert.throws(() => new ContextSelectorClient({ baseUrl: "http://localhost", timeoutMs: 0 }), SelectionError);
});
