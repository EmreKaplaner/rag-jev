import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ContextSelectorClient, SelectionError } from "../dist/index.js";

const request = JSON.parse(readFileSync(new URL("../../../examples/request.json", import.meta.url)));
request.documents = request.documents.map(d => ({ group_id: null, pinned: false, ...d }));
const baseUrl = process.env.RAG_JEV_TEST_URL;

test("TypeScript -> HTTP -> Python -> official SDK -> controlled upstream", async () => {
  assert.ok(baseUrl, "RAG_JEV_TEST_URL must point at the test service");
  const client = new ContextSelectorClient({ baseUrl, apiToken: "integration-test-token" });
  const filtered = await client.select(request);
  assert.deepEqual(filtered.selected_ids, ["refund", "receipt"]);
  assert.equal(filtered.status, "applied");
  assert.equal(filtered.documents[0].metadata.source, "refunds.md");
  assert.equal(filtered.usage.input_tokens, 75);

  const reranked = await client.select({
    query: request.query, documents: request.documents, mode: "rerank", top_n: 1,
  });
  assert.deepEqual(reranked.selected_ids, ["refund"]);
  const shadow = await client.select({ ...request, shadow: true });
  assert.deepEqual(shadow.documents, request.documents);
  assert.equal(shadow.status, "shadow");
  const empty = await client.select({ ...request, min_relevance: 1 });
  assert.deepEqual(empty.documents, []);
  const bypass = await client.select({ ...request, query: "__fail__", top_n: 1 });
  assert.deepEqual(bypass.documents, request.documents);
  assert.equal(bypass.status, "bypassed");
  const timeout = await client.select({ ...request, query: "__timeout__" });
  assert.equal(timeout.error_code, "deadline_exceeded");

  await assert.rejects(new ContextSelectorClient({ baseUrl }).select(request),
    error => error instanceof SelectionError && error.status === 401);
  await assert.rejects(client.select({ ...request, min_relevance: 2 }),
    error => error instanceof SelectionError && error.status === 422);
});
