import { readFileSync } from "node:fs";
import { ContextSelectorClient } from "../clients/typescript/dist/index.js";

const client = new ContextSelectorClient({
  baseUrl: process.env.RAG_JEV_URL ?? "http://127.0.0.1:8000",
  apiToken: process.env.RAG_JEV_API_TOKEN,
});
const input = JSON.parse(readFileSync(new URL("./request.json", import.meta.url)));
const result = await client.select(input);
console.log(JSON.stringify(result, null, 2));
// Feed result.documents to your existing answering model.
