import type { components } from "./schema.js";

export type Document = components["schemas"]["Document"];
export type SelectionResult = components["schemas"]["SelectionResult"];
type WireRequest = components["schemas"]["SelectRequest"];
type BaseRequest = Omit<WireRequest, "mode" | "min_relevance">;
export type SelectRequest = BaseRequest & (
  | { mode?: "filter" | "filter_and_rerank"; min_relevance: number }
  | { mode: "rerank"; min_relevance?: never }
);

export class SelectionError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "SelectionError";
  }
}

export interface ClientOptions {
  baseUrl: string;
  /** Service bearer token, not the upstream TypeSafe API key. */
  apiToken?: string;
  /** Network deadline. Keep longer than the service's scoring deadline. */
  timeoutMs?: number;
  fetch?: typeof globalThis.fetch;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nonnegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (isRecord(value)) return `{${Object.keys(value).sort()
    .map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
  return JSON.stringify(value);
}

function validateResponse(value: unknown, request: SelectRequest): SelectionResult {
  if (!isRecord(value) || !["applied", "shadow", "bypassed"].includes(String(value.status))
      || !Array.isArray(value.documents) || !Array.isArray(value.decisions)
      || !Array.isArray(value.models) || !isRecord(value.usage)
      || typeof value.policy_version !== "string" || typeof value.prompt_version !== "string"
      || typeof value.elapsed_ms !== "number" || !Number.isFinite(value.elapsed_ms)
      || value.elapsed_ms < 0 || !value.models.every(x => typeof x === "string")
      || !nonnegativeInteger(value.input_characters) || !nonnegativeInteger(value.returned_characters)
      || (value.selected_characters !== null && !nonnegativeInteger(value.selected_characters))
      || !nonnegativeInteger(value.usage.input_tokens) || !nonnegativeInteger(value.usage.output_tokens)
      || !nonnegativeInteger(value.usage.completed_documents) || typeof value.usage.complete !== "boolean"
      || (value.error_code !== null && typeof value.error_code !== "string")) {
    throw new SelectionError("Invalid selection service response");
  }
  const known = new Map(request.documents.map(d => [d.id, d]));
  if ((value.selected_context_tokens !== undefined && value.selected_context_tokens !== null
        && !nonnegativeInteger(value.selected_context_tokens))
      || (value.context_token_budget_exceeded !== undefined && value.context_token_budget_exceeded !== null
        && typeof value.context_token_budget_exceeded !== "boolean")) {
    throw new SelectionError("Invalid context budget diagnostics");
  }
  const returned = new Set<string>();
  for (const doc of value.documents) {
    if (!isRecord(doc) || typeof doc.id !== "string" || returned.has(doc.id)
        || !known.has(doc.id) || doc.text !== known.get(doc.id)?.text || !isRecord(doc.metadata)
        || canonical(doc.metadata) !== canonical(known.get(doc.id)?.metadata ?? {})
        || (doc.group_id ?? null) !== (known.get(doc.id)?.group_id ?? null)
        || (doc.pinned ?? false) !== (known.get(doc.id)?.pinned ?? false)) {
      throw new SelectionError("Invalid document mapping in selection response");
    }
    returned.add(doc.id);
  }
  if (value.decisions.length !== request.documents.length) {
    throw new SelectionError("Incomplete selection decisions");
  }
  const bypassed = value.status === "bypassed";
  const selected = value.selected_ids;
  if (bypassed ? selected !== null : (!Array.isArray(selected)
      || !selected.every(id => typeof id === "string" && known.has(id))
      || new Set(selected).size !== selected.length)) {
    throw new SelectionError("Invalid selected document IDs");
  }
  const selectedSet = new Set(Array.isArray(selected) ? selected : []);
  const expectedReturned = value.status === "applied" ? selected : request.documents.map(d => d.id);
  if (canonical([...returned]) !== canonical(expectedReturned)) {
    throw new SelectionError("Selection status does not match returned documents");
  }
  value.decisions.forEach((decision, i) => {
    if (!isRecord(decision) || decision.id !== request.documents[i]?.id
        || decision.input_index !== i || typeof decision.returned !== "boolean"
        || decision.returned !== returned.has(String(decision.id))
        || decision.selected !== (bypassed ? null : selectedSet.has(decision.id))
        || (decision.model !== null && typeof decision.model !== "string")
        || (decision.input_tokens !== null && !nonnegativeInteger(decision.input_tokens))
        || (decision.output_tokens !== null && !nonnegativeInteger(decision.output_tokens))
        || (!bypassed && (decision.relevance === null || decision.model === null
          || (request.scoring_strategy !== "contextual"
            && (decision.input_tokens === null || decision.output_tokens === null))))
        || (bypassed && decision.reason !== "bypassed")
        || !["retained", "below_threshold", "beyond_top_n", "beyond_token_budget", "bypassed", "pinned", "group_retained"].includes(String(decision.reason))
        || (decision.relevance !== null && (typeof decision.relevance !== "number"
          || !Number.isFinite(decision.relevance) || decision.relevance < 0 || decision.relevance > 1))) {
      throw new SelectionError("Invalid selection decision");
    }
  });
  return value as unknown as SelectionResult;
}

export class ContextSelectorClient {
  private readonly options: ClientOptions;
  constructor(options: ClientOptions) {
    let url: URL;
    try { url = new URL(options.baseUrl); }
    catch { throw new SelectionError("baseUrl must be a valid HTTP(S) URL"); }
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password
        || url.search || url.hash) {
      throw new SelectionError("baseUrl must be an HTTP(S) URL without credentials/query/fragment");
    }
    if (!Number.isSafeInteger(options.timeoutMs ?? 10000) || (options.timeoutMs ?? 10000) <= 0) {
      throw new SelectionError("timeoutMs must be a positive integer");
    }
    this.options = { ...options, baseUrl: options.baseUrl.replace(/\/+$/, "") };
  }

  async select(request: SelectRequest, options: { signal?: AbortSignal } = {}): Promise<SelectionResult> {
    const controller = new AbortController();
    const abort = () => controller.abort(options.signal?.reason);
    if (options.signal?.aborted) abort();
    options.signal?.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(() => controller.abort(), this.options.timeoutMs ?? 10000);
    try {
      const response = await (this.options.fetch ?? globalThis.fetch)(`${this.options.baseUrl}/v1/select`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(this.options.apiToken ? { Authorization: `Bearer ${this.options.apiToken}` } : {}),
        },
        body: JSON.stringify(request),
        signal: controller.signal,
        redirect: "error",
      });
      if (!response.ok) {
        // Don't reflect upstream response bodies; they may contain private inputs.
        throw new SelectionError(`Selection service returned HTTP ${response.status}`, response.status);
      }
      return validateResponse(await response.json(), request);
    } catch (error) {
      if (error instanceof SelectionError) throw error;
      throw new SelectionError(controller.signal.aborted
        ? "Selection request aborted or timed out" : "Selection request failed");
    } finally {
      clearTimeout(timer);
      options.signal?.removeEventListener("abort", abort);
    }
  }
}
