const $ = (id) => document.getElementById(id);
const state = {
  view: null,
  comparison: null,
  examples: [],
  generation: false,
  dirty: false,
  busy: false,
  imported: false,
  replayOnly: false,
  recordings: [],
  rankingReport: null,
};
function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function status(text, error = false) {
  $("status").textContent = text;
  $("status").className = error ? "error" : "";
}
function updateActions() {
  $("score").disabled = state.busy || state.replayOnly;
  $("research-profile").disabled = state.busy || state.replayOnly;
  $("compare").disabled =
    state.busy || !state.view || state.dirty || !state.generation;
  $("export").disabled = state.busy || !state.view || state.dirty;
  $("threshold").disabled = state.busy || ["rerank", "fusion"].includes($("mode").value);
  $("rankings").disabled = state.busy || !state.view || state.dirty;
  $("compare-rankings").disabled = state.busy || !state.view || state.dirty || !state.generation;
  $("export-rankings").disabled = state.busy || !state.rankingReport || state.dirty;
}
function busy(value) {
  state.busy = value;
  document
    .querySelectorAll("[data-lock]")
    .forEach((el) => (el.disabled = value));
  $("import").disabled = value;
  updateActions();
}
async function api(path, body) {
  const token = $("token").value.trim();
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const data = await response.json();
  if (!response.ok) {
    const issue = data.issues?.[0];
    throw new Error(
      issue
        ? `${issue.loc.join(".")}: ${issue.message}`
        : data.detail || data.error || `HTTP ${response.status}`,
    );
  }
  return data;
}
async function task(fn) {
  if (state.busy) return;
  busy(true);
  try {
    await fn();
  } catch (e) {
    status(e.message, true);
  } finally {
    busy(false);
  }
}
function optionalIDs(id) {
  const text = $(id).value.trim();
  if (!text) return null;
  const value = JSON.parse(text);
  if (!Array.isArray(value) || value.some((x) => typeof x !== "string"))
    throw new Error(`${id} must be a JSON array of IDs.`);
  return value;
}
function policy() {
  return {
    mode: $("mode").value,
    min_relevance:
      ["rerank", "fusion"].includes($("mode").value) ? null : Number($("threshold").value),
    top_n: $("top-n").value ? Number($("top-n").value) : null,
    max_context_tokens: $("token-budget").value ? Number($("token-budget").value) : null,
    shadow: $("shadow").checked,
    relevance_guidance: $("guidance").value.trim() || null,
  };
}
function study() {
  return {
    request: {
      query: $("query").value,
      scoring_strategy: $("scoring-strategy").value,
      documents: JSON.parse($("documents").value),
      retrieval_query: $("retrieval-query").value.trim() || null,
      ...policy(),
    },
    relevant_ids: optionalIDs("relevant-ids"),
    baseline_ids: optionalIDs("baseline-ids"),
  };
}
function replayBody() {
  return { record: state.view.record, policy: policy() };
}
function clearAnswers() {
  state.rankingReport = null;
  $("ranking-results").replaceChildren();
  state.comparison = null;
  $("answers").replaceChildren(
    node(
      "p",
      "Generate answers for this selection. Changing the policy clears the previous comparison.",
      "hint",
    ),
  );
}
function loadInputs(record) {
  const r = record.request;
  $("query").value = r.query;
  $("scoring-strategy").value = r.scoring_strategy || "independent";
  $("documents").value = JSON.stringify(r.documents, null, 2);
  $("retrieval-query").value = r.retrieval_query || "";
  $("guidance").value = r.relevance_guidance || "";
  $("relevant-ids").value =
    record.relevant_ids === null ? "" : JSON.stringify(record.relevant_ids);
  $("baseline-ids").value =
    record.baseline_ids == null ? "" : JSON.stringify(record.baseline_ids);
  $("input-count").textContent = `/ ${r.documents.length}`;
}
function loadPolicy(p) {
  $("mode").value = p.mode;
  $("threshold").value = p.min_relevance ?? 0.2;
  $("threshold-value").textContent = Number($("threshold").value).toFixed(2);
  $("top-n").value = p.top_n ?? "";
  $("token-budget").value = p.max_context_tokens ?? "";
  $("shadow").checked = p.shadow;
}
function render(view) {
  state.view = view;
  state.dirty = false;
  const r = view.record,
    s = view.selection;
  const ids = new Set(s.selected_ids);
  const gold = r.relevant_ids === null ? null : new Set(r.relevant_ids);
  $("provenance").textContent =
    `${state.imported ? "IMPORTED · " : ""}${r.source === "fixture" ? "HAND-AUTHORED FIXTURE" : "RECORDED LIVE RUN"} · ${r.request.scoring_strategy || "independent"} · ${s.models.join(", ")}`;
  const metrics = $("metrics");
  metrics.hidden = false;
  metrics.replaceChildren();
  for (const [label, value, detail] of [
    [
      "Context passages",
      `${view.baseline_ids.length} → ${ids.size}`,
      `${r.request.documents.length} candidates scored`,
    ],
    [
      "Context tokens ≈",
      `${view.baseline_context_tokens} → ${view.selected_context_tokens}`,
      "cl100k_base estimate",
    ],
    [
      "Evidence recall",
      view.evidence_recall === null
        ? "Unmeasured"
        : `${Math.round(view.evidence_recall * 100)}%`,
      gold?.size
        ? "Of labeled candidate evidence"
        : "Requires nonempty relevance labels",
    ],
    [
      "Original scoring",
      r.source === "fixture"
        ? "Fixture"
        : `${Math.round(r.scoring_elapsed_ms)} ms`,
      "Policy replay makes no scoring calls",
    ],
  ]) {
    const m = node("div", undefined, "metric");
    m.append(node("p", label), node("strong", value), node("small", detail));
    metrics.append(m);
  }
  const notes = [];
  if (s.status === "shadow")
    notes.push(
      "Shadow returns every original candidate. Below is the proposed selection.",
    );
  if (s.outcome === "no_context_selected")
    notes.push(
      "No context selected. This does not establish that the knowledge base has no answer.",
    );
  if (s.top_n_exceeded)
    notes.push("Top N exceeded to preserve pinned passages or whole groups.");
  if (s.context_token_budget_exceeded)
    notes.push("Token budget exceeded to preserve pinned evidence and its whole groups.");
  if (view.relevant_dropped?.length)
    notes.push(
      `Labeled evidence dropped: ${view.relevant_dropped.join(", ")}.`,
    );
  if (gold?.size === 0)
    notes.push("This example is labeled unanswerable from its candidate set.");
  $("policy-note").textContent =
    notes.join(" ") ||
    "Every decision is inspectable. Scores are judgments, not guarantees.";
  const evidence = $("evidence");
  evidence.replaceChildren();
  if (!r.request.documents.length)
    evidence.append(
      node(
        "p",
        "No candidates supplied. Retrieval must find evidence before selection can help.",
        "hint",
      ),
    );
  r.request.documents.forEach((d, i) => {
    const decision = s.decisions[i],
      kept = ids.has(d.id);
    const row = node(
      "article",
      undefined,
      `passage ${kept ? "kept" : "dropped"}`,
    );
    row.id = `passage-${i}`;
    const top = node("div", undefined, "passage-top");
    top.append(node("span", d.id, "passage-id"));
    if (gold?.has(d.id)) top.append(node("span", "Labeled relevant", "gold"));
    const meter = node("meter");
    meter.min = 0;
    meter.max = 1;
    meter.value = decision.relevance;
    meter.setAttribute("aria-label", `${d.id} relevance ${decision.relevance}`);
    top.append(
      meter,
      node("span", decision.relevance.toFixed(2)),
      node(
        "span",
        `${kept ? "Keep" : "Drop"} · ${decision.reason.replaceAll("_", " ")}`,
        "verdict",
      ),
    );
    row.append(top, node("p", d.text));
    if (decision.fusion_score != null) {
      row.append(node("p",
        `Original #${decision.original_rank} · Jev #${decision.jev_rank} · RRF ${decision.fusion_score.toFixed(6)} · Selected #${kept ? s.selected_ids.indexOf(d.id) + 1 : "—"}`,
        "hint"));
    }
    const source = [
      d.metadata?.source,
      d.group_id ? `group: ${d.group_id}` : "",
      d.pinned ? "pinned" : "",
    ]
      .filter(Boolean)
      .join(" · ");
    if (source) row.append(node("div", source, "source"));
    evidence.append(row);
  });
  updateActions();
}
function renderAnswers(c, recorded = false) {
  state.comparison = c;
  const area = $("answers");
  area.replaceChildren();
  for (const [title, a, total, cost] of [
    ["Existing pipeline", c.baseline, c.baseline_stage_ms, c.baseline_cost_usd],
    ["Jev selection", c.selected, c.selected_stage_ms, c.selected_cost_usd],
  ]) {
    const panel = node("section", undefined, "answer");
    panel.append(
      node("h3", title),
      node(
        "small",
        `${recorded ? "Imported recorded answer · " : ""}${a.model || "Application fallback"} · ${a.status}`,
      ),
      node("p", a.text),
    );
    const links = node("div", undefined, "citations");
    for (const [label, id] of Object.entries(a.source_map || {})) {
      if (!a.citations.includes(id)) continue;
      const i = state.view.record.request.documents.findIndex(
        (d) => d.id === id,
      );
      const link = node("a", `[${label}] ${id}`);
      link.href = `#passage-${i}`;
      links.append(link);
    }
    panel.append(
      links,
      node(
        "p",
        `Prompt tokens: ${a.input_tokens ?? "unavailable"} · Output: ${a.output_tokens ?? "unavailable"}\nStage: ${Math.round(total)} ms · Estimated cost: ${cost === null ? "rates/usage unavailable" : `$${cost.toFixed(6)}`}`,
      ),
    );
    if (a.unknown_citations?.length)
      panel.append(
        node("p", `Unrecognized citations: ${a.unknown_citations.join(", ")}`),
      );
    if (a.error_code) panel.append(node("small", a.error_code));
    if (a.finish_reason === "length")
      panel.append(
        node(
          "small",
          "Output reached its token limit; the answer may be incomplete.",
        ),
      );
    area.append(panel);
  }
  area.append(
    node(
      "div",
      `${c.timing_basis} ${c.cost_basis} ${c.citation_check}.`,
      "comparison-notes",
    ),
  );
}
function invalidate() {
  state.dirty = true;
  clearAnswers();
  updateActions();
  status("Inputs changed. Score again to compare or export these inputs.");
}
async function initialize() {
  const [config, examples, recordings] = await Promise.all([
    api("/v1/config"),
    api("/v1/examples"),
    api("/v1/recordings"),
  ]);
  state.replayOnly = config.replay_only;
  state.recordings = recordings;
  $("recording").replaceChildren(node("option", "Choose an improvement or regression"));
  $("recording").firstChild.value = "";
  recordings.forEach((r) => {
    const option = node("option", r.label); option.value = r.id; $("recording").append(option);
  });
  $("replay-notice").hidden = !state.replayOnly;
  $("replay-notice").textContent = "REPLAY MODE · No keys needed. Live scoring and generation are disabled. Recorded answers are illustrative examples, not new measurements.";
  state.examples = examples;
  state.generation = config.generation_configured;
  $("generation-status").textContent = state.generation
    ? `Model: ${config.generation_model}. Citations can be inspected below.`
    : state.replayOnly ? "Recorded answers only. Restart without --replay-only to enable configured providers."
    : "Set RAG_JEV_GENERATION_BASE_URL and RAG_JEV_GENERATION_MODEL on the server, plus a key if required, then restart.";
  if (!state.view) {
    const c = examples[0];
    loadInputs({
      request: c,
      relevant_ids: c.relevant_ids,
      baseline_ids: null,
    });
  }
  updateActions();
  if (state.replayOnly && !state.view && recordings.length) await loadRecording(recordings[0].id);
}
async function loadRecording(id) {
  const entry = state.recordings.find((r) => r.id === id);
  if (!entry) return;
  const bundle = await api(entry.path);
  const result = await api("/v1/import", bundle);
  state.imported = true;
  loadInputs(result.view.record); loadPolicy(result.view.policy); clearAnswers(); render(result.view);
  if (result.comparison) renderAnswers(result.comparison, true);
  $("recording").value = id;
  status(`${entry.label}. ${entry.note}`);
}
$("recording").onchange = () => task(() => loadRecording($("recording").value));
$("connect").onclick = () => task(initialize);
$("research-profile").onclick = () => {
  if (state.busy) return;
  $("scoring-strategy").value = "contextual";
  loadPolicy({
    mode: "filter_and_rerank",
    min_relevance: 0.2,
    top_n: null,
    shadow: $("shadow").checked,
  });
  invalidate();
  status(
    "Research v2 applied: passages together, cutoff 0.20, filter & rerank, no top-N cap. Score again to use these settings. Your query and guidance are preserved.",
  );
};
$("example").onchange = () => {
  const c = state.examples.find((c) => c.id === $("example").value);
  if (!c) return;
  loadInputs({ request: c, relevant_ids: c.relevant_ids, baseline_ids: null });
  invalidate();
};
for (const id of [
  "query",
  "scoring-strategy",
  "documents",
  "retrieval-query",
  "guidance",
  "baseline-ids",
  "relevant-ids",
])
  $(id).addEventListener("input", () => {
    if (id === "query" || id === "documents") {
      $("relevant-ids").value = "";
      $("baseline-ids").value = "";
    }
    invalidate();
  });
$("score").onclick = () =>
  task(async () => {
    const body = study();
    status("Scoring candidates with Jev…");
    const view = await api("/v1/runs", body);
    state.imported = false;
    clearAnswers();
    render(view);
    status(
      "Live scoring complete. Tune the policy below without additional Jev calls.",
    );
  });
$("fixture").onclick = () =>
  task(async () => {
    const view = await api(`/v1/fixtures/${$("example").value}`);
    state.imported = false;
    loadInputs(view.record);
    loadPolicy(view.policy);
    clearAnswers();
    render(view);
    status("Hand-authored fixture loaded. These are not Jev predictions.");
  });
for (const id of ["threshold", "mode", "top-n", "token-budget", "shadow"]) {
  $(id).addEventListener("input", () => {
    $("threshold-value").textContent = Number($("threshold").value).toFixed(2);
    clearAnswers();
    updateActions();
  });
  $(id).addEventListener("change", () => {
    // A native blur can repeat an already-applied change. Avoid disabling the
    // button the user is clicking when the policy is already current.
    const next = policy();
    if (state.view && !state.dirty && Object.entries(next).every(([key, value]) => state.view.policy[key] === value)) return;
    return task(async () => {
      if (!state.view || state.dirty) return;
      const view = await api("/v1/replay", replayBody());
      render(view);
      status("Policy applied to saved judgments. No Jev call made.");
    });
  });
}
$("compare").onclick = () =>
  task(async () => {
    status("Generating answers with the configured model…");
    const c = await api("/v1/compare", replayBody());
    renderAnswers(c);
    status(
      c.baseline.status === "error" || c.selected.status === "error"
        ? "Comparison finished with a generation error; inspect each answer."
        : "Comparison complete. Review correctness and supporting citations.",
      c.baseline.status === "error" || c.selected.status === "error",
    );
  });
$("export").onclick = () => {
  const blob = new Blob(
    [
      JSON.stringify(
        { ...replayBody(), comparison: state.comparison },
        null,
        2,
      ),
    ],
    { type: "application/json" },
  );
  const a = node("a");
  a.href = URL.createObjectURL(blob);
  a.download = "rag-jev-run.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  status(
    "Replay exported with documents and available answers. Review its contents before sharing.",
  );
};
$("import").onchange = () =>
  task(async () => {
    const file = $("import").files[0];
    if (!file) return;
    if (file.size > 2_000_000)
      throw new Error("Replay exceeds the 2 MB limit.");
    const result = await api("/v1/import", JSON.parse(await file.text()));
    state.imported = true;
    loadInputs(result.view.record);
    loadPolicy(result.view.policy);
    clearAnswers();
    render(result.view);
    if (result.comparison) renderAnswers(result.comparison, true);
    status(
      "Imported replay. Integrity checked against its inputs; provenance and recorded answers are supplied by the file author.",
    );
    $("import").value = "";
  });
task(initialize);

function rankingBody() {
  return {record: state.view.record, top_n: Number($("top-n").value || 10),
    max_context_tokens: $("token-budget").value ? Number($("token-budget").value) : null};
}
function renderRankings(report) {
  state.rankingReport = report;
  const area = $("ranking-results");
  area.replaceChildren();
  area.append(node("p", report.source === "fixture"
    ? "HAND-AUTHORED FIXTURE · These rankings are not live Jev predictions."
    : "RECORDED SCORES · No new Jev calls. Answer quality requires review.", "hint"));
  const grid = node("div", undefined, "ranking-grid");
  area.append(grid);
  const titles = {original: "Original retrieval", jev: "Jev reranking", fusion: "Fusion · original + Jev"};
  const money = (v) => v == null ? "unavailable" : `$${v.toFixed(6)}`;
  for (const arm of report.arms) {
    const panel = node("section", undefined, "answer");
    panel.append(node("h3", titles[arm.name]));
    panel.append(node("p", `Order: ${arm.selection.selected_ids.join(" → ") || "Empty"}`));
    panel.append(node("p", `Evidence recall: ${arm.evidence_recall == null ? "unlabeled" : (arm.evidence_recall * 100).toFixed(1) + "%"} · nDCG@${report.top_n}: ${arm.ndcg == null ? "unlabeled" : arm.ndcg.toFixed(3)} · Context tokens ≈ ${arm.selection.selected_context_tokens}`));
    panel.append(node("p", `Estimated API cost: ${money(arm.total_cost_usd)} · Stage: ${Math.round(arm.stage_elapsed_ms)} ms${arm.answer ? " · With generation" : " · Selection only"}`));
    if (arm.answer) {
      panel.append(node("p", arm.answer.text), node("small", `${arm.answer.model || "Application fallback"} · ${arm.answer.status}${arm.answer_reused_from ? " · Answer reused from " + titles[arm.answer_reused_from] : ""}`));
      if (arm.answer.unknown_citations.length) panel.append(node("p", "Unrecognized citations: " + arm.answer.unknown_citations.join(", ")));
    }
    grid.append(panel);
  }
  const details = node("details");
  details.append(node("summary", "Measurement details"));
  for (const note of [report.selection_basis, report.quality_basis, report.cost_basis, report.timing_basis]) details.append(node("p", note, "hint"));
  area.append(details);
  updateActions();
}
for (const [id, endpoint] of [["rankings", "/v1/rankings"], ["compare-rankings", "/v1/compare-rankings"]]) {
  $(id).onclick = () => task(async () => {
    status(id === "rankings" ? "Comparing saved rankings…" : "Generating the three-way comparison…");
    const report = await api(endpoint, rankingBody());
    renderRankings(report);
    const failed = report.arms.some(a => a.answer?.status === "error");
    status(`Three-way comparison complete. ${report.scoring_calls} new Jev calls; ${report.generation_calls} generation calls.${failed ? " Generation errors are shown; inspect each arm." : ""}`, failed);
  });
}
$("export-rankings").onclick = () => {
  const blob = new Blob([JSON.stringify(state.rankingReport, null, 2)], {type: "application/json"});
  const a = node("a"); a.href = URL.createObjectURL(blob); a.download = "rag-jev-rankings.json";
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
