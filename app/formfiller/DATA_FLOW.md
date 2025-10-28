# Form Filler — Data Flow Model

This document describes the runtime data flow for the Form Filler agentic AI system in `app/formfiller`.
It maps components, state variables (from `CHAT_REQUEST_VARS.md`), the decision points (orchestrator), and the typical message/state transformations as a request flows through the LangGraph.

Goals
- Provide a single reference for developers integrating UI, agents, and storage.
- Show where each `FormFillerState` variable is read/updated.
- Show sequence diagrams and important edge cases for robust integration.

---

## Components (short)
- Frontend UI — sends user requests (payload containing `user_message`, `form_fields` optionally) and receives `response_message`, `filled_fields`, etc.
- Orchestrator node — entry point in the LangGraph; decides which subflow to run.
- `initial_query_agent` — handles cases where only `user_message` is present (no `form_fields`). Uses `search_tool` (RAG) + LLM to return user-facing answer.
- `analyze_form` — maps `user_message` to `form_fields`, identifies `missing_fields`, produces `filled_fields` where possible.
- `process_field_input` — prompts for/validates the `current_field`, updates `filled_fields`, removes items from `missing_fields`.
- `search_tool` — wrapper around `ai_search_tool` used to fetch context for RAG.
- LLM executors (`analyze_form_executor`, `process_field_executor`, `initial_query_executor`) — the LLM chains or agents that produce outputs for each node.
- Checkpointer (MemorySaver / Redis) — persists runtime state for recovery and debugging.

---

## State variables and single-line mapping to components
- `user_message` — Input from UI; used by orchestrator, initial_query_agent, analyze_form, and process_field_input.
- `form_fields` — Input from UI or prior state; used by orchestrator and analyze_form; used to compute `filled_data_ids` and validate missing fields.
- `filled_fields` — Produced by `analyze_form` and `process_field_input`. Merged back into `form_fields` when producing final payload.
- `missing_fields` — Produced by `analyze_form`; consumed by `process_field_input` and UI.
- `current_field` — Set by analyze/agents to indicate what to ask the user next; used by UI to highlight.
- `conversation_history` — Updated across nodes; summarized before LLM calls (via `add_history_summary`).
- `status` — Controls routing and termination (`in_progress`, `awaiting_info`, `completed`).
- `response_message` — Short text for the UI; set by agents.
- `thread_id` — Persisted identifier for the session.

---

## End-to-end flow (simple sequence)

Below is the common case flows and transitions. Use these as the canonical sequences the graph implements.

1) Frontend -> Orchestrator (entry)
   - Payload (minimal): `{ user_message, form_fields? }`
   - Orchestrator checks `form_fields` presence/emptiness.

2A) If `form_fields` missing or empty -> `initial_query_agent`
   - `initial_query_agent` calls `search_tool(user_message)` (RAG)
   - Prepares prompt `user_message + search_results`
   - Calls `initial_query_executor` (LLM chain/agent)
   - On success: sets `response_message`, `search_results`, and `status = 'completed'` (ends flow)
   - UI receives `response_message` and may ask user for clarification or to provide a form to populate.

2B) If `form_fields` present -> `analyze_form`
   - `analyze_form` calls `search_tool({ user_message, formFields })` to fetch context or evidence.
   - Calls `analyze_form_executor` (LLM) with `message`, `formFields`, `search_results`.
   - LLM returns structured analysis (usually JSON-like) with `filled_fields` and `missing_fields` and optional `message`.
   - `analyze_form` updates `filled_fields` (append/dedupe), computes `missing_fields` filtered against `form_fields`, sets `current_field` to first missing item (if any), and sets `status = 'awaiting_info'` or `completed`.
   - If `missing_fields` non-empty: calls `process_field_input` to get values for the `current_field`.

3) `process_field_input` (loop until fields are filled)
   - Uses `process_field_executor` to validate/transform `user_message` into `current_field` value.
   - On success, append to `filled_fields`, remove from `missing_fields`, set `current_field` to next.
   - When no missing fields remain, set `status = 'completed'` and return final `filled_fields`.

4) Persist/Return
   - The final state (including `filled_fields`) is available to the caller. The UI can merge `filled_fields` into `form_fields` before submission.
   - The `checkpointer` may have stored intermediate snapshots for recovery.

---

## ASCII data-flow diagram

The diagram shows nodes and the primary variables passed along. Arrows annotate primary state changes.

Frontend
  |
  | send: `{ user_message, form_fields? }`
  v
Orchestrator (entry)
  |-- if form_fields present --> analyze_form
  |                              reads: `user_message, form_fields`
  |                              writes: `filled_fields, missing_fields, current_field, response_message, status`
  |
  |-- if form_fields empty ----> initial_query_agent
                                 reads: `user_message`
                                 calls: `search_tool(user_message)`
                                 writes: `response_message, status='completed', search_results`

analyze_form --> (may call) process_field_input --> updates `filled_fields` and `missing_fields` until done

LLM executors and search_tool are invoked by agents and are the main external calls.

Persistence: Checkpointer (MemorySaver / Redis) snapshots `FormFillerState` between node runs for recovery.

---

## Data transformations and schemas

- Input: UI request
  - Minimal: `{ "user_message": "..." }`
  - Form-based: `{ "user_message": "...", "form_fields": [ ... ] }`

- LLM output expectations (analyze_form):
  ```json
  {
    "filled_fields": [ {"data_id":"owner_name","fieldValue":"ACME Ltd"} ],
    "missing_fields": ["parcel_number"],
    "message": "I filled owner name. I still need parcel number."
  }
  ```

- LLM output (process_field_input) expected minimal shape:
  ```json
  {"success": true, "current_field_details": {"data_id":"parcel_number","fieldValue":"123-456"}, "message":"Thanks"}
  ```

- Final returned state to UI: contains `response_message`, `status`, `filled_fields`, and `form_fields` (optional merged result).

---

## Checkpoints, recovery, and retries

- The `checkpointer` (MemorySaver in development) should snapshot state prior to each node execution.
- On unexpected errors in a node, load last snapshot, set `status` to `awaiting_info` and include an error trace in `response_message` for debugging.
- For long-running flows or human-in-loop, persist `thread_id` and keep snapshots for resumption.

---

## Decision points & route rules (quick)
- Orchestrator: route to `analyze_form` only if `form_fields` exists and is non-empty.
- After `initial_query_agent`: mark `status = 'completed'` to end the graph for the invocation.
- `route_next_step`: only route to `analyze_form` if `status` is `awaiting_info` *and* `form_fields` present; otherwise end.

---

## Developer notes & edge cases
- Token limits: Do not pass the full `conversation_history` to LLMs without summarization.
- Mixed `missing_fields` format: accept both strings and dicts; use `data_id` when present.
- Dedupe by `data_id` when appending to `filled_fields`.
- UI merge strategy: prefer `filled_fields` -> merge into `form_fields` on the client before a submit action.
- RAG tuning: `search_tool` results are raw; agents should sanitize and weigh them in prompts.

---

## Quick check-list for implementers
- [ ] Ensure `orchestrator` is always the START node in the graph.
- [ ] Validate incoming `form_fields` for `data_id` presence early.
- [ ] Implement graceful error handling and snapshot restore via checkpointer.
- [ ] Add unit tests that simulate: (a) only user_message, (b) user_message + form_fields, (c) repeated process_field_input loops.

---

## Next steps I can help with
- Produce a Pydantic schema (`app/formfiller/schemas.py`) for `FormFillerState` and `FormField`.
- Add unit tests that run `compiled_graph.ainvoke` through the orchestrator with mocked executors.


---

Created as part of the developer documentation tasks. If you'd like a visual diagram (SVG/PNG) I can output a PlantUML or Mermaid diagram file that you can render locally.