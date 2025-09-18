# src/orchestrator/orchestrator.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from app.src.models import (
    EnrichedField,
    Violation,
    AgentResult,
    OrchestratorRequest,
    OrchestratorResponse,
)

# Agents (updated in PR to accept context=...)
from app.src.agents.source_agent import run_source_agent
from app.src.agents.usage_agent import run_usage_agent
from app.src.agents.permits_agent import run_permits_agent

# Mapping helpers (used to compute clarifications)
from app.src.mapping.helpers import load_mapping, label_for

# Session memory (Redis)
from app.src.memory.session_store import get_prompt_context, append_turn


# ---------------------------- Internal Utilities ---------------------------- #

def _deep_merge_fields(
    base: Dict[str, EnrichedField],
    patch: Optional[Dict[str, EnrichedField]],
) -> Dict[str, EnrichedField]:
    """Merge EnrichedField dicts (patch wins)."""
    if not patch:
        return base
    merged = dict(base)
    for k, v in patch.items():
        merged[k] = v
    return merged


def _concat(a: Optional[List], b: Optional[List]) -> List:
    out: List = []
    if a:
        out.extend(a)
    if b:
        out.extend(b)
    return out


def _dedupe_preserve_order(strings: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in strings:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _coerce_enriched_fields(obj: Dict[str, Any]) -> Dict[str, EnrichedField]:
    """
    Convert a dict[str, Any] into dict[str, EnrichedField].
    Values can already be EnrichedField or plain dicts with 'value'/'enriched'.
    """
    out: Dict[str, EnrichedField] = {}
    for k, v in obj.items():
        if isinstance(v, EnrichedField):
            out[k] = v
        elif isinstance(v, dict):
            # Accept either {value: ..., enriched: {...}} or a bare {value: ...}
            out[k] = EnrichedField(
                value=v.get("value") if "value" in v else v.get("Value") if "Value" in v else v,
                enriched=v.get("enriched", {}) if isinstance(v.get("enriched", {}), dict) else {},
            )
        else:
            out[k] = EnrichedField(value=v)
    return out


def _serialize_enriched_fields(fields: Dict[str, EnrichedField]) -> Dict[str, Any]:
    """
    Make EnrichedField dict JSON-serializable for Redis persistence.
    """
    return {
        k: {"value": v.value, "enriched": v.enriched if isinstance(v.enriched, dict) else {}}
        for k, v in fields.items()
    }


def _coerce_violations(lst: Optional[List[Any]]) -> List[Violation]:
    out: List[Violation] = []
    if not lst:
        return out
    for item in lst:
        if isinstance(item, Violation):
            out.append(item)
        elif isinstance(item, dict):
            out.append(Violation(
                code=item.get("code", ""),
                severity=item.get("severity", "error"),
                fieldId=item.get("fieldId"),
                message=item.get("message"),
            ))
        else:
            # Fallback textual violation
            out.append(Violation(code=str(item), severity="error"))
    return out


def _coerce_clarifications(raw: Any) -> List[str]:
    clarifications: List[str] = []
    if not raw:
        return clarifications

    if isinstance(raw, (list, tuple, set)):
        items = raw
    else:
        items = [raw]

    for item in items:
        if item is None:
            continue
        text = item if isinstance(item, str) else str(item)
        text = text.strip()
        if text:
            clarifications.append(text)
    return clarifications


def _coerce_agent_result(res: Union[AgentResult, Dict[str, Any], None]) -> AgentResult:
    """
    Accepts:
      - AgentResult
      - dict with keys updatedFields/violations/references
      - dict that is *just* the updatedFields payload
      - None
    Returns a normalized AgentResult.
    """
    if isinstance(res, AgentResult):
        return res

    if res is None:
        return AgentResult()

    if isinstance(res, dict):
        if any(k in res for k in ("updatedFields", "violations", "references", "clarifications")):
            updated = _coerce_enriched_fields(res.get("updatedFields", {}) or {})
            viols = _coerce_violations(res.get("violations"))
            refs = res.get("references")
            clarifications = _coerce_clarifications(res.get("clarifications"))
            return AgentResult(
                updatedFields=updated,
                violations=viols,
                references=refs,
                clarifications=clarifications,
            )
        else:
            # Treat the whole dict as updatedFields
            updated = _coerce_enriched_fields(res)
            return AgentResult(updatedFields=updated, violations=[], references=None)

    # Unexpected type; return empty
    return AgentResult()


def _missing_required_questions(fields: Dict[str, EnrichedField]) -> List[str]:
    """
    Turn missing required fields (from mapping.json) into natural-language questions.
    Adds simple conditional logic for hydraulic-connection when well == Yes.
    """
    mapping = load_mapping() or {}
    fmap: Dict[str, Dict[str, Any]] = {}
    if isinstance(mapping, dict):
        fmap = mapping.get("fields") if isinstance(mapping.get("fields"), dict) else mapping
        if not isinstance(fmap, dict):
            fmap = {}

    questions: List[str] = []

    # 1) Base required fields (mapping.json: { fields.<id>.required = true })
    for fid, meta in fmap.items():
        if not isinstance(meta, dict):
            continue
        if not meta.get("required"):
            continue

        current: EnrichedField = fields.get(fid) or EnrichedField()
        val = current.value
        if val in (None, ""):
            label = label_for(fid) or fid.replace("-", " ")
            questions.append(f"Please provide {label}.")

    # 2) Conditional: if source-water-well == Yes -> ask for hydraulic-connection
    sw = (fields.get("source-water-well") or EnrichedField()).value
    if sw == "Yes":
        hc = (fields.get("hydraulic-connection") or EnrichedField()).value
        if hc in (None, ""):
            questions.append(
                "Is there a hydraulic connection to a stream or surface water? (Yes/No)"
            )

    # de-duplicate, preserve order
    return _dedupe_preserve_order(questions)


# ------------------------------- Orchestrator ------------------------------- #

def orchestrate(req: OrchestratorRequest, session_id: Optional[str] = None) -> OrchestratorResponse:
    """
    Pipeline:
      1) Source agent (source-water-well, hydraulic-connection, etc.)
      2) Usage agent  (purpose, annual-quantity, etc.)
      3) Permits agent (high-level compliance)
      4) Compute clarifications from effective fields (req + updates)
      5) Persist turn (best-effort) if session memory is enabled

    Agents may return either AgentResult or plain dicts; we normalize them here.
    """
    # Start with the request fields
    fields: Dict[str, EnrichedField] = _coerce_enriched_fields(dict(req.fields or {}))
    serialized_request_fields = _serialize_enriched_fields(fields)

    # Fetch session context for agents (no-op dict if disabled/missing)
    context: Dict[str, Any] = get_prompt_context(session_id)

    # ---- 1. Source agent
    src_res = _coerce_agent_result(run_source_agent(fields, context=context))
    fields = _deep_merge_fields(fields, src_res.updatedFields)

    # ---- 2. Usage agent
    usage_res = _coerce_agent_result(run_usage_agent(fields, context=context))
    fields = _deep_merge_fields(fields, usage_res.updatedFields)

    # ---- 3. Permits agent
    permits_res = _coerce_agent_result(run_permits_agent(fields, context=context))
    fields = _deep_merge_fields(fields, permits_res.updatedFields)

    # Collect violations (order: source -> usage -> permits)
    violations: List[Violation] = _concat(src_res.violations, usage_res.violations)
    violations = _concat(violations, permits_res.violations)

    # Collect references (snippets/citations)
    references = None
    for res in (src_res, usage_res, permits_res):
        if res.references:
            references = (references or []) + res.references

    # ---- 4. Clarifications from the effective view of fields
    agent_clarifications: List[str] = []
    for res in (src_res, usage_res, permits_res):
        agent_clarifications.extend(res.clarifications or [])

    clarifications = _dedupe_preserve_order(
        agent_clarifications + _missing_required_questions(fields)
    )

    # ---- 5. Persist this turn to Redis (best-effort; swallow errors)
    try:
        append_turn(
            session_id=session_id,
            user_fields=serialized_request_fields,
            merged_result=_serialize_enriched_fields(fields),
        )
    except Exception:
        # Do not break the request path if Redis is misconfigured/unavailable
        pass

    return OrchestratorResponse(
        updatedFields=fields,          # final merged fields after all agents
        violations=violations,
        references=references,
        clarifications=clarifications, # for chat/UX
    )
