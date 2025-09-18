import os
from typing import Any, Dict
from app.src.rules.engine import RulesEngine
from app.src.rules.loader import load_rules
from app.src.retrieval.client import retrieve
from app.src.mapping.helpers import label_for

def finalize_agent_result(agent_result: Dict[str, Any], merged_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    - Run rules engine and apply updates (normalized values, units)
    - Attach citations when RAG_MODE != 'off'
    """
    # 1) Rules
    engine = RulesEngine(load_rules())
    rr = engine.evaluate({"fields": merged_fields})
    agent_result.setdefault("violations", []).extend(rr["violations"])

    for u in rr["updates"]:
        fid = u["fieldId"]
        ef = agent_result["updatedFields"].setdefault(fid, {"enriched": {}})
        ef["value"] = u["value"]
        meta = (u.get("meta") or {})
        if meta:
            ef.setdefault("enriched", {}).update(meta)

    # 2) Citations (RAG)
    rag_mode = os.getenv("RAG_MODE", "off")
    if rag_mode != "off":
        for fid, ef in agent_result.get("updatedFields", {}).items():
            try:
                q = f"{label_for(fid)} guidance"
                res = retrieve(query=q, scope="bc-water")
                ef.setdefault("enriched", {})["citations"] = (res.get("snippets") or [])[:3]
            except Exception:
                # log-only; don't show user-facing errors for retrieval failures
                pass

    return agent_result
