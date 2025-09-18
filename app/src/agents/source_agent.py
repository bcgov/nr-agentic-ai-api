from typing import Dict, Any, List
from app.src.agents.common import finalize_agent_result
from app.src.llm.client import json_complete

def run_source_agent(payload_fields: Dict[str, Any], context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    # 1) LLM enrichment (no-op if USE_GPT4O=0 or AOAI not configured)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are the Source agent. Normalize source-related fields and propose clarifications. Output JSON with keys: updatedFields, clarifications."},
        {"role": "user", "content": f"payload_fields={payload_fields}"},
        {"role": "user", "content": f"session_context={context or {}}"},
    ]
    response_format = {
        "name": "SourceAgentSchema",
        "schema": {
            "type": "object",
            "properties": {
                "updatedFields": {"type": "object", "additionalProperties": {"type": "object"}},
                "clarifications": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["updatedFields"]
        },
        "strict": True
    }
    llm_out = json_complete(messages, response_format)
    result = {
        "updatedFields": llm_out.get("updatedFields", {}),
        "clarifications": llm_out.get("clarifications", []),
    }

    # 2) Rules + (optional) retrieval citations
    merged_fields = payload_fields | result["updatedFields"]
    return finalize_agent_result(result, merged_fields)
