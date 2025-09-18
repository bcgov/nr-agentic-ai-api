from typing import Dict, Any, List
from app.src.agents.common import finalize_agent_result
from app.src.llm.client import json_complete

def run_permits_agent(payload_fields: Dict[str, Any], context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are the Permits agent. Check compliance hints and fill derived flags. Output JSON: updatedFields, clarifications."},
        {"role": "user", "content": f"payload_fields={payload_fields}"},
        {"role": "user", "content": f"session_context={context or {}}"},
    ]
    response_format = {
        "name": "PermitsAgentSchema",
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

    merged_fields = payload_fields | result["updatedFields"]
    return finalize_agent_result(result, merged_fields)
