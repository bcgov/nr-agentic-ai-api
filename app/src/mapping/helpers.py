import json
import os
from functools import lru_cache
from typing import Any, Dict

MAPPING_PATH = os.getenv(
    "MAPPING_PATH", os.path.join(os.path.dirname(__file__), "mapping.json")
)


@lru_cache(maxsize=1)
def load_mapping() -> Dict[str, Dict[str, Any]]:
    """Load the mapping file and normalize to {"fields": {...}}.

    Returns an empty mapping if the file is missing or malformed.
    """
    try:
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"fields": {}}

    if not isinstance(data, dict):
        return {"fields": {}}

    fields = data.get("fields") if isinstance(data.get("fields"), dict) else None
    if fields is not None:
        return {"fields": fields}

    # No top-level "fields" key; treat the entire object as fields.
    return {"fields": data}


def label_for(field_id: str) -> str:
    fields = load_mapping().get("fields", {})
    meta = fields.get(field_id) or {}
    return meta.get("label", field_id)

