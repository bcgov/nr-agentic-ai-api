from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

try:
    import redis  # type: ignore[import]
except ModuleNotFoundError:
    redis = None  # type: ignore[assignment]
    _redis_available = False
else:
    _redis_available = True


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _default_tls(host: str, port: str) -> bool:
    if port == "6380":
        return True
    return host.endswith(".redis.cache.windows.net")


def _resolve_redis_url() -> str:
    explicit = os.getenv("REDIS_URL") or os.getenv("AZURE_REDIS_URL")
    if explicit:
        return explicit

    host = os.getenv("REDIS_HOST") or os.getenv("AZURE_REDIS_HOST")
    if not host:
        return "redis://localhost:6379/0"

    username = (
        os.getenv("REDIS_USERNAME")
        or os.getenv("REDIS_USER")
        or os.getenv("AZURE_REDIS_USERNAME")
        or os.getenv("AZURE_REDIS_USER")
    )
    password = os.getenv("REDIS_PASSWORD") or os.getenv("AZURE_REDIS_PASSWORD") or os.getenv("AZURE_REDIS_KEY")
    if not username and password and host.endswith(".redis.cache.windows.net"):
        # Azure Cache for Redis defaults to the "default" user when using keys
        username = "default"

    port = (
        os.getenv("REDIS_PORT")
        or os.getenv("AZURE_REDIS_PORT")
        or ("6380" if host.endswith(".redis.cache.windows.net") else "6379")
    )
    db = os.getenv("REDIS_DB", "0")
    use_tls = _env_flag("REDIS_USE_TLS", default=_default_tls(host, port))
    scheme = "rediss" if use_tls else "redis"

    auth = ""
    if username or password:
        user = quote_plus(username) if username else ""
        pwd = quote_plus(password) if password else ""
        if username and password:
            auth = f"{user}:{pwd}@"
        elif password and not username:
            auth = f":{pwd}@"
        else:
            auth = f"{user}@"

    query = os.getenv("REDIS_QUERY", "").strip()
    if query:
        query = f"?{query.lstrip('?')}"

    return f"{scheme}://{auth}{host}:{port}/{db}{query}"


REDIS_URL = _resolve_redis_url()
ENABLE_MEMORY = _env_flag("ENABLE_MEMORY")
SESSION_TTL_MIN = int(os.getenv("SESSION_TTL_MIN", "1440"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "12"))

_r: Optional[redis.Redis] = None
_in_memory_conversations: Dict[str, Dict[str, Any]] = {}


def _json_safe_copy(obj: Any) -> Any:
    """Recursively coerce objects into JSON-serializable primitives."""
    if hasattr(obj, "model_dump"):
        return _json_safe_copy(obj.model_dump())
    if isinstance(obj, dict):
        return {str(k): _json_safe_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe_copy(item) for item in obj]
    if isinstance(obj, tuple):
        return [_json_safe_copy(item) for item in obj]
    if isinstance(obj, set):
        return [_json_safe_copy(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _client() -> redis.Redis:
    """Singleton Redis client."""
    if not _redis_available or redis is None:
        raise RuntimeError(
            "Redis support is not available. Install the 'redis' package or disable "
            "memory features."
        )
    global _r
    if _r is None:
        _r = redis.from_url(REDIS_URL, decode_responses=True)
    return _r


@dataclass
class SessionContext:
    summary: Optional[str]
    last_turns: List[Dict[str, Any]]


def _k_prefix(session_id: str) -> str:
    return f"session:{session_id}"


@dataclass
class ConversationState:
    thread_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    field_values: Dict[str, Any] = field(default_factory=dict)
    form_fields: List[Dict[str, Any]] = field(default_factory=list)
    filled_fields: List[Dict[str, Any]] = field(default_factory=list)
    missing_fields: List[Dict[str, Any]] = field(default_factory=list)
    current_field: List[Dict[str, Any]] = field(default_factory=list)
    response_message: str = ""
    status: str = "awaiting_info"


def _conversation_state_from_dict(data: Optional[Dict[str, Any]]) -> ConversationState:
    if not data:
        return ConversationState()
    thread_id = data.get("thread_id") or data.get("threadId")
    if thread_id is not None:
        thread_id = str(thread_id)

    history_items: List[Dict[str, Any]] = []
    for item in data.get("conversation_history", []) or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if content is None:
            content = item.get("message")
        if role not in {"user", "assistant"}:
            continue
        history_items.append(
            {
                "role": role,
                "content": str(content) if content is not None else "",
            }
        )

    field_values = data.get("field_values")
    if not isinstance(field_values, dict):
        fallback_fields = data.get("filled_fields")
        if isinstance(fallback_fields, dict):
            field_values = dict(fallback_fields)
        else:
            field_values = {}
    else:
        field_values = dict(field_values)

    def _coerce_payload_list(raw: Any) -> List[Dict[str, Any]]:
        coerced: List[Dict[str, Any]] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    payload = dict(item)
                    if "fieldValue" not in payload and "field_value" in payload:
                        payload["fieldValue"] = payload.get("field_value")
                    coerced.append(payload)
        return coerced

    form_fields = _coerce_payload_list(data.get("form_fields"))
    if not form_fields:
        form_fields = _coerce_payload_list(data.get("field_descriptors"))

    filled_fields = _coerce_payload_list(data.get("filled_fields"))

    missing_raw = data.get("missing_fields")
    missing_fields = _coerce_payload_list(missing_raw)
    if not missing_fields and isinstance(missing_raw, list):
        for idx, item in enumerate(missing_raw):
            if isinstance(item, dict):
                missing_fields.append(dict(item))
            elif isinstance(item, str):
                text = item.strip()
                payload = {
                    "dataId": f"missing-{idx}",
                    "fieldLabel": text or f"Missing field {idx + 1}",
                    "validationMessage": text or None,
                }
                missing_fields.append(
                    {k: v for k, v in payload.items() if v is not None}
                )

    current_field = _coerce_payload_list(data.get("current_field"))

    message = data.get("response_message")
    if not isinstance(message, str):
        message = ""
    status = data.get("status")
    if not isinstance(status, str):
        status = "awaiting_info"
    status_map = {
        "in_progress": "awaiting_info",
        "complete": "ready_to_submit",
    }
    status = status_map.get(status, status)

    return ConversationState(
        thread_id=thread_id or None,
        conversation_history=history_items,
        field_values=field_values,
        form_fields=form_fields,
        filled_fields=filled_fields,
        missing_fields=missing_fields,
        current_field=current_field,
        response_message=message,
        status=status,
    )


def get_session(session_id: str) -> SessionContext:
    """Fetch summary and last K turns for a session."""
    if not ENABLE_MEMORY:
        return SessionContext(summary=None, last_turns=[])

    try:
        r = _client()
        pfx = _k_prefix(session_id)
        summary = r.get(f"{pfx}:summary")
        # Keep the most recent MAX_TURNS items
        turns = r.lrange(f"{pfx}:turns", 0, MAX_TURNS - 1) or []
        last_turns = [json.loads(x) for x in turns]
    except Exception:
        return SessionContext(summary=None, last_turns=[])

    return SessionContext(summary=summary, last_turns=last_turns)


def get_prompt_context(session_id: Optional[str]) -> Dict[str, Any]:
    """Shape used by agents: {'summary': str|None, 'last_turns': [..]}"""
    if not ENABLE_MEMORY or not session_id:
        return {}
    ctx = get_session(session_id)
    if ctx.summary is None and not ctx.last_turns:
        return {}
    return {"summary": ctx.summary, "last_turns": ctx.last_turns}


def get_conversation_state(session_id: Optional[str]) -> ConversationState:
    """Retrieve stored conversational context for a session (if any)."""
    if not session_id:
        return ConversationState()

    if ENABLE_MEMORY:
        try:
            r = _client()
            raw = r.get(f"{_k_prefix(session_id)}:conversation")
            if raw:
                return _conversation_state_from_dict(json.loads(raw))
        except Exception:
            # Redis misconfiguration should not break request flow
            pass
    else:
        cached = _in_memory_conversations.get(session_id)
        if cached:
            return _conversation_state_from_dict(cached)

    return ConversationState()


def set_conversation_state(session_id: Optional[str], state: ConversationState) -> None:
    """Persist conversational context for subsequent turns."""
    if not session_id:
        return

    history_payload: List[Dict[str, Any]] = []
    for item in state.conversation_history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if content is None:
                content = item.get("message")
        elif hasattr(item, "model_dump"):
            record = item.model_dump()
            role = record.get("role")
            content = record.get("content")
            if content is None:
                content = record.get("message")
        else:
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
            if content is None:
                content = getattr(item, "message", None)
        if role in {"user", "assistant"} and content is not None:
            history_payload.append({"role": role, "content": str(content)})

    payload = {
        "thread_id": state.thread_id,
        "conversation_history": history_payload,
        "field_values": dict(state.field_values),
        "form_fields": [_json_safe_copy(item) for item in state.form_fields],
        "filled_fields": [_json_safe_copy(item) for item in state.filled_fields],
        "missing_fields": [_json_safe_copy(item) for item in state.missing_fields],
        "current_field": [_json_safe_copy(item) for item in state.current_field],
        "response_message": state.response_message,
        "status": state.status,
    }
    payload["field_descriptors"] = [
        dict(item) for item in payload["form_fields"] if isinstance(item, dict)
    ]

    if ENABLE_MEMORY:
        try:
            r = _client()
            ttl = timedelta(minutes=SESSION_TTL_MIN)
            r.setex(
                f"{_k_prefix(session_id)}:conversation",
                ttl,
                json.dumps(payload, separators=(",", ":")),
            )
        except Exception:
            pass
    else:
        _in_memory_conversations[session_id] = {
            "thread_id": payload.get("thread_id"),
            "conversation_history": [
                dict(item) for item in payload["conversation_history"]
            ],
            "field_values": dict(payload.get("field_values", {})),
            "form_fields": [
                dict(item) for item in payload.get("form_fields", []) if isinstance(item, dict)
            ],
            "filled_fields": [
                dict(item) for item in payload.get("filled_fields", []) if isinstance(item, dict)
            ],
            "missing_fields": [
                dict(item) for item in payload.get("missing_fields", []) if isinstance(item, dict)
            ],
            "current_field": [
                dict(item) for item in payload.get("current_field", []) if isinstance(item, dict)
            ],
            "field_descriptors": [
                dict(item)
                for item in payload.get("field_descriptors", [])
                if isinstance(item, dict)
            ],
            "response_message": payload.get("response_message", ""),
            "status": payload.get("status", "awaiting_info"),
        }


def append_turn(session_id: Optional[str], user_fields: Dict[str, Any], merged_result: Dict[str, Any]) -> None:
    """Append a compact turn and refresh TTLs (best-effort; POC)."""
    if not ENABLE_MEMORY or not session_id:
        return
    r = _client()
    pfx = _k_prefix(session_id)
    blob = {
        "user_fields": _json_safe_copy(user_fields or {}),
        "result": _json_safe_copy(merged_result or {}),
    }
    pipe = r.pipeline()
    # LPUSH newest to head, then LTRIM to fixed window
    pipe.lpush(f"{pfx}:turns", json.dumps(blob, separators=(",", ":")))
    pipe.ltrim(f"{pfx}:turns", 0, MAX_TURNS - 1)
    ttl = timedelta(minutes=SESSION_TTL_MIN)
    pipe.expire(f"{pfx}:turns", ttl)
    pipe.expire(f"{pfx}:summary", ttl)
    pipe.execute()


def set_summary(session_id: Optional[str], summary: str) -> None:
    if not ENABLE_MEMORY or not session_id:
        return
    r = _client()
    pfx = _k_prefix(session_id)
    r.setex(f"{pfx}:summary", timedelta(minutes=SESSION_TTL_MIN), summary)


def delete_session(session_id: str) -> None:
    """Hard delete session keys."""
    if ENABLE_MEMORY:
        r = _client()
        pfx = _k_prefix(session_id)
        r.delete(f"{pfx}:summary")
        r.delete(f"{pfx}:turns")
        r.delete(f"{pfx}:conversation")
    _in_memory_conversations.pop(session_id, None)
