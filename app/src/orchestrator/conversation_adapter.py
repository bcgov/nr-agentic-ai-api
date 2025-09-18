"""Utilities for adapting conversational requests to the orchestrator pipeline."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import ValidationError

from app.src.models import (
    ConversationField,
    ConversationRequest,
    ConversationResponse,
    ConversationTurn,
    EnrichedField,
    OrchestratorRequest,
    OrchestratorResponse,
    Violation,
)
from app.src.memory.session_store import ConversationState


def _normalize_form_fields(fields: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Ensure fields are shaped like {id: {value: ..., enriched: {...}}}."""
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, value in (fields or {}).items():
        if isinstance(value, EnrichedField):
            normalized[key] = {
                "value": value.value,
                "enriched": value.enriched if isinstance(value.enriched, dict) else {},
            }
        elif isinstance(value, dict):
            enriched = value.get("enriched") if isinstance(value.get("enriched"), dict) else {}
            normalized[key] = {
                "value": value.get("value"),
                "enriched": enriched,
            }
        else:
            normalized[key] = {"value": value}
    return normalized


def _coerce_history(turns: List[Any]) -> List[ConversationTurn]:
    history: List[ConversationTurn] = []
    for turn in turns or []:
        if isinstance(turn, ConversationTurn):
            history.append(turn)
            continue
        if isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content")
            if content is None:
                content = turn.get("message")
        else:
            role = getattr(turn, "role", None)
            content = getattr(turn, "content", None)
            if content is None:
                content = getattr(turn, "message", None)
        if role not in {"user", "assistant"}:
            continue
        if content is None:
            content = ""
        history.append(ConversationTurn(role=role, content=str(content)))
    return history


def _serialize_enriched_fields(fields: Dict[str, EnrichedField]) -> Dict[str, Dict[str, Any]]:
    return {
        key: {
            "value": field.value,
            "enriched": field.enriched if isinstance(field.enriched, dict) else {},
        }
        for key, field in (fields or {}).items()
    }


def _coerce_descriptor_list(raw: Any) -> List[ConversationField]:
    descriptors: List[ConversationField] = []
    if not raw:
        return descriptors

    items: Iterable[Any]
    if isinstance(raw, dict):
        items = raw.values()
    else:
        items = raw

    for item in items:
        if isinstance(item, ConversationField):
            descriptors.append(item)
            continue
        if isinstance(item, dict):
            try:
                descriptors.append(ConversationField.model_validate(item))
            except ValidationError:
                continue
        else:
            continue
    return descriptors


def _clone_fields(fields: List[ConversationField]) -> List[ConversationField]:
    return [field.model_copy() for field in fields]


def _merge_descriptor_lists(
    persisted: List[ConversationField], incoming: List[ConversationField]
) -> List[ConversationField]:
    merged: List[ConversationField] = []
    index: Dict[str, int] = {}

    for field in persisted:
        if field.data_id in index:
            continue
        index[field.data_id] = len(merged)
        merged.append(field)

    for field in incoming:
        idx = index.get(field.data_id)
        if idx is not None:
            merged[idx] = field
        else:
            index[field.data_id] = len(merged)
            merged.append(field)

    return merged


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _descriptor_payload(field: ConversationField) -> Dict[str, Any]:
    payload = field.model_dump(by_alias=True, exclude_none=True)
    payload["dataId"] = field.data_id
    payload["fieldValue"] = field.field_value
    return payload


def _resolve_validation_message(
    field: ConversationField, clarifications: List[str]
) -> str:
    label = (field.field_label or "").strip()
    data_id = (field.data_id or "").strip()
    lowered_label = label.lower()
    lowered_id = data_id.lower()

    for idx, text in enumerate(list(clarifications)):
        if not isinstance(text, str):
            continue
        candidate = text.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered_label and lowered_label in lowered:
            clarifications.pop(idx)
            return candidate
        if lowered_id and lowered_id in lowered:
            clarifications.pop(idx)
            return candidate

    for idx, text in enumerate(list(clarifications)):
        if not isinstance(text, str):
            continue
        candidate = text.strip()
        if not candidate:
            continue
        clarifications.pop(idx)
        return candidate

    display = label or data_id.replace("-", " ")
    if display:
        return f"Please provide {display}."
    return "Please provide the required information."


def _assistant_message(clarifications: List[str]) -> str:
    questions = [q for q in clarifications if isinstance(q, str) and q.strip()]
    if not questions:
        return "All required information has been collected."
    if len(questions) == 1:
        return questions[0]
    bullet_list = "\n".join(f"- {q}" for q in questions)
    return f"I still need the following information:\n{bullet_list}"


def _violation_message(violations: List[Violation]) -> str:
    messages: List[str] = []
    for violation in violations:
        if getattr(violation, "severity", "error") != "error":
            continue
        text = ""
        message = getattr(violation, "message", None)
        if isinstance(message, str) and message.strip():
            text = message.strip()
        else:
            field_id = getattr(violation, "fieldId", None)
            code = getattr(violation, "code", None)
            if field_id and code:
                text = f"{field_id}: {code}"
            elif field_id:
                text = str(field_id)
            elif code:
                text = str(code)
        if text:
            messages.append(text)

    if not messages:
        return "A validation error prevented completion."
    if len(messages) == 1:
        return messages[0]
    bullet_list = "\n".join(f"- {msg}" for msg in messages)
    return f"I encountered the following issues:\n{bullet_list}"


def prepare_conversation(
    req: ConversationRequest,
    state: ConversationState,
) -> Tuple[
    OrchestratorRequest,
    List[ConversationTurn],
    bool,
    List[ConversationField],
]:
    """Merge stored state with the new request before hitting the orchestrator."""
    stored_history = state.conversation_history or [
        turn.model_dump() for turn in req.conversation_history
    ]
    history = _coerce_history(stored_history)

    appended_user = False
    if req.user_message:
        history.append(ConversationTurn(role="user", content=req.user_message))
        appended_user = True

    stored_descriptor_source: Any = getattr(state, "form_fields", None)
    if not stored_descriptor_source:
        stored_descriptor_source = getattr(state, "field_descriptors", None)

    persisted_descriptors = _clone_fields(
        _coerce_descriptor_list(stored_descriptor_source or [])
    )
    incoming_descriptors = _clone_fields(_coerce_descriptor_list(req.form_fields))
    combined_descriptors = _merge_descriptor_lists(persisted_descriptors, incoming_descriptors)

    stored_field_values: Any = getattr(state, "field_values", None)
    if stored_field_values is None:
        stored_field_values = getattr(state, "filled_fields", {})

    fields = _normalize_form_fields(stored_field_values)

    for field in incoming_descriptors:
        existing = fields.get(field.data_id, {})
        existing_enriched = (
            existing.get("enriched")
            if isinstance(existing.get("enriched"), dict)
            else {}
        )
        override_enriched = (
            field.model_extra.get("enriched")
            if hasattr(field, "model_extra")
            else None
        )
        if not isinstance(override_enriched, dict):
            override_enriched = None

        value_provided = "field_value" in getattr(field, "model_fields_set", set())
        payload_value = existing.get("value")
        if value_provided or field.data_id not in fields:
            payload_value = field.field_value

        payload_enriched = override_enriched if override_enriched is not None else existing_enriched

        fields[field.data_id] = {
            "value": payload_value,
            "enriched": payload_enriched if isinstance(payload_enriched, dict) else {},
        }

    orch_request = OrchestratorRequest(fields=fields)
    return orch_request, history, appended_user, combined_descriptors


def build_conversation_response(
    orch_response: OrchestratorResponse,
    history: List[ConversationTurn],
    appended_user: bool,
    field_descriptors: List[ConversationField],
    thread_id: Optional[str] = None,
) -> Tuple[ConversationResponse, ConversationState]:
    """Convert orchestrator output into conversation response + new state."""

    clarifications = [str(item) for item in (orch_response.clarifications or []) if item is not None]
    clarification_pool = list(clarifications)
    response_history = list(history)

    error_violations: List[Violation] = []
    for violation in list(orch_response.violations or []):
        if not isinstance(violation, Violation):
            try:
                violation = Violation.model_validate(violation)
            except ValidationError:
                continue
        if getattr(violation, "severity", "error") == "error":
            error_violations.append(violation)

    serialized_fields = _serialize_enriched_fields(orch_response.updatedFields)

    form_field_models: List[ConversationField] = []
    filled_field_models: List[ConversationField] = []
    missing_field_models: List[ConversationField] = []

    for descriptor in field_descriptors:
        if not isinstance(descriptor, ConversationField):
            try:
                descriptor = ConversationField.model_validate(descriptor)
            except ValidationError:
                continue
        value_record = serialized_fields.get(descriptor.data_id)
        new_value = descriptor.field_value
        if isinstance(value_record, dict) and "value" in value_record:
            new_value = value_record.get("value")
        descriptor_copy = descriptor.model_copy(update={"field_value": new_value})
        form_field_models.append(descriptor_copy)
        if _has_value(new_value):
            filled_field_models.append(descriptor_copy)
        else:
            validation_message = _resolve_validation_message(
                descriptor_copy, clarification_pool
            )
            missing_field_models.append(
                descriptor_copy.model_copy(
                    update={"validation_message": validation_message}
                )
            )

    # Surface any remaining clarifications even if they could not be matched.
    placeholder_index = 0
    for remaining in list(clarification_pool):
        text = remaining.strip()
        if not text:
            clarification_pool.remove(remaining)
            continue
        placeholder = ConversationField(
            dataId=f"clarification-{placeholder_index}",
            fieldLabel=text,
            fieldValue=None,
            validationMessage=text,
        )
        placeholder_index += 1
        missing_field_models.append(placeholder)
        clarification_pool.remove(remaining)

    missing_messages = [
        field.validation_message
        for field in missing_field_models
        if isinstance(field.validation_message, str) and field.validation_message.strip()
    ]

    if missing_field_models:
        status = "awaiting_info"
        response_message = _assistant_message(missing_messages)
    elif error_violations:
        status = "error"
        response_message = _violation_message(error_violations)
    else:
        status = "ready_to_submit"
        response_message = _assistant_message([])

    if appended_user or missing_field_models or error_violations:
        response_history.append(
            ConversationTurn(role="assistant", content=response_message)
        )

    current_field_models = missing_field_models[:1]

    response = ConversationResponse(
        thread_id=thread_id or "",
        conversation_history=response_history,
        form_fields=form_field_models,
        filled_fields=filled_field_models,
        missing_fields=missing_field_models,
        current_field=current_field_models,
        response_message=response_message,
        status=status,
    )

    state = ConversationState(
        thread_id=thread_id or "",
        conversation_history=[turn.model_dump() for turn in response_history],
        field_values=serialized_fields,
        form_fields=[_descriptor_payload(field) for field in form_field_models],
        filled_fields=[_descriptor_payload(field) for field in filled_field_models],
        missing_fields=[_descriptor_payload(field) for field in missing_field_models],
        current_field=[_descriptor_payload(field) for field in current_field_models],
        response_message=response_message,
        status=status,
    )
    return response, state
