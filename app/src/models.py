# src/models.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ------------------------- Field / Helper Models ------------------------- #

class EnrichedField(BaseModel):
    """
    Canonical field container used across the system.
    .get(...) is provided so code that treats this like a dict continues to work.
    """
    value: Optional[Any] = None
    enriched: Dict[str, Any] = Field(default_factory=dict)

    # Dict-like compatibility for older code/tests: field.get("value") etc.
    def get(self, key: str, default: Any = None) -> Any:
        if key == "value":
            return self.value if self.value is not None else default
        if key == "enriched":
            return self.enriched if self.enriched is not None else default
        # allow getattr fallback
        return getattr(self, key, default)


class Violation(BaseModel):
    code: str
    severity: Literal["error", "warn", "info"] = "error"
    fieldId: Optional[str] = None
    message: Optional[str] = None

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v: Any) -> str:
        """Normalize various severity synonyms to error/warn/info."""
        if isinstance(v, str):
            lookup = {
                "error": "error",
                "err": "error",
                "critical": "error",
                "major": "error",
                "warn": "warn",
                "warning": "warn",
                "minor": "warn",
                "info": "info",
                "information": "info",
                "informational": "info",
            }
            return lookup.get(v.lower(), "error")
        return "error"


class AgentResult(BaseModel):
    """
    Standard result shape returned by agents (source, usage, permits).
    """
    updatedFields: Dict[str, EnrichedField] = Field(default_factory=dict)
    violations: List[Violation] = Field(default_factory=list)
    references: Optional[List[Dict[str, Any]]] = None
    clarifications: List[str] = Field(default_factory=list)


# --------------------------- Orchestrator I/O ---------------------------- #

class OrchestratorRequest(BaseModel):
    """
    Public API request for /orchestrate.
    """
    fields: Dict[str, EnrichedField] = Field(default_factory=dict)


class OrchestratorResponse(BaseModel):
    """
    Public API response for /orchestrate.
    """
    updatedFields: Dict[str, EnrichedField] = Field(default_factory=dict)
    violations: List[Violation] = Field(default_factory=list)
    references: Optional[List[Dict[str, Any]]] = None

    # Natural-language questions for missing required info.
    clarifications: List[str] = Field(default_factory=list)


# --------------------------- Conversation Models --------------------------- #


class ConversationField(BaseModel):
    """Descriptor for a single conversational form field."""

    data_id: str = Field(alias="dataId")
    field_label: Optional[str] = Field(default=None, alias="fieldLabel")
    field_type: Optional[str] = Field(default=None, alias="fieldType")
    field_value: Any = Field(default=None, alias="fieldValue")
    options: Optional[List[Any]] = None
    is_required: Optional[bool] = Field(default=None, alias="isRequired")
    validation_message: Optional[str] = Field(
        default=None, alias="validationMessage"
    )

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _normalize_identifiers(cls, data: Any) -> Any:
        """Accept multiple identifier spellings and legacy value shapes."""
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        identifier = (
            normalized.get("data_id")
            or normalized.get("dataId")
            or normalized.get("fieldId")
            or normalized.get("id")
        )
        if identifier is None:
            raise ValueError("ConversationField requires a data identifier")
        normalized["data_id"] = str(identifier)

        # Support legacy {value: ..., enriched: ...} payloads.
        if "field_value" not in normalized and "fieldValue" not in normalized:
            if "value" in normalized:
                normalized.setdefault("field_value", normalized.get("value"))
        return normalized


class ConversationTurn(BaseModel):
    """Single utterance exchanged during a conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(
        default="",
        validation_alias=AliasChoices("content", "message"),
    )

    model_config = ConfigDict(populate_by_name=True)


class ConversationRequest(BaseModel):
    """Incoming request payload for the conversational orchestrator."""

    thread_id: Optional[str] = Field(default=None, alias="threadId")
    user_message: Optional[str] = None
    form_fields: List[ConversationField] = Field(default_factory=list)
    conversation_history: List[ConversationTurn] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_form_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw_fields = data.get("form_fields")
        if isinstance(raw_fields, dict):
            coerced: List[Dict[str, Any]] = []
            for key, value in raw_fields.items():
                payload: Dict[str, Any]
                if isinstance(value, dict):
                    payload = dict(value)
                    if "fieldValue" not in payload and "field_value" not in payload:
                        if "value" in value:
                            payload.setdefault("fieldValue", value.get("value"))
                    if "enriched" in value and isinstance(value.get("enriched"), dict):
                        payload.setdefault("enriched", value.get("enriched"))
                elif hasattr(value, "value"):
                    payload = {"fieldValue": getattr(value, "value", None)}
                    enriched = getattr(value, "enriched", None)
                    if isinstance(enriched, dict):
                        payload["enriched"] = enriched
                else:
                    payload = {"fieldValue": value}
                payload.setdefault("data_id", key)
                coerced.append(payload)
            data["form_fields"] = coerced
        elif raw_fields is None:
            data["form_fields"] = []
        return data


class ConversationResponse(BaseModel):
    """Response envelope for the conversational orchestrator endpoint."""

    thread_id: str = Field(default="", alias="threadId")
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    form_fields: List[ConversationField] = Field(default_factory=list)
    filled_fields: List[ConversationField] = Field(default_factory=list)
    missing_fields: List[ConversationField] = Field(default_factory=list)
    current_field: List[ConversationField] = Field(default_factory=list)
    response_message: str = ""
    status: Literal["awaiting_info", "ready_to_submit", "error"] = "awaiting_info"

    model_config = ConfigDict(populate_by_name=True)
