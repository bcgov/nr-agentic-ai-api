"""Pydantic models describing the form state payload exchanged with the frontend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FormField(BaseModel):
    id: str
    label: Optional[str] = None
    type: Optional[str] = None
    required: bool = False
    value: Any = None
    visibleWhen: Optional[str] = None
    helpRef: Optional[str] = None


class FormPage(BaseModel):
    id: str
    title: Optional[str] = None
    fields: List[FormField] = Field(default_factory=list)


class SessionContext(BaseModel):
    threadId: Optional[str] = None
    applicant: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session: SessionContext
    page: FormPage
    form_state: List[FormField] = Field(..., alias="fields")
    history: List[ChatMessage] = Field(default_factory=list)
    prompt: Optional[str] = None


class ChatResponse(BaseModel):
    messages: List[ChatMessage]
    sources: List[str] = Field(default_factory=list)


class ValidateRequest(BaseModel):
    session: SessionContext
    page: FormPage
    form_state: List[FormField] = Field(..., alias="fields")


class ApplyRequest(BaseModel):
    session: SessionContext
    page: FormPage
    form_state: List[FormField] = Field(..., alias="fields")
    confirmed: bool = False


class ApplyResponse(BaseModel):
    stored: bool
    ready_for_next: bool
    validation: Optional[Dict[str, Any]] = None
    applied_fields: List[Dict[str, Any]] = Field(default_factory=list)


class ValidationSummary(BaseModel):
    blockingMissingRequired: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    advisories: List[str] = Field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.blockingMissingRequired and not self.errors
