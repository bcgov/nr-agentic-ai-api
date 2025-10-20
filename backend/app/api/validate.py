"""Validation endpoint exposing deterministic policy checks."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_cosmos_logger, get_validator
from ..models.forms import ValidateRequest, ValidationSummary
from ..services.azure_clients import AzureCosmosLogger
from ..services.validation import PageValidationResult, Validator

router = APIRouter(tags=["validate"])


@router.post("/validate", response_model=ValidationSummary)
async def validate(
    payload: ValidateRequest,
    validator: Validator = Depends(get_validator),
    cosmos_logger: AzureCosmosLogger = Depends(get_cosmos_logger),
) -> ValidationSummary:
    """Validate the provided form state against schema-driven rules."""

    fields_payload = [field.dict() for field in payload.form_state]
    result: PageValidationResult = validator.validate_page(payload.page.id, fields_payload)
    cosmos_logger.log_validation(payload.session.dict(), payload.page.dict(), result)
    return ValidationSummary(**result.dict())
