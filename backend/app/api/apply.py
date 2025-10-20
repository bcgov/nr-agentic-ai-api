"""Apply endpoint persists state and signals readiness."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_cosmos_logger, get_redis_store, get_validator
from ..models.forms import ApplyRequest, ApplyResponse, ValidationSummary
from ..services.azure_clients import AzureCosmosLogger, AzureRedisStore
from ..services.validation import Validator, _normalise_value

router = APIRouter(tags=["apply"])


@router.post("/apply", response_model=ApplyResponse)
async def apply(
    payload: ApplyRequest,
    validator: Validator = Depends(get_validator),
    redis_store: AzureRedisStore = Depends(get_redis_store),
    cosmos_logger: AzureCosmosLogger = Depends(get_cosmos_logger),
) -> ApplyResponse:
    """Persist the provided form state and report readiness for the next step."""

    fields_payload = [field.dict() for field in payload.form_state]
    validation_result = validator.validate_page(payload.page.id, fields_payload)
    summary = ValidationSummary(**validation_result.dict())

    if not summary.ready:
        cosmos_logger.log_apply(
            session=payload.session.dict(),
            page=payload.page.dict(),
            fields=fields_payload,
            stored=False,
        )
        return ApplyResponse(
            stored=False,
            ready_for_next=False,
            validation=summary.dict(),
        )

    stored = await redis_store.store_page(
        payload.session.threadId,
        payload.page.dict(),
        fields_payload,
    )
    cosmos_logger.log_apply(
        session=payload.session.dict(),
        page=payload.page.dict(),
        fields=fields_payload,
        stored=stored,
    )
    applied_fields = [
        {"id": field.id, "value": _normalise_value(field.value)}
        for field in payload.form_state
    ]
    return ApplyResponse(
        stored=stored,
        ready_for_next=stored and summary.ready,
        applied_fields=applied_fields,
    )
