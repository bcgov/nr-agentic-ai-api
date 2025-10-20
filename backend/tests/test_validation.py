from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.services.schema_loader import SchemaLoader
from backend.app.services.validation import Validator

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "app" / "schemas"


@pytest.fixture(scope="module")
def validator() -> Validator:
    loader = SchemaLoader(SCHEMA_DIR)
    return Validator(loader)


def test_groundwater_requires_aquifer_and_valid_well_tag(validator: Validator) -> None:
    data = [
        {"id": "source_type", "value": "groundwater"},
        {"id": "aquifer_name", "value": ""},
        {"id": "well_tag_number", "value": "12a"},
    ]
    result = validator.validate_page("step-3", data)
    assert "aquifer_name" in result.blocking_missing_required
    assert any("format" in error.lower() for error in result.errors)

    data[1]["value"] = "Nanaimo Aquifer"
    data[2]["value"] = "1234567"
    result = validator.validate_page("step-3", data)
    assert not result.blocking_missing_required
    assert not result.errors


def test_irrigation_requires_positive_area_and_quantity(validator: Validator) -> None:
    data = [
        {"id": "purpose", "value": "irrigation"},
        {"id": "irrigated_area_ha", "value": "0"},
        {"id": "quantity_value", "value": "0"},
        {"id": "quantity_unit", "value": "invalid"},
    ]
    result = validator.validate_page("step-4", data)
    assert any("irrigated area" in error.lower() for error in result.errors)
    assert any("quantity" in error.lower() for error in result.errors)

    data[1]["value"] = "2.5"
    data[2]["value"] = "10"
    data[3]["value"] = "m3_per_day"
    result = validator.validate_page("step-4", data)
    assert not result.blocking_missing_required
    assert not result.errors


def test_visibility_handles_surface_stream_requirement(validator: Validator) -> None:
    data = [
        {"id": "source_type", "value": "surface"},
        {"id": "stream_name", "value": ""},
    ]
    result = validator.validate_page("step-3", data)
    assert "stream_name" in result.blocking_missing_required

    data[1]["value"] = "Koksilah River"
    result = validator.validate_page("step-3", data)
    assert not result.blocking_missing_required
    assert not result.errors
