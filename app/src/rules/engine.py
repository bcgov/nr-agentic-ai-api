from __future__ import annotations

import re
from typing import Any, Dict, List


class RulesEngine:
    def __init__(self, spec: Dict[str, Any] | None = None):
        self.spec = spec or {}

    def evaluate(self, enriched_json: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        fields = (enriched_json or {}).get("fields", {}) or {}
        violations: List[Dict[str, Any]] = []
        updates: List[Dict[str, Any]] = []

        number_pattern = re.compile(r"[-+]?\d[\d_,]*(?:\.\d+)?")

        def field_value(field_id: str, key: str = "value") -> Any:
            return (fields.get(field_id) or {}).get(key)

        def is_present(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return value.strip() != ""
            return True

        def add_violation(code: str, field_id: str | None, message: str | None, severity: str = "error") -> None:
            violations.append(
                {
                    "code": code,
                    "severity": severity,
                    "fieldId": field_id,
                    "message": message,
                }
            )

        def evaluate_condition(condition: Dict[str, Any] | None) -> bool:
            if not condition:
                return True

            if "all" in condition:
                return all(evaluate_condition(child) for child in condition.get("all", []) or [])
            if "any" in condition:
                return any(evaluate_condition(child) for child in condition.get("any", []) or [])

            field_id = condition.get("field")
            value = field_value(field_id) if field_id else None

            if "equals" in condition:
                return value == condition.get("equals")
            if "not_equals" in condition:
                return value != condition.get("not_equals")
            if "in" in condition:
                options = condition.get("in") or []
                return value in options
            if "present" in condition:
                expected = bool(condition.get("present"))
                presence = is_present(value)
                return presence if expected else not presence

            return False

        # Field normalizations driven by spec
        for field_id, config in (self.spec.get("fields") or {}).items():
            if not isinstance(config, dict):
                continue

            field_type = config.get("type")
            raw_value = field_value(field_id)

            if field_type == "number" and raw_value is not None:
                if isinstance(raw_value, str):
                    if raw_value.strip() == "":
                        continue
                    match = number_pattern.search(raw_value)
                    parsed_value = None
                    if match:
                        try:
                            parsed_value = float(match.group(0).replace(",", "").replace("_", ""))
                        except ValueError:
                            parsed_value = None
                    if parsed_value is not None:
                        update: Dict[str, Any] = {"fieldId": field_id, "value": parsed_value}
                        units = config.get("units")
                        if units:
                            update["meta"] = {"units": units}
                        updates.append(update)
                    else:
                        messages = config.get("messages") or {}
                        message = messages.get("parse_error") or "Enter a valid number."
                        severity = messages.get("parse_error_severity", "error")
                        code = messages.get("parse_error_code") or f"{field_id.replace('-', '_').upper()}_PARSE"
                        add_violation(code, field_id, message, severity)

        # Rule assertions driven by spec
        for rule in self.spec.get("rules", []) or []:
            if not isinstance(rule, dict):
                continue

            if not evaluate_condition(rule.get("when")):
                continue

            assertion = rule.get("assert")
            if evaluate_condition(assertion):
                continue

            field_id = None
            if isinstance(assertion, dict):
                field_id = assertion.get("field")
            field_id = field_id or rule.get("fieldId")

            add_violation(
                code=rule.get("id", "RULE_VIOLATION"),
                field_id=field_id,
                message=rule.get("message"),
                severity=rule.get("severity", "error"),
            )

        return {"violations": violations, "updates": updates}
