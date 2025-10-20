"""Schema-driven validation utilities used by the API endpoints."""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .schema_loader import SchemaLoader


@dataclass
class PageValidationResult:
    """Result container returned to the frontend."""

    blocking_missing_required: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    advisories: List[str] = field(default_factory=list)

    def dict(self) -> Dict[str, Any]:
        return {
            "blockingMissingRequired": list(self.blocking_missing_required),
            "errors": list(self.errors),
            "advisories": list(self.advisories),
        }


class ExpressionEvaluator(ast.NodeVisitor):
    """Tiny expression evaluator supporting schema rule syntax."""

    BOOL_OPERATORS = {ast.And: operator.and_, ast.Or: operator.or_}
    COMPARE_OPERATORS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.In: lambda left, right: left in right if right is not None else False,
        ast.NotIn: lambda left, right: left not in right if right is not None else True,
    }
    UNARY_OPERATORS = {
        ast.Not: operator.not_,
        ast.USub: operator.neg,
        ast.UAdd: lambda value: value,
    }

    FUNCTIONS = {
        "Number": lambda value: _to_number(value),
        "Boolean": lambda value: bool(_normalise_value(value)),
        "String": lambda value: "" if value is None else str(value),
        "__includes__": lambda collection, value: value in (collection or []),
        "__trim__": lambda value: "" if value is None else str(value).strip(),
        "__length__": lambda value: len(value or ""),
    }

    def __init__(self, context: Dict[str, Any]):
        self.context = context

    # Python 3.11 tightened the visitor typing; keep override quiet.
    def visit(self, node: ast.AST):  # type: ignore[override]
        return super().visit(node)

    def visit_Module(self, node: ast.Module):  # pragma: no cover - defensive
        return self.visit(node.body[0].value)  # type: ignore[index]

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_BoolOp(self, node: ast.BoolOp):
        values = [self.visit(value) for value in node.values]
        result = values[0]
        for value in values[1:]:
            operator_fn = self.BOOL_OPERATORS[type(node.op)]
            result = operator_fn(bool(result), bool(value))
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operator_fn = self.UNARY_OPERATORS[type(node.op)]
        return operator_fn(self.visit(node.operand))

    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        for operator_node, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            operation = self.COMPARE_OPERATORS[type(operator_node)]
            if not operation(left, right):
                return False
            left = right
        return True

    def visit_Name(self, node: ast.Name):
        if node.id in self.FUNCTIONS:
            return self.FUNCTIONS[node.id]
        return self.context.get(node.id)

    def visit_Call(self, node: ast.Call):
        func = self.visit(node.func)
        if func not in self.FUNCTIONS.values():
            raise ValueError(f"Function {ast.dump(node.func)} not permitted")
        args = [self.visit(arg) for arg in node.args]
        return func(*args)

    def visit_List(self, node: ast.List):
        return [self.visit(value) for value in node.elts]

    def visit_Tuple(self, node: ast.Tuple):  # pragma: no cover - defensive
        return tuple(self.visit(value) for value in node.elts)

    def visit_Constant(self, node: ast.Constant):
        return node.value

    def generic_visit(self, node):  # pragma: no cover - defensive
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")


class Validator:
    """Perform schema-based validation for a page."""

    def __init__(self, schema_loader: SchemaLoader) -> None:
        self.schema_loader = schema_loader

    # Public API ---------------------------------------------------------
    def validate_page(
        self, page_id: str, fields: List[Dict[str, Any]]
    ) -> PageValidationResult:
        """Flatten the field list then run schema validation."""

        data = {
            field.get("id"): field.get("value")
            for field in fields
            if isinstance(field, dict) and field.get("id")
        }
        return self.validate(page_id, data)

    def validate(self, page_id: str, data: Dict[str, Any]) -> PageValidationResult:
        schema = self.schema_loader.load_schema(page_id)
        result = PageValidationResult()

        field_schemas = schema.get("fields", [])
        context = self._build_context(field_schemas, data)

        for field_schema in field_schemas:
            field_id = field_schema.get("id")
            if not field_id:
                continue

            visible = self._is_visible(field_schema.get("visibleWhen"), context)
            required = bool(field_schema.get("required"))
            value = data.get(field_id)

            if required and visible and self._is_empty(value):
                result.blocking_missing_required.append(field_id)
            elif visible:
                self._apply_field_constraints(field_schema, value, result)

        self._apply_rules(schema, context, result)
        return result

    # Helpers ------------------------------------------------------------
    def _build_context(
        self, field_schemas: Iterable[Dict[str, Any]], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        for field_schema in field_schemas:
            field_id = field_schema.get("id")
            if not field_id:
                continue
            context[field_id] = self._normalise_value(data.get(field_id))
        for key, value in data.items():
            context.setdefault(key, self._normalise_value(value))
        return context

    def _is_visible(self, expression: Optional[str], context: Dict[str, Any]) -> bool:
        if not expression:
            return True
        return self._safe_evaluate(expression, context, default=True)

    def _apply_field_constraints(
        self, field_schema: Dict[str, Any], value: Any, result: PageValidationResult
    ) -> None:
        if self._is_empty(value):
            return

        label = field_schema.get("label") or field_schema.get("id") or "field"

        pattern = field_schema.get("pattern")
        if pattern and isinstance(value, str):
            if not re.match(pattern, value.strip()):
                result.errors.append(f"{label} does not match the required format.")

        minimum = field_schema.get("min")
        if minimum is not None:
            numeric = _to_number(value)
            if numeric < float(minimum):
                result.errors.append(f"{label} must be at least {minimum}.")

        maximum = field_schema.get("max")
        if maximum is not None:
            numeric = _to_number(value)
            if numeric > float(maximum):
                result.errors.append(f"{label} must be at most {maximum}.")

    def _apply_rules(
        self, schema: Dict[str, Any], context: Dict[str, Any], result: PageValidationResult
    ) -> None:
        rules = schema.get("validation", {}).get("rules", [])
        for rule in rules:
            assert_expr = rule.get("assert")
            message = rule.get("message") or rule.get("id")
            if not assert_expr or not message:
                continue

            when_expr = rule.get("when")
            if when_expr and not self._safe_evaluate(when_expr, context, default=True):
                continue

            passed = self._safe_evaluate(assert_expr, context, default=True)
            if passed:
                continue

            severity = rule.get("severity", "error")
            if severity == "advisory":
                result.advisories.append(message)
            else:
                result.errors.append(message)

    def _safe_evaluate(
        self, expression: str, context: Dict[str, Any], *, default: bool
    ) -> bool:
        try:
            tree = ast.parse(self._prepare_expression(expression), mode="eval")
            evaluator = ExpressionEvaluator(context)
            value = evaluator.visit(tree)
            return bool(value)
        except Exception:
            return default

    def _prepare_expression(self, expression: str) -> str:
        expr = expression
        replacements = {
            "&&": " and ",
            "||": " or ",
            "===": "==",
            "!==": "!=",
        }
        for original, replacement in replacements.items():
            expr = expr.replace(original, replacement)
        expr = re.sub(r"\btrue\b", "True", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bfalse\b", "False", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bnull\b", "None", expr, flags=re.IGNORECASE)
        expr = re.sub(r"!(?!=)", " not ", expr)
        expr = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\.trim\(\)", r"__trim__(\\1)", expr)
        expr = re.sub(
            r"(__trim__\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*)\.length",
            r"__length__(\\1)",
            expr,
        )
        expr = re.sub(
            r"(?P<target>(?:\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*))\.includes\(",
            r"__includes__(\g<target>, ",
            expr,
        )
        return expr

    def _normalise_value(self, value: Any) -> Any:
        return _normalise_value(value)

    def _is_empty(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, (list, tuple, set)):
            return len(value) == 0
        if isinstance(value, dict):
            return all(self._is_empty(v) for v in value.values())
        return False


def _normalise_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered in {"yes", "true"}:
            return True
        if lowered in {"no", "false"}:
            return False
        return stripped
    return value


def _to_number(value: Any) -> float:
    value = _normalise_value(value)
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

