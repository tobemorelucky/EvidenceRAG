"""Restricted Decimal expression evaluation for verified skill operands."""

from __future__ import annotations

import ast
from decimal import Decimal, DecimalException, ROUND_HALF_UP
from typing import Mapping


class DecimalCalculationError(ValueError):
    pass


def _evaluate(node: ast.AST, values: Mapping[str, Decimal]) -> Decimal:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, values)
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand, values)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _evaluate(node.left, values)
        right = _evaluate(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise DecimalCalculationError("division_by_zero")
        return left / right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "average":
        if len(node.args) < 2 or node.keywords:
            raise DecimalCalculationError("average_requires_at_least_two_operands")
        items = [_evaluate(item, values) for item in node.args]
        return sum(items, Decimal("0")) / Decimal(len(items))
    raise DecimalCalculationError("unsupported_expression_node")


def calculate_decimal(
    expression: str,
    values: Mapping[str, Decimal],
    rounding_decimal_places: int | None = None,
) -> dict[str, str | int | None]:
    if not expression or len(expression) > 500:
        raise DecimalCalculationError("invalid_expression")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree, values)
    except (SyntaxError, DecimalException) as exc:
        raise DecimalCalculationError("invalid_expression") from exc
    unknown_names = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id != "average" and node.id not in values
    }
    if unknown_names:
        raise DecimalCalculationError(f"unknown_operands:{','.join(sorted(unknown_names))}")
    display = format(result, "f")
    places = None
    if rounding_decimal_places is not None:
        places = max(0, min(12, int(rounding_decimal_places)))
        quantum = Decimal(1).scaleb(-places)
        rounded = result.quantize(quantum, rounding=ROUND_HALF_UP)
        display = f"{rounded:.{places}f}"
    return {
        "expression": expression,
        "full_precision_result": format(result, "f"),
        "display_result": display,
        "rounding_decimal_places": places,
    }
