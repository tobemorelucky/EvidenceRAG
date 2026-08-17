"""Bounded, auditable tools used by the EvidenceRAG deep retrieval path."""

import ast
import operator
from decimal import Decimal, InvalidOperation

from document_page_store import DocumentPageStore


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculate_decimal(expression: str) -> str:
    """Evaluate a small arithmetic expression using Decimal only."""

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("division by zero")
            return _BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError("unsupported calculation")

    try:
        tree = ast.parse((expression or "").strip(), mode="eval")
        result = evaluate(tree)
    except (SyntaxError, InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"invalid calculation: {exc}") from exc
    return format(result.normalize(), "f")


def find_in_evidence(documents: list[dict], patterns: list[str], limit: int = 5) -> list[dict]:
    normalized = [pattern.strip().lower() for pattern in patterns if pattern and pattern.strip()]
    if not normalized:
        return []
    matches = []
    for document in documents:
        text = str(document.get("text") or "")
        lowered = text.lower()
        if any(pattern in lowered for pattern in normalized):
            matches.append(document)
            if len(matches) >= limit:
                break
    return matches


def open_page(filename: str, page_number: int, adjacent_window: int = 0) -> list[dict]:
    if not filename:
        return []
    pages = DocumentPageStore().get_pages_by_filenames([filename])
    lower = max(0, int(page_number) - max(0, adjacent_window))
    upper = int(page_number) + max(0, adjacent_window)
    return [page for page in pages if lower <= int(page.get("page_number", 0) or 0) <= upper]

