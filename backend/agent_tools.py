"""Bounded, deterministic tools used by EvidenceRAG's agentic route."""

from __future__ import annotations

import ast
import re
from decimal import Decimal, InvalidOperation
from typing import Any


_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?")
_MAX_EXPRESSION_LENGTH = 120


def find_evidence(question: str, documents: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Return keyword-focused evidence candidates without a second model call."""
    query_terms = {term.lower() for term in _TOKEN_PATTERN.findall(question or "") if len(term) > 2}
    scored: list[dict[str, Any]] = []
    for document in documents:
        text = str(document.get("text") or document.get("page_text") or "")
        terms = {term.lower() for term in _TOKEN_PATTERN.findall(text)}
        overlap = len(query_terms & terms) / max(1, len(query_terms))
        score = float(document.get("rerank_score", document.get("score", 0.0)) or 0.0) + overlap
        scored.append({**document, "find_score": score})
    scored.sort(key=lambda item: item["find_score"], reverse=True)
    return scored[: max(1, limit)]


def select_pages(documents: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Select unique cited pages that an agent may open; never expands document scope."""
    pages: list[dict[str, Any]] = []
    seen: set[tuple[str, Any]] = set()
    for document in documents:
        key = (str(document.get("filename") or ""), document.get("page_number"))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        pages.append({"filename": key[0], "page_number": key[1]})
        if len(pages) >= max(1, limit):
            break
    return pages


def open_pages(pages: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Read only pages selected from already retrieved evidence."""
    if not pages:
        return []
    from document_page_store import DocumentPageStore

    selected = pages[: max(1, limit)]
    keys = {(item["filename"], item.get("page_number")) for item in selected}
    records = DocumentPageStore().get_pages_by_filenames(
        [item["filename"] for item in selected],
        warm_cache=False,
    )
    return [
        {
            "filename": record.get("filename"),
            "page_number": record.get("page_number"),
            "text": record.get("page_text", ""),
            "page_text": record.get("page_text", ""),
            "table_text": record.get("table_text", ""),
            "source": "agent_open_page",
        }
        for record in records
        if (record.get("filename"), record.get("page_number")) in keys
    ]


def calculate(expression: str) -> dict[str, str]:
    """Evaluate Decimal-only arithmetic and return a compact, auditable result."""
    source = (expression or "").strip()
    if not source or len(source) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("invalid calculation expression")
    if not re.fullmatch(r"[\d.\s()+\-*/]+", source):
        raise ValueError("calculation supports only numeric arithmetic")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid calculation expression") from exc

    def evaluate(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ValueError("division by zero")
            return left / right
        raise ValueError("calculation supports only numeric arithmetic")

    try:
        result = evaluate(tree)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    operands = _NUMBER_PATTERN.findall(source)
    return {"expression": source, "operands": ", ".join(operands), "result": format(result, "f")}


def calculate_decimal(expression: str) -> str:
    """Compatibility wrapper for callers that only require the numeric result."""
    try:
        return calculate(expression)["result"]
    except ValueError as exc:
        raise ValueError("invalid calculation expression") from exc
