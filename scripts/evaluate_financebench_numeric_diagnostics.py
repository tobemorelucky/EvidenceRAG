"""Add deterministic numeric-equivalence diagnostics to FinanceBench results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
NUMBER = re.compile(
    r"(?<![A-Za-z0-9])(?P<sign>[-+])?\$?\(?\s*(?P<value>\d[\d,]*(?:\.\d+)?)\s*\)?"
    r"\s*(?P<scale>billion|million|thousand|bn|mm|k)?\s*(?P<percent>%|percent)?",
    re.IGNORECASE,
)
SCALES = {
    "billion": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
    "million": Decimal("1000000"),
    "mm": Decimal("1000000"),
    "thousand": Decimal("1000"),
    "k": Decimal("1000"),
}
RESULT_MARKERS = re.compile(
    r"\b(?:answer|result|therefore|thus|equals?|approximately|about|ratio|change|difference|"
    r"increased?|decreased?|grew|declined|higher|lower)\b",
    re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _numbers(text: str, *, implicit_currency_scale: Decimal = Decimal("1")) -> list[tuple[Decimal, bool]]:
    cleaned = re.sub(r"\[source:[^\]]+\]", "", text or "", flags=re.IGNORECASE)
    values: list[tuple[Decimal, bool]] = []
    for match in NUMBER.finditer(cleaned):
        raw = match.group("value").replace(",", "")
        try:
            value = Decimal(raw)
        except InvalidOperation:
            continue
        token = match.group(0)
        if match.group("sign") == "-" or ("(" in token and ")" in token):
            value = -value
        scale = str(match.group("scale") or "").casefold()
        multiplier = SCALES.get(scale, Decimal("1"))
        if not scale and "$" in token:
            multiplier = implicit_currency_scale
        value *= multiplier
        is_percent = bool(match.group("percent"))
        if not scale and not is_percent and value == value.to_integral() and Decimal("1900") <= value <= Decimal("2100"):
            continue
        values.append((value, is_percent))
    return values


def _last_number(text: str) -> tuple[Decimal, bool] | None:
    values = _numbers(text)
    return values[-1] if values else None


def _candidate_result_numbers(
    text: str,
    *,
    implicit_currency_scale: Decimal = Decimal("1"),
) -> tuple[list[tuple[Decimal, bool]], str]:
    """Prefer explicit conclusion values without treating every operand as an answer."""
    cleaned = re.sub(r"\[source:[^\]]+\]", "", text or "", flags=re.IGNORECASE)
    bold_values: list[tuple[Decimal, bool]] = []
    for segment in re.findall(r"\*\*([^*]+)\*\*", cleaned):
        bold_values.extend(_numbers(segment, implicit_currency_scale=implicit_currency_scale))
    if bold_values:
        return bold_values, "bold_result"

    segments = [part.strip() for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", cleaned) if part.strip()]
    conclusion_values: list[tuple[Decimal, bool]] = []
    for segment in segments:
        if RESULT_MARKERS.search(segment):
            conclusion_values.extend(_numbers(segment, implicit_currency_scale=implicit_currency_scale))
    if conclusion_values:
        return conclusion_values, "conclusion_segment"

    if segments:
        final_values = _numbers(segments[-1], implicit_currency_scale=implicit_currency_scale)
        if final_values:
            return final_values, "final_segment"
    values = _numbers(cleaned, implicit_currency_scale=implicit_currency_scale)
    return (values[-1:] if values else []), "last_numeric"


def _equivalent_pair(
    candidate_value: tuple[Decimal, bool],
    reference_value: tuple[Decimal, bool],
) -> tuple[bool, Decimal, Decimal]:
    candidate_number, candidate_percent = candidate_value
    reference_number, reference_percent = reference_value
    normalized_candidates = {candidate_number}
    if candidate_percent != reference_percent:
        normalized_candidates.update({candidate_number * Decimal("100"), candidate_number / Decimal("100")})
    absolute_tolerance = Decimal("0.005") if max(abs(reference_number), Decimal("1")) < Decimal("100") else Decimal("0.5")
    relative_tolerance = abs(reference_number) * Decimal("0.005")
    tolerance = max(absolute_tolerance, relative_tolerance)
    delta = min(abs(value - reference_number) for value in normalized_candidates)
    return delta <= tolerance, delta, tolerance


def _requested_currency_scale(question: str) -> Decimal:
    lowered = (question or "").casefold()
    if re.search(r"\b(?:usd\s+)?billions?\b", lowered):
        return Decimal("1000000000")
    if re.search(r"\b(?:usd\s+)?millions?\b", lowered):
        return Decimal("1000000")
    if re.search(r"\b(?:usd\s+)?thousands?\b", lowered):
        return Decimal("1000")
    return Decimal("1")


def _numeric_equivalent(candidate: str, reference: str, question: str = "") -> tuple[bool | None, dict]:
    implicit_scale = _requested_currency_scale(question)
    candidate_values, extraction_method = _candidate_result_numbers(
        candidate,
        implicit_currency_scale=implicit_scale,
    )
    reference_values, reference_extraction_method = _candidate_result_numbers(
        reference,
        implicit_currency_scale=implicit_scale,
    )
    if not candidate_values or not reference_values:
        return None, {"reason": "no_comparable_final_numeric_value"}
    comparisons = [
        (_equivalent_pair(candidate_value, reference_value), candidate_value, reference_value)
        for candidate_value in candidate_values
        for reference_value in reference_values
    ]
    comparisons.sort(key=lambda item: item[0][1])
    (equivalent, delta, tolerance), matched_candidate, matched_reference = comparisons[0]
    return equivalent, {
        "candidate_result_numeric": str(matched_candidate[0]),
        "reference_result_numeric": str(matched_reference[0]),
        "absolute_delta": str(delta),
        "tolerance": str(tolerance),
        "candidate_extraction_method": extraction_method,
        "reference_extraction_method": reference_extraction_method,
        "candidate_result_count": len(candidate_values),
        "reference_result_count": len(reference_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    answers = _read_jsonl(args.answers)
    judges = _read_jsonl(args.judge)
    judges_by_run = {str(item.get("run_id") or ""): item for item in judges}
    with args.dataset.open("r", encoding="utf-8-sig", newline="") as handle:
        references = {str(row.get("financebench_id") or ""): str(row.get("answer") or "") for row in csv.DictReader(handle)}

    output = []
    for answer in answers:
        financebench_id = str(answer.get("financebench_id") or "")
        judge = judges_by_run.get(str(answer.get("langsmith_trace_id") or ""), {})
        task_type = str((answer.get("rag_trace") or {}).get("task_type") or "")
        question = str(answer.get("question") or "")
        equivalent, details = _numeric_equivalent(
            str(answer.get("answer") or ""),
            references.get(financebench_id, ""),
            question,
        )
        output.append({
            "financebench_id": financebench_id,
            "official_judge_correct": bool(int(judge.get("score") or 0)),
            "numeric_equivalent": equivalent,
            "numeric_diagnostic": details,
            "judgment_diagnostic_label": "manual_review_required" if task_type == "judgment" else None,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in output), encoding="utf-8")
    comparable = [item for item in output if item["numeric_equivalent"] is not None]
    print(json.dumps({
        "records": len(output),
        "numeric_comparable": len(comparable),
        "numeric_equivalent": sum(bool(item["numeric_equivalent"]) for item in comparable),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
