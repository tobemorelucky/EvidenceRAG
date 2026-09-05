"""Evaluate Answer Verification v1 on cached Jina answers without APIs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from answer_verifier_v1 import verify_answer_v1  # noqa: E402


DEFAULT_ANSWERS = ROOT / "reports" / "jina_full_baseline_input120_all100" / "answers.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "answer_failure_audit_v1.json"
DEFAULT_OUTPUT = ROOT / "reports" / "answer_verifier_shadow_v1.json"
HIGH_RISK_THRESHOLD = 0.25


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _theoretical_fix(result: dict) -> tuple[bool, str]:
    details = result["details"]
    if details.get("invalid_refusal"):
        return True, "refusal_can_be_reopened_because_relevant_operands_exist"
    for formula in details.get("explicit_formulas") or []:
        if formula.get("operands_supported") and not formula.get("consistent"):
            return True, "explicit_formula_can_be_recomputed_deterministically"
    return False, "diagnostic_only_no_deterministic_rewrite"


def _evaluate_record(source: dict, *, label: str, category: str) -> dict:
    verification = verify_answer_v1(source["question"], source["answer"], source["evidence"]).to_dict()
    fixable, fix_reason = _theoretical_fix(verification)
    return {
        "question_id": source["financebench_id"],
        "label": label,
        "category": category,
        "question": source["question"],
        "answer": source["answer"],
        "stored_judge": source.get("judge"),
        "evidence_chars": len(source.get("evidence") or ""),
        "verification": verification,
        "warning_detected": bool(verification["warnings"]),
        "high_risk_flag": verification["risk_score"] >= HIGH_RISK_THRESHOLD,
        "theoretically_fixable": fixable,
        "theoretical_fix_reason": fix_reason,
    }


def _summary(failures: list[dict], controls: list[dict]) -> dict:
    failure_flagged = sum(item["high_risk_flag"] for item in failures)
    control_flagged = sum(item["high_risk_flag"] for item in controls)
    categories = {}
    for category in sorted({item["category"] for item in failures}):
        items = [item for item in failures if item["category"] == category]
        categories[category] = {
            "questions": len(items),
            "warning_detected": sum(item["warning_detected"] for item in items),
            "high_risk_flagged": sum(item["high_risk_flag"] for item in items),
            "theoretically_fixable": sum(item["theoretically_fixable"] for item in items),
        }
    return {
        "failure_questions": len(failures),
        "correct_control_questions": len(controls),
        "risk_threshold": HIGH_RISK_THRESHOLD,
        "failures_with_any_warning": sum(item["warning_detected"] for item in failures),
        "failures_high_risk_flagged": failure_flagged,
        "failure_detection_rate": round(failure_flagged / max(1, len(failures)), 4),
        "theoretically_fixable_failures": sum(item["theoretically_fixable"] for item in failures),
        "theoretically_fixable_ids": [item["question_id"] for item in failures if item["theoretically_fixable"]],
        "controls_with_any_warning": sum(item["warning_detected"] for item in controls),
        "control_false_positives": control_flagged,
        "control_false_positive_rate": round(control_flagged / max(1, len(controls)), 4),
        "high_risk_precision_on_balanced_shadow_set": round(failure_flagged / max(1, failure_flagged + control_flagged), 4),
        "high_risk_specificity_on_controls": round((len(controls) - control_flagged) / max(1, len(controls)), 4),
        "confusion_matrix_on_15_failure_plus_15_control": {
            "true_positive": failure_flagged,
            "false_negative": len(failures) - failure_flagged,
            "false_positive": control_flagged,
            "true_negative": len(controls) - control_flagged,
        },
        "warning_counts_failures": dict(sorted(Counter(
            warning for item in failures for warning in item["verification"]["warnings"]
        ).items())),
        "warning_counts_controls": dict(sorted(Counter(
            warning for item in controls for warning in item["verification"]["warnings"]
        ).items())),
        "categories": categories,
    }


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Answer Verification Shadow v1", "",
        "- Source: existing Jina full baseline answers and stored evidence",
        "- External calls: LLM=0, Judge=0, Jina=0, LangSmith=0",
        "- The fixed 15 Judge-false/context-hit audit questions are the target set.",
        "- Fifteen stored Judge-correct answers are a deterministic read-only control for false-positive estimation.",
        "- Verification only diagnoses; it never rewrites an answer.", "",
        "## 汇总", "",
        f"- Failure questions with any warning: {summary['failures_with_any_warning']}/15",
        f"- High-risk failures detected (threshold {summary['risk_threshold']}): {summary['failures_high_risk_flagged']}/15 ({_pct(summary['failure_detection_rate'])})",
        f"- Theoretically deterministic-fixable: {summary['theoretically_fixable_failures']}/15",
        f"- Theoretically fixable IDs: `{summary['theoretically_fixable_ids']}`",
        f"- Correct controls with any warning: {summary['controls_with_any_warning']}/15",
        f"- Correct-control false positives: {summary['control_false_positives']}/15 ({_pct(summary['control_false_positive_rate'])})",
        f"- High-risk precision/specificity on this balanced shadow set: {_pct(summary['high_risk_precision_on_balanced_shadow_set'])} / {_pct(summary['high_risk_specificity_on_controls'])}",
        f"- Confusion matrix: `{summary['confusion_matrix_on_15_failure_plus_15_control']}`",
        f"- Failure warnings: `{summary['warning_counts_failures']}`",
        f"- Control warnings: `{summary['warning_counts_controls']}`", "",
        "## Failure category", "", "| Category | N | Any warning | High risk | Theoretically fixable |", "|---|---:|---:|---:|---:|",
    ]
    for category, item in summary["categories"].items():
        lines.append(
            f"| {category} | {item['questions']} | {item['warning_detected']} | {item['high_risk_flagged']} | {item['theoretically_fixable']} |"
        )
    lines += ["", "## 固定15题", ""]
    for index, item in enumerate(payload["failure_records"], 1):
        result = item["verification"]
        lines += [
            f"### {index}. {item['question_id']} — `{item['category']}`", "",
            f"- Question: {item['question']}",
            f"- Flags: numeric={result['numeric_ok']}, metric={result['metric_ok']}, period={result['period_ok']}, formula={result['formula_ok']}",
            f"- Warnings/risk: `{result['warnings']}` / {result['risk_score']}",
            f"- Detected/high-risk/fixable: {item['warning_detected']} / {item['high_risk_flag']} / {item['theoretically_fixable']}",
            f"- Fix assessment: `{item['theoretical_fix_reason']}`", "",
        ]
    lines += ["## Correct controls and false positives", "", "| ID | Risk | Flagged | Warnings |", "|---|---:|---|---|"]
    for item in payload["control_records"]:
        lines.append(
            f"| {item['question_id']} | {item['verification']['risk_score']} | {item['high_risk_flag']} | `{item['verification']['warnings']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--failure-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    answers = _load_jsonl(args.answers)
    by_id = {record["financebench_id"]: record for record in answers}
    if len(answers) != 100 or len(by_id) != 100:
        raise ValueError("Expected the existing complete 100-question Jina baseline")
    audit = json.loads(args.failure_audit.read_text(encoding="utf-8"))
    audit_items = audit.get("items") or []
    if len(audit_items) != 15:
        raise ValueError("Expected exactly 15 answer_failure_audit_v1 items")

    failures = []
    for audit_item in audit_items:
        source = by_id[audit_item["question_id"]]
        if int((source.get("judge") or {}).get("score", 0)) != 0:
            raise ValueError(f"Failure audit/Judge drift for {audit_item['question_id']}")
        if source["answer"] != audit_item["model_answer"]:
            raise ValueError(f"Failure audit/answer drift for {audit_item['question_id']}")
        failures.append(_evaluate_record(source, label="stored_judge_incorrect", category=audit_item["category"]))

    failure_ids = {item["question_id"] for item in failures}
    correct_candidates = sorted(
        (record for record in answers if record["financebench_id"] not in failure_ids and int((record.get("judge") or {}).get("score", 0)) == 1),
        key=lambda record: record["financebench_id"],
    )
    controls = [_evaluate_record(record, label="stored_judge_correct_control", category="correct_control") for record in correct_candidates[:15]]
    if len(controls) != 15:
        raise ValueError("Expected 15 stored Judge-correct controls")

    payload = {
        "schema": "answer_verifier_shadow_v1",
        "inputs": {"answers": str(args.answers), "failure_audit": str(args.failure_audit)},
        "selection": {
            "failure_set": "answer_failure_audit_v1 exact 15",
            "control_set": "lexicographically first 15 stored Judge-correct IDs excluding failure set",
        },
        "external_calls": {"llm": 0, "judge": 0, "jina": 0, "langsmith": 0},
        "summary": _summary(failures, controls),
        "failure_records": failures,
        "control_records": controls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
