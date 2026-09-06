"""Offline attribution of final100 answer failures; no external calls."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL100 = ROOT / "reports/final100"
PRO_SHADOW = ROOT / "reports/answer_model_shadow_v1"
DATASET = ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv"
OUTPUT = ROOT / "reports/answer_failure_attribution_v2"

CATEGORY_NAMES = {
    "A": "Evidence insufficient",
    "B": "Metric definition error",
    "C": "Entity scope error",
    "D": "Period selection error",
    "E": "Calculation execution error",
    "F": "Reasoning direction error",
    "G": "Refusal / unnecessary uncertainty",
    "H": "Other",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _has(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def classify_failure(question: str, answer: str, judge_reason: str) -> tuple[str, str]:
    """Classify from question/answer/Judge text only; gold/reference is excluded."""
    q, a, reason = question.lower(), answer.lower(), judge_reason.lower()
    combined = f"{q}\n{a}\n{reason}"
    answer_refuses = _has(
        r"cannot (?:answer|determine|calculate|identify)|insufficient (?:data|evidence|information)|not enough (?:data|evidence|information)",
        a,
    )
    judge_confirms_refusal = _has(
        r"fails? to (?:answer|provide)|does not answer|claims? insufficient|cannot be answered|does not provide the (?:requested|required)",
        reason,
    )
    refusal = answer_refuses and judge_confirms_refusal
    if refusal:
        return "G", "Answer refuses, claims insufficiency, or omits the explicitly requested result."
    if _has(r"wrong (?:company|business|segment|division|entity)|excludes? the .*segment|different (?:company|segment)|scope", reason):
        return "C", "Judge identifies an entity, segment, or scope mismatch."
    if _has(r"wrong (?:year|quarter|period)|fy\s*\d+|q[1-4]|later|earlier|instead of the .*compan", reason) and _has(r"year|quarter|period|later|earlier|acquir|spin", combined):
        return "D", "The selected reporting period or time-specific event is inconsistent."
    calculation_question = _has(r"\bcalculate\b|\bratio\b|\bmargin\b|\bgrowth\b|\baverage\b|\bpercent\b|\bturnover\b|\bdpo\b|round", q)
    calculation_reason = _has(r"comput|calculat|formula|round|does not match|different .*source|final .*\d|ratio of|arithmetic", reason)
    if calculation_question and calculation_reason:
        return "E", "Required calculation is attempted, but its operands, formula, arithmetic, or rounding is wrong."
    if _has(r"increase|decrease|improv|declin|higher|lower|accelerat|decelerat|largest|lowest|most|least|contradict", combined):
        return "F", "The answer selects the wrong direction, trend, or ordered alternative."
    if _has(r"gross margin|operating margin|ebitda|\bebit\b|quick ratio|working capital|adjusted eps|metric|different ratio|not (?:useful|relevant)", combined):
        return "B", "The answer applies or interprets the requested financial metric inconsistently."
    if _has(r"evidence (?:does not|lacks|missing)|context (?:does not|lacks|missing)|necessary facts? (?:are|is) missing", reason):
        return "A", "The existing Judge reason explicitly identifies missing evidence facts."
    return "H", "Failure is a factual selection, unsupported-detail, or multi-part omission not covered above."


def remediation(category: str, exact_span_hit: bool) -> str:
    # Gold-derived span hit is used only for this offline diagnostic, never classification.
    if not exact_span_hit:
        return "retrieval_or_evidence_flow"
    if category in {"B", "C", "D", "E", "F"}:
        return "structured_financial_reasoning"
    return "answer_prompt"


def excerpt(text: str, limit: int = 600) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def distribution(ids: list[str], cases: dict[str, dict], pro_records: dict[str, dict], rows: dict[str, dict]) -> dict:
    counter: Counter = Counter()
    outside = 0
    for question_id in ids:
        if question_id in cases:
            counter[cases[question_id]["category"]] += 1
            continue
        pro = pro_records.get(question_id, {})
        judge = pro.get("judge", {})
        baseline_answer = pro.get("answer", "")
        row = rows.get(question_id, {})
        if judge and row:
            category, _ = classify_failure(row.get("question", ""), baseline_answer, judge.get("reason", ""))
            counter[category] += 1
        else:
            outside += 1
    return {"categories": dict(sorted(counter.items())), "unavailable": outside}


def main() -> None:
    # Reading every requested artifact is intentional: count and ID contracts are audited below.
    final_summary = json.loads((FINAL100 / "final_summary.json").read_text(encoding="utf-8"))
    answers = {record["question_id"]: record for record in load_jsonl(FINAL100 / "answers.jsonl")}
    judges = {record["question_id"]: record for record in load_jsonl(FINAL100 / "judge_results.jsonl")}
    jina = json.loads((FINAL100 / "jina_results.json").read_text(encoding="utf-8"))["records"]
    retrieval = json.loads((FINAL100 / "retrieval_snapshot.json").read_text(encoding="utf-8"))["records"]
    with DATASET.open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["financebench_id"]: row for row in csv.DictReader(stream)}
    if not all(len(value) == 100 for value in (answers, judges, rows)) or len(jina) != 100 or len(retrieval) != 100:
        raise ValueError("Expected 100 complete records in every final100 input")
    if final_summary["strict_judge"]["correct"] != 68:
        raise ValueError("Unexpected final100 baseline")

    cases = []
    for question_id, answer_record in answers.items():
        judge_record = judges[question_id]
        context = answer_record.get("context_metrics", {})
        if judge_record["judge"]["score"] != 0 or not context.get("candidate_gold_page_hit"):
            continue
        row = rows[question_id]
        category, notes = classify_failure(row["question"], answer_record["answer"], judge_record["judge"]["reason"])
        exact_span_hit = bool(context.get("context_evidence_span_hit"))
        cases.append({
            "id": question_id,
            "question": row["question"],
            "judge_reason": judge_record["judge"]["reason"],
            "category": category,
            "category_name": CATEGORY_NAMES[category],
            "evidence_available": exact_span_hit,
            "answer_excerpt": excerpt(answer_record["answer"]),
            "reference_answer": row["answer"],
            "notes": notes,
            "offline_remediation": remediation(category, exact_span_hit),
        })
    if len(cases) != 22:
        raise ValueError(f"Expected 22 context-hit answer failures, found {len(cases)}")

    case_map = {case["id"]: case for case in cases}
    category_counts = Counter(case["category"] for case in cases)
    remediation_counts = Counter(case["offline_remediation"] for case in cases)
    pro_summary_path = PRO_SHADOW / "summary.json"
    pro_records = {record["question_id"]: record for record in load_jsonl(PRO_SHADOW / "answers.jsonl")} if (PRO_SHADOW / "answers.jsonl").exists() else {}
    pro_summary = json.loads(pro_summary_path.read_text(encoding="utf-8")) if pro_summary_path.exists() else {"question_ids": {}}
    newly_correct = pro_summary.get("question_ids", {}).get("newly_correct", [])
    regressed = pro_summary.get("question_ids", {}).get("regressed", [])
    migrations = {
        "flash_incorrect_pro_correct": {"ids": newly_correct, **distribution(newly_correct, case_map, pro_records, rows)},
        "flash_correct_pro_incorrect": {"ids": regressed, **distribution(regressed, case_map, pro_records, rows)},
    }
    summary = {
        "cohort": {"strict_incorrect_and_final_context_page_hit": len(cases), "final100_strict_correct": 68},
        "category_distribution": {
            key: {"name": CATEGORY_NAMES[key], "count": category_counts.get(key, 0), "share": category_counts.get(key, 0) / len(cases)}
            for key in CATEGORY_NAMES
        },
        "model_migration_distribution": migrations,
        "potential_remediation": {
            "retrieval_or_evidence_flow": remediation_counts.get("retrieval_or_evidence_flow", 0),
            "answer_prompt": remediation_counts.get("answer_prompt", 0),
            "structured_financial_reasoning": remediation_counts.get("structured_financial_reasoning", 0),
            "method": "Exact-span availability is an offline gold diagnostic only; primary category never uses gold/reference.",
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "failure_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Answer Failure Attribution v2", "", "## Scope and method", "",
        "- Cohort: Strict Judge=false and final context gold-page hit=true.",
        "- Classification uses only question, frozen answer, and existing Judge reason.",
        "- Reference answer and gold-derived exact-span hit are output/offline diagnostics only; they do not select the category.",
        "- No LLM, Judge, Jina, retrieval, or production pipeline call is made.", "",
        "## Distribution", "", "| Category | Meaning | Count | Share |", "|---|---|---:|---:|"]
    for key, name in CATEGORY_NAMES.items():
        count = category_counts.get(key, 0)
        lines.append(f"| {key} | {name} | {count} | {count / len(cases):.2%} |")
    lines += ["", "## Potential remediation", "",
        f"- Retrieval/evidence flow: **{remediation_counts.get('retrieval_or_evidence_flow', 0)}**",
        f"- Answer prompt/response contract: **{remediation_counts.get('answer_prompt', 0)}**",
        f"- Structured financial reasoning: **{remediation_counts.get('structured_financial_reasoning', 0)}**", "",
        "These are attribution estimates, not an accuracy forecast; categories are mutually exclusive.", "", "## Flash/Pro migrations", ""]
    for title, data in migrations.items():
        lines += [f"### {title}", "", f"- IDs: {', '.join(data['ids']) or 'None'}", f"- Category distribution: `{json.dumps(data['categories'], ensure_ascii=False)}`", ""]
    lines += ["## Failure cases", ""]
    for case in cases:
        lines += [f"### {case['id']} — {case['category']}. {case['category_name']}", "",
            f"- Question: {case['question']}", f"- Evidence exact span available: {case['evidence_available']}",
            f"- Judge reason: {case['judge_reason']}", f"- Notes: {case['notes']}",
            f"- Offline remediation: `{case['offline_remediation']}`", f"- Answer excerpt: {case['answer_excerpt']}",
            f"- Reference answer: {case['reference_answer']}", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
