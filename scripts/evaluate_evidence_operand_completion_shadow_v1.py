"""Search frozen RRF Top120 for operands missing from frozen Jina contexts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from evidence_operand_completion_shadow_v1 import search_missing_operands_v1  # noqa: E402


DEFAULT_ANSWERS = ROOT / "reports/jina_full_baseline_input120_all100/answers.jsonl"
DEFAULT_AUDIT = ROOT / "reports/answer_failure_audit_v1.json"
DEFAULT_OPERATIONS = ROOT / "reports/financial_operation_schema_shadow_v1.json"
DEFAULT_RECALL = ROOT / "reports/jina_full_baseline_input120_all100/recall.json"
DEFAULT_JSON = ROOT / "reports/evidence_operand_completion_shadow_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/evidence_operand_completion_shadow_v1.md"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(records: list[dict]) -> dict:
    eligible = [record for record in records if record["status"] == "searched_missing_operands"]
    return {
        "questions": len(records), "schema_recognized": sum(record["schema_recognized"] for record in records),
        "eligible_missing_operand_questions": len(eligible),
        "rrf_candidate_has_any_missing_operand": sum(record["after_candidate_has_operand"] for record in eligible),
        "potentially_recoverable": sum(record["potentially_recoverable"] for record in eligible),
        "statuses": dict(Counter(record["status"] for record in records)),
        "missing_operand_counts": dict(Counter(operand for record in eligible for operand in record["missing_operands"])),
        "external_calls": {"llm": 0, "jina": 0, "judge": 0, "langsmith": 0, "retrieval": 0},
    }


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evidence Operand Completion Shadow v1", "",
        "本报告复用冻结 Jina context、Financial Operation Schema v1结果和同一基线的冻结 RRF Top120。没有重新检索，也未调用 Jina、LLM、Judge 或 LangSmith。", "",
        "## 汇总", "",
        f"- 题目：{summary['questions']}",
        f"- Schema成功：{summary['schema_recognized']}",
        f"- 需要搜索missing operand：{summary['eligible_missing_operand_questions']}",
        f"- Top120含任一missing operand：{summary['rrf_candidate_has_any_missing_operand']}",
        f"- 所有missing operands均可从context外恢复：{summary['potentially_recoverable']}",
        f"- Missing operands：`{json.dumps(summary['missing_operand_counts'], ensure_ascii=False)}`",
        f"- Status：`{json.dumps(summary['statuses'], ensure_ascii=False)}`", "",
        "`potentially_recoverable` 只在每个缺失operand都存在满足同entity、同period、operand行语义约束且尚未进入context的RRF候选时为true。", "",
        "## 逐题", "",
    ]
    for index, record in enumerate(payload["records"], 1):
        lines.extend([
            f"### {index}. {record['question_id']}", "", f"**问题：** {record['question']}", "",
            f"- Status：`{record['status']}`",
            f"- Missing operands：`{record['missing_operands']}`",
            f"- Before context has operand：`{record['before_context_has_operand']}`",
            f"- After candidate has operand：`{record['after_candidate_has_operand']}`",
            f"- Potentially recoverable：`{record['potentially_recoverable']}`", "",
        ])
        for operand, candidates in record["found_candidates"].items():
            lines.append(f"**{operand} candidates：**")
            lines.append("")
            if not candidates:
                lines.append("- 无")
            for candidate in candidates:
                first = candidate["matched_lines"][0]
                lines.append(
                    f"- `{candidate['document']} p.{candidate['page']}` RRF={candidate['rrf_rank']} "
                    f"in_context={candidate['already_in_context']} — {first['text']}"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--operations", type=Path, default=DEFAULT_OPERATIONS)
    parser.add_argument("--recall", type=Path, default=DEFAULT_RECALL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    answers = {row["financebench_id"]: row for row in load_jsonl(args.answers)}
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    operations = json.loads(args.operations.read_text(encoding="utf-8"))
    recall = json.loads(args.recall.read_text(encoding="utf-8"))
    if len(audit.get("items", [])) != 15 or recall.get("schema") != "jina_full_baseline_recall_v1":
        raise ValueError("Expected the exact 15-item audit and frozen Jina baseline recall")
    operation_by_id = {record["question_id"]: record for record in operations["records"]}
    recall_by_id = {record["question_id"]: record for record in recall["records"]}
    records = []
    for index, item in enumerate(audit["items"], 1):
        question_id = item["question_id"]
        source, operation, frozen = answers[question_id], operation_by_id[question_id], recall_by_id[question_id]
        if frozen["question"] != source["question"] or len(frozen["chunks"]) != 120:
            raise ValueError(f"Frozen RRF snapshot drift for {question_id}")
        recognized = bool(operation["schema"]["recognized"])
        missing = list(operation["operand_evaluation"]["missing_operands"])
        if not recognized:
            status, search = "schema_unrecognized", {"found_candidates": {}, "candidate_has_operand": {}, "outside_context_candidate_has_operand": {}, "all_missing_operands_recoverable": False}
        elif not missing:
            status, search = "operands_already_complete", {"found_candidates": {}, "candidate_has_operand": {}, "outside_context_candidate_has_operand": {}, "all_missing_operands_recoverable": False}
        else:
            status = "searched_missing_operands"
            search = search_missing_operands_v1(operation["schema"], missing, frozen["chunks"], source.get("context_documents") or [])
        before = {operand: operand in operation["operand_evaluation"]["found_operands"] for operand in missing}
        after = {operand: bool(search["outside_context_candidate_has_operand"].get(operand)) for operand in missing}
        record = {
            "question_id": question_id, "question": source["question"], "schema_recognized": recognized,
            "status": status, "missing_operands": missing, "found_candidates": search["found_candidates"],
            "before_context_has_operand": bool(missing) and all(before.values()),
            "after_candidate_has_operand": bool(missing) and any(after.values()),
            "before_context_operand_status": before, "after_candidate_operand_status": after,
            "potentially_recoverable": bool(search["all_missing_operands_recoverable"]),
        }
        records.append(record)
        print(f"[{index:02d}/15] {question_id}: {status} recoverable={record['potentially_recoverable']}", flush=True)
    payload = {"schema": "evidence_operand_completion_shadow_v1", "summary": summarize(records), "records": records}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.output_json}\nMarkdown: {args.output_markdown}")


if __name__ == "__main__":
    main()
