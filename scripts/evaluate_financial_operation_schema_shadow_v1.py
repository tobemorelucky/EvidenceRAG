"""Evaluate Financial Operation Schema v1 on the frozen 15 answer failures."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from financial_operation_schema_v1 import build_financial_operation_schema_v1, find_required_operands_v1  # noqa: E402


DEFAULT_ANSWERS = ROOT / "reports/jina_full_baseline_input120_all100/answers.jsonl"
DEFAULT_AUDIT = ROOT / "reports/answer_failure_audit_v1.json"
DEFAULT_JSON = ROOT / "reports/financial_operation_schema_shadow_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/financial_operation_schema_shadow_v1.md"
_REFUSAL_RE = re.compile(r"\b(?:cannot|can't|unable to|insufficient|not enough|not meaningful|does not provide)\b", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def explain(schema: dict, operands: dict, baseline_answer: str) -> dict:
    if not schema["recognized"]:
        return {"can_explain": False, "bottleneck": "operation_schema_out_of_scope", "reason": "Question is outside the eight v1 financial operations."}
    if not operands["complete"]:
        return {
            "can_explain": True, "bottleneck": "frozen_context_operand_gap",
            "reason": f"Schema identified the operation, but frozen context lacks: {', '.join(operands['missing_operands'])}.",
        }
    refusal = bool(_REFUSAL_RE.search(str(baseline_answer or "")))
    return {
        "can_explain": True,
        "bottleneck": "operation_structure_missing_or_unused" if refusal else "operation_execution_or_metric_mapping_failure",
        "reason": "All schema operands are present, but the baseline refused." if refusal else "All schema operands are present, but the stored answer remained incorrect.",
    }


def summary(records: list[dict]) -> dict:
    recognized = [record for record in records if record["schema"]["recognized"]]
    complete = [record for record in recognized if record["operand_evaluation"]["complete"]]
    return {
        "questions": len(records), "schema_recognized": len(recognized),
        "required_operands_complete": len(complete),
        "required_operands_incomplete": len(recognized) - len(complete),
        "failure_explained": sum(record["failure_explanation"]["can_explain"] for record in records),
        "operation_structure_candidates": sum(record["failure_explanation"]["bottleneck"].startswith("operation_") and record["schema"]["recognized"] for record in records),
        "frozen_context_operand_gaps": sum(record["failure_explanation"]["bottleneck"] == "frozen_context_operand_gap" for record in records),
        "recognized_metrics": dict(Counter(record["schema"]["metric"] for record in recognized)),
        "bottlenecks": dict(Counter(record["failure_explanation"]["bottleneck"] for record in records)),
        "external_calls": {"llm": 0, "jina": 0, "judge": 0, "langsmith": 0, "retrieval": 0},
    }


def markdown(payload: dict) -> str:
    values = payload["summary"]
    lines = [
        "# Financial Operation Schema Shadow v1", "",
        "固定复用 Jina baseline 的实际最终 context。Schema 和 operand matching 均为确定性规则；参考答案未参与识别或匹配。未调用 LLM、Jina、Judge、LangSmith 或 Retrieval。", "",
        "## 汇总", "",
        f"- 题目：{values['questions']}",
        f"- Schema 识别：{values['schema_recognized']}/{values['questions']}",
        f"- Required operands 完整：{values['required_operands_complete']}/{values['schema_recognized']}",
        f"- Frozen context operand gap：{values['frozen_context_operand_gaps']}",
        f"- Operation structure候选：{values['operation_structure_candidates']}",
        f"- 能解释failure来源：{values['failure_explained']}/{values['questions']}",
        f"- Metrics：`{json.dumps(values['recognized_metrics'], ensure_ascii=False)}`",
        f"- Bottlenecks：`{json.dumps(values['bottlenecks'], ensure_ascii=False)}`", "",
        "只有 operands 完整但答案仍失败的题，才支持“缺少财务计算结构”的假设；operand gap 不应归因于回答模型。", "",
        "## 逐题", "",
    ]
    for index, record in enumerate(payload["records"], 1):
        schema, operands, explanation = record["schema"], record["operand_evaluation"], record["failure_explanation"]
        lines.extend([
            f"### {index}. {record['question_id']} — `{record['audit_failure_type']}`", "",
            f"**问题：** {record['question']}", "",
            f"- Schema成功：`{schema['recognized']}`",
            f"- Metric / operation：`{schema['metric']}` / `{schema['operation_type']}`",
            f"- Formula：`{schema['formula']}`",
            f"- Period requirement：`{schema['period_requirement']}`",
            f"- Entity requirement：`{schema['entity_requirement']}`",
            f"- Required operands：`{[item['key'] for item in schema['required_operands']]}`",
            f"- Found operands：`{list(operands['found_operands'])}`",
            f"- Missing operands：`{operands['missing_operands']}`",
            f"- 能解释failure：`{explanation['can_explain']}` — `{explanation['bottleneck']}`",
            f"- 解释：{explanation['reason']}", "",
        ])
        for key, candidates in operands["found_operands"].items():
            first = candidates[0]
            lines.append(f"  - `{key}`: {first['document']} p.{first['page']} — {first['text']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    answers = {row["financebench_id"]: row for row in load_jsonl(args.answers)}
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if len(audit.get("items", [])) != 15:
        raise ValueError("Expected the exact 15-item answer_failure_audit_v1 set")
    records = []
    for index, item in enumerate(audit["items"], 1):
        source = answers[item["question_id"]]
        schema = build_financial_operation_schema_v1(source["question"])
        operands = find_required_operands_v1(schema, source["evidence"], source.get("context_documents") or [])
        explanation = explain(schema, operands, source["answer"])
        records.append({
            "question_id": item["question_id"], "question": source["question"], "audit_failure_type": item["category"],
            "schema": schema, "operand_evaluation": operands, "failure_explanation": explanation,
        })
        print(f"[{index:02d}/15] {item['question_id']}: schema={schema['recognized']} complete={operands['complete']}", flush=True)
    payload = {"schema": "financial_operation_schema_shadow_v1", "summary": summary(records), "records": records}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_markdown.write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.output_json}\nMarkdown: {args.output_markdown}")


if __name__ == "__main__":
    main()

