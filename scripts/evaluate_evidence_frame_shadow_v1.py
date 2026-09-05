"""Evaluate rule-only EvidenceFrame extraction on the frozen 15 failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from evidence_frame_shadow_v1 import build_evidence_frame_shadow_v1, explain_answer_failure  # noqa: E402


DEFAULT_ANSWERS = ROOT / "reports/jina_full_baseline_input120_all100/answers.jsonl"
DEFAULT_AUDIT = ROOT / "reports/answer_failure_audit_v1.json"
DEFAULT_JSON = ROOT / "reports/evidence_frame_shadow_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/evidence_frame_shadow_v1.md"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(records: list[dict]) -> dict:
    operand_applicable = [record for record in records if record["frame"]["diagnostics"]["required_operands_found"] is not None]
    return {
        "questions": len(records),
        "key_metric_found": sum(record["frame"]["diagnostics"]["key_metric_found"] for record in records),
        "requested_period_applicable": sum(bool(record["frame"]["diagnostics"]["requested_periods"]) for record in records),
        "requested_period_found": sum(record["frame"]["diagnostics"]["requested_period_found"] for record in records),
        "required_operands_applicable": len(operand_applicable),
        "required_operands_found": sum(record["frame"]["diagnostics"]["required_operands_found"] is True for record in operand_applicable),
        "failure_type_explained": sum(record["explains_audit_failure_type"] for record in records),
        "question_types": dict(Counter(record["frame"]["question_type"] for record in records)),
        "failure_signals": dict(Counter(signal for record in records for signal in record["failure_explanation"]["signals"])),
        "external_calls": {"llm": 0, "jina": 0, "judge": 0, "langsmith": 0, "retrieval": 0},
    }


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evidence Frame Shadow v1",
        "",
        "本报告只解析冻结的 Jina baseline 最终 context。EvidenceFrame 不读取参考答案；参考 failure category 仅用于生成后的离线解释命中统计。未调用 LLM、Jina、Judge、LangSmith 或 Retrieval。",
        "",
        "## 汇总",
        "",
        f"- 题目：{summary['questions']}",
        f"- 找到关键 metric：{summary['key_metric_found']}/{summary['questions']}",
        f"- 找到 requested period：{summary['requested_period_found']}/{summary['requested_period_applicable']}",
        f"- 显式 required operands 完整：{summary['required_operands_found']}/{summary['required_operands_applicable']}",
        f"- 能解释人工 failure category：{summary['failure_type_explained']}/{summary['questions']}",
        f"- Question type：`{json.dumps(summary['question_types'], ensure_ascii=False)}`",
        f"- Failure signals：`{json.dumps(summary['failure_signals'], ensure_ascii=False)}`",
        "",
        "`required operands` 只评估问题显式给出公式或 numerator/denominator 的题；不会偷偷补充隐含金融公式。",
        "",
        "## 逐题",
        "",
    ]
    for index, record in enumerate(payload["records"], 1):
        frame = record["frame"]
        diagnostics = frame["diagnostics"]
        lines.extend([
            f"### {index}. {record['question_id']} — `{record['audit_failure_type']}`",
            "",
            f"**问题：** {record['question']}",
            "",
            f"- Question type：`{frame['question_type']}`",
            f"- 关键 metric：`{[item['value'] for item in frame['metric_candidates']]}`；found=`{diagnostics['key_metric_found']}`",
            f"- Period：requested=`{diagnostics['requested_periods']}`；found=`{diagnostics['requested_period_found']}`",
            f"- Required operands：`{diagnostics['required_operand_names']}`；status=`{diagnostics['required_operands_status']}`",
            f"- Formula：`{[item['expression'] for item in frame['formula_candidates']]}`",
            f"- Failure signals：`{record['failure_explanation']['signals']}`",
            f"- 可解释人工分类：`{record['explains_audit_failure_type']}`",
            "",
            "**Top evidence spans：**",
            "",
        ])
        for span in frame["evidence_spans"][:5]:
            lines.append(f"- `{span['document']} p.{span['page']}` score={span['score']}: {span['text']}")
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
        frame = build_evidence_frame_shadow_v1(source["question"], source["evidence"], source.get("context_documents") or [])
        explanation = explain_answer_failure(frame, source["answer"])
        record = {
            "question_id": item["question_id"], "question": source["question"],
            "audit_failure_type": item["category"], "frame": frame,
            "failure_explanation": explanation,
            "explains_audit_failure_type": item["category"] in explanation["explained_failure_types"],
        }
        records.append(record)
        print(
            f"[{index:02d}/15] {item['question_id']}: metric={frame['diagnostics']['key_metric_found']} "
            f"period={frame['diagnostics']['requested_period_found']} operands={frame['diagnostics']['required_operands_status']}",
            flush=True,
        )
    payload = {"schema": "evidence_frame_shadow_v1", "summary": summarize(records), "records": records}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.output_json}\nMarkdown: {args.output_markdown}")


if __name__ == "__main__":
    main()

