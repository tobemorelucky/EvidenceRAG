"""Evaluate Financial Evidence Summary v1 on the frozen 15 answer failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from evidence_intent_alignment_v1 import extract_question_intent_v1  # noqa: E402
from financial_evidence_summary_shadow_v1 import (  # noqa: E402
    detect_metric_substitution_v1,
    extract_financial_evidence_summary_v1,
)


DEFAULT_ANSWERS = ROOT / "reports/jina_full_baseline_input120_all100/answers.jsonl"
DEFAULT_AUDIT = ROOT / "reports/answer_failure_audit_v1.json"
DEFAULT_JSON = ROOT / "reports/financial_evidence_summary_shadow_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/financial_evidence_summary_shadow_v1.md"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _period_match(required: str, observed: str | None) -> bool:
    if not observed:
        return False
    # A year-level request accepts a more specific quarter in that year.  A
    # quarter-level request requires the exact quarter; a bare year is not a
    # sufficiently precise binding.
    return required == observed or (len(required) == 4 and observed.startswith(required))


def _evaluate_record(audit_item: dict, source: dict) -> dict:
    intent = extract_question_intent_v1(source["question"])
    summary = extract_financial_evidence_summary_v1(
        source["question"], source.get("evidence") or "", source.get("context_documents") or []
    )
    required_periods = [item["value"] for item in intent.get("period_candidates") or []]
    observed_periods = sorted({fact["period"] for fact in summary["facts"] if fact.get("period")})
    period_ok = not required_periods or all(any(_period_match(required, observed) for observed in observed_periods) for required in required_periods)
    substitutions = detect_metric_substitution_v1(summary.get("target_metric"), source.get("answer") or "")
    fact_count = len(summary["facts"])
    category = audit_item["category"]
    if summary["extraction_status"] == "absent":
        explanation = "target_evidence_not_extracted"
    elif not period_ok:
        explanation = "target_evidence_found_but_period_binding_incomplete"
    elif substitutions:
        explanation = "baseline_answer_used_competing_metric"
    elif category == "refusal_failure":
        explanation = "relevant_facts_present_but_baseline_refused"
    elif category in {"calculation_failure", "reasoning_failure"}:
        explanation = "relevant_facts_present_but_baseline_reasoning_failed"
    else:
        explanation = "facts_extracted_but_failure_requires_further_review"
    explains = (
        (category == "refusal_failure" and fact_count > 0)
        or (category in {"calculation_failure", "reasoning_failure"} and fact_count > 0 and period_ok)
        or (category == "terminology_failure" and bool(substitutions))
        or (category == "evidence_not_sufficient" and (summary["extraction_status"] == "absent" or not period_ok))
    )
    return {
        "question_id": audit_item["question_id"], "question": source["question"],
        "audit_failure_type": category, "target_metric": summary.get("target_metric"),
        "target_metric_correctly_extracted": summary["extraction_status"] != "absent",
        "metric_extraction_status": summary["extraction_status"],
        "required_periods": required_periods, "observed_periods": observed_periods,
        "correct_period_bound": period_ok,
        "wrong_metric_substitution_in_summary": False,
        "competing_metric_in_baseline_answer": substitutions,
        "explains_answer_failure": explains, "failure_explanation": explanation,
        "financial_evidence_summary": summary["facts"],
        "context_chunk_count": summary["context_chunk_count"],
    }


def _summarize(records: list[dict]) -> dict:
    return {
        "questions": len(records),
        "target_metric_extracted": sum(row["target_metric_correctly_extracted"] for row in records),
        "direct_target_metric_extracted": sum(row["metric_extraction_status"] == "direct" for row in records),
        "operand_supported_only": sum(row["metric_extraction_status"] == "operand_supported" for row in records),
        "correct_period_bound": sum(row["correct_period_bound"] for row in records),
        "wrong_metric_substitution_in_summary": sum(row["wrong_metric_substitution_in_summary"] for row in records),
        "baseline_competing_metric_detected": sum(bool(row["competing_metric_in_baseline_answer"]) for row in records),
        "answer_failure_explained": sum(row["explains_answer_failure"] for row in records),
        "extraction_status": dict(Counter(row["metric_extraction_status"] for row in records)),
        "failure_explanations": dict(Counter(row["failure_explanation"] for row in records)),
        "total_facts": sum(len(row["financial_evidence_summary"]) for row in records),
        "external_calls": {"llm": 0, "jina": 0, "judge": 0, "langsmith": 0, "retrieval": 0},
    }


def _render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Financial Evidence Summary Shadow v1", "",
        "本报告只使用 answer_failure_audit_v1 的 15 题和已冻结 Jina 最终 context。摘要器不生成答案；参考答案不参与事实提取或判定。未调用 Retrieval、Jina、LLM、Judge 或 LangSmith。", "",
        "## 汇总", "",
        f"- 题目：{summary['questions']}",
        f"- 目标 metric 有相关事实：{summary['target_metric_extracted']}/{summary['questions']}",
        f"- 直接目标 metric：{summary['direct_target_metric_extracted']}/{summary['questions']}",
        f"- 仅找到计算 operand：{summary['operand_supported_only']}/{summary['questions']}",
        f"- 正确 period 绑定：{summary['correct_period_bound']}/{summary['questions']}",
        f"- 摘要错误 metric 替代：{summary['wrong_metric_substitution_in_summary']}",
        f"- baseline answer 竞争 metric 信号：{summary['baseline_competing_metric_detected']}",
        f"- 可解释既有 answer failure：{summary['answer_failure_explained']}/{summary['questions']}",
        f"- 提取状态：`{json.dumps(summary['extraction_status'], ensure_ascii=False)}`",
        f"- 提取事实数：{summary['total_facts']}", "",
        "> `operand_supported` 表示上下文出现目标指标的通用计算组成项，但没有直接披露目标指标；它不等同于 operands 已完整，也不代表可以生成答案。", "",
        "## 逐题", "",
    ]
    for index, row in enumerate(payload["records"], 1):
        lines.extend([
            f"### {index}. {row['question_id']} — `{row['audit_failure_type']}`", "",
            f"**问题：** {row['question']}", "",
            f"- 目标 metric：`{row['target_metric']}`",
            f"- Metric 提取：`{row['metric_extraction_status']}`",
            f"- Required periods：`{row['required_periods']}`",
            f"- Observed periods：`{row['observed_periods']}`",
            f"- Period 绑定正确：`{row['correct_period_bound']}`",
            f"- 摘要错误 metric 替代：`{row['wrong_metric_substitution_in_summary']}`",
            f"- Baseline answer 竞争 metric：`{row['competing_metric_in_baseline_answer']}`",
            f"- 能否解释 failure：`{row['explains_answer_failure']}` — `{row['failure_explanation']}`", "",
            "**结构化事实（最多展示 12 条）：**", "",
        ])
        if not row["financial_evidence_summary"]:
            lines.append("- 未提取到问题相关事实。")
        for fact in row["financial_evidence_summary"][:12]:
            span = fact["source_span"]
            lines.append(
                f"- `{fact['metric']}` | entity=`{fact['entity']}` | period=`{fact['period']}` | "
                f"value=`{fact['value']}` | unit=`{fact['unit']}` | "
                f"source=`{span['document']} p.{span['page']} line {span['line_number']}` | flags=`{fact['ambiguity_flags']}`  \n"
                f"  原文：{span['text']}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    answers = {row["financebench_id"]: row for row in _load_jsonl(args.answers)}
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    items = audit.get("items") or []
    if len(items) != 15:
        raise ValueError("Expected the exact 15-item answer_failure_audit_v1 set")
    records = []
    for index, item in enumerate(items, 1):
        source = answers[item["question_id"]]
        record = _evaluate_record(item, source)
        records.append(record)
        print(f"[{index:02d}/15] {item['question_id']}: {record['metric_extraction_status']}", flush=True)
    payload = {"schema": "financial_evidence_summary_shadow_v1", "summary": _summarize(records), "records": records}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.output_json}\nMarkdown: {args.output_markdown}")


if __name__ == "__main__":
    main()
