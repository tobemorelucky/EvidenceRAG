"""Audit answer failures in the frozen Jina full-baseline artifacts.

This script is deliberately offline: it reads the completed answer records and the
FinanceBench CSV, then writes a reproducible report.  It does not import or call any
retrieval, reranking, model, judge, or tracing client.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWERS = ROOT / "reports/jina_full_baseline_input120_all100/answers.jsonl"
DEFAULT_DATASET = ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_JSON = ROOT / "reports/answer_failure_audit_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/answer_failure_audit_v1.md"

ALLOWED_CATEGORIES = {
    "calculation_failure",
    "terminology_failure",
    "reasoning_failure",
    "evidence_not_sufficient",
    "refusal_failure",
    "other",
}

# Human-reviewed annotations over the frozen 15-record audit set.  These are report
# labels, not runtime rules, and are never imported by the production pipeline.
ANNOTATIONS = {
    "financebench_id_00606": (
        "reasoning_failure",
        "The answer notices that store payroll and benefits deleveraged, but makes the "
        "opposite final choice by substituting the direction of total SG&A.",
    ),
    "financebench_id_01911": (
        "refusal_failure",
        "The answer refuses to derive the requested result even though the supplied "
        "reconciliation and interest expense support the benchmark conclusion.",
    ),
    "financebench_id_00678": (
        "terminology_failure",
        "The answer substitutes operating margin for gross margin and then concludes "
        "that gross margin is not useful instead of deriving the requested measure.",
    ),
    "financebench_id_03856": (
        "calculation_failure",
        "The answer states 1.47, then says the ratio cannot be calculated, while the "
        "requested cash-flow/current-liability calculation should produce 0.83.",
    ),
    "financebench_id_00790": (
        "refusal_failure",
        "The answer lists relevant asset-base evidence but declines to make the "
        "requested evidence-based financial judgment.",
    ),
    "financebench_id_01279": (
        "reasoning_failure",
        "The answer reports operations at 3,565 and investing at 1,999 but selects "
        "investing as the largest, contradicting its own operands.",
    ),
    "financebench_id_00540": (
        "refusal_failure",
        "The answer rejects inventory turnover as not meaningful and does not perform "
        "the calculation expected from the available financial values.",
    ),
    "financebench_id_00651": (
        "reasoning_failure",
        "The answer mixes guidance periods and a later post-separation update, producing "
        "acceleration instead of the requested FY2022-to-FY2023 comparison.",
    ),
    "financebench_id_01936": (
        "evidence_not_sufficient",
        "The final context contains a truncated restructuring table without the "
        "December 2022 liability row needed to establish the 81/93 employee share.",
    ),
    "financebench_id_00711": (
        "refusal_failure",
        "The answer says inventory balances are absent and refuses the turnover "
        "calculation, although the frozen run marks the relevant evidence span present.",
    ),
    "financebench_id_00460": (
        "reasoning_failure",
        "The answer compares a domestic subset rather than the total store population "
        "requested by the question.",
    ),
    "financebench_id_00603": (
        "other",
        "The model's three drivers are present in the benchmark gold evidence, but the "
        "reference answer and Judge accept only the new-store driver; this is primarily "
        "a reference/Judge completeness mismatch.",
    ),
    "financebench_id_00005": (
        "other",
        "The model applies the standard total-current-assets minus total-current-"
        "liabilities definition.  The reference silently uses an operating-only subset, "
        "so the failure is driven by an unstated benchmark convention.",
    ),
    "financebench_id_02416": (
        "reasoning_failure",
        "The answer follows newer 10-Q acquisitions instead of respecting the question's "
        "specified 10-K document scope.",
    ),
    "financebench_id_02419": (
        "reasoning_failure",
        "The answer treats the historical completion date as disqualifying and reaches "
        "the opposite conclusion from the report-level question.",
    ),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_references(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["financebench_id"]: row for row in csv.DictReader(handle)}


def select_failures(records: list[dict]) -> list[dict]:
    """Select the 15 failures implied by the published 73/58 funnel.

    The stored artifact calls the 73-record signal ``candidate_gold_page_hit`` under
    ``actual_context``.  Its stricter ``context_hit`` signal contains 70 records and
    would yield 13 failures, so it cannot represent the requested 15-record cohort.
    """
    return [
        record
        for record in records
        if not bool(record.get("judge", {}).get("score"))
        and bool(
            record.get("retrieval_metrics", {})
            .get("actual_context", {})
            .get("candidate_gold_page_hit")
        )
    ]


def build_audit(records: list[dict], references: dict[str, dict]) -> dict:
    failures = select_failures(records)
    ids = {record["financebench_id"] for record in failures}
    if len(failures) != 15:
        raise ValueError(f"Expected 15 frozen failures, found {len(failures)}")
    if ids != set(ANNOTATIONS):
        raise ValueError(
            "Frozen failure cohort changed: "
            f"missing_annotations={sorted(ids - set(ANNOTATIONS))}, "
            f"stale_annotations={sorted(set(ANNOTATIONS) - ids)}"
        )

    items = []
    for record in failures:
        question_id = record["financebench_id"]
        reference = references[question_id]
        category, rationale = ANNOTATIONS[question_id]
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"Invalid category for {question_id}: {category}")
        metrics = record["retrieval_metrics"]["actual_context"]
        items.append(
            {
                "question_id": question_id,
                "question_type": reference.get("question_type"),
                "question": record["question"],
                "reference_answer": reference.get("answer"),
                "model_answer": record.get("answer"),
                "judge_reason": record.get("judge", {}).get("reason"),
                "category": category,
                "classification_rationale": rationale,
                "artifact_metrics": {
                    "candidate_gold_page_hit": bool(metrics.get("candidate_gold_page_hit")),
                    "context_hit": bool(metrics.get("context_hit")),
                    "context_all_gold_pages_hit": bool(metrics.get("context_all_gold_pages_hit")),
                    "context_evidence_span_hit": bool(metrics.get("context_evidence_span_hit")),
                    "gold_page_rank": metrics.get("gold_page_rank"),
                },
                "citations": record.get("citations", []),
            }
        )

    counts = Counter(item["category"] for item in items)
    strict_context_failures = sum(item["artifact_metrics"]["context_hit"] for item in items)
    return {
        "audit": "answer_failure_audit_v1",
        "source": "reports/jina_full_baseline_input120_all100/answers.jsonl",
        "selection_contract": {
            "judge_false": True,
            "stored_hit_field": "retrieval_metrics.actual_context.candidate_gold_page_hit",
            "reason": (
                "This is the stored 73-hit signal used by the published 73 total / "
                "58 correct / 15 failure funnel. The stricter actual_context.context_hit "
                "signal has 70 total and 13 Judge-false records."
            ),
        },
        "summary": {
            "audited": len(items),
            "strict_context_hit_within_audit": strict_context_failures,
            "strict_context_miss_within_audit": len(items) - strict_context_failures,
            "category_counts": {name: counts.get(name, 0) for name in sorted(ALLOWED_CATEGORIES)},
        },
        "items": items,
    }


def render_markdown(audit: dict) -> str:
    summary = audit["summary"]
    lines = [
        "# Answer Failure Audit v1",
        "",
        "## 范围与口径",
        "",
        "- 数据源：`reports/jina_full_baseline_input120_all100/answers.jsonl`。",
        "- 仅审计已有结果；没有调用 LLM、Judge、Jina 或 LangSmith。",
        "- 目标集合：`Judge=false` 且 "
        "`retrieval_metrics.actual_context.candidate_gold_page_hit=true`，共 15 题。",
        "- 注意：该字段是现有 73/58/15 漏斗中的“gold context hit”口径，但字段名实际表示候选 gold page 命中。"
        " 更严格的 `actual_context.context_hit` 在全量中为 70 题，本集合中为 "
        f"{summary['strict_context_hit_within_audit']} 题。",
        "",
        "## 分类汇总",
        "",
        "| 分类 | 数量 | 占比 |",
        "|---|---:|---:|",
    ]
    for category, count in summary["category_counts"].items():
        lines.append(f"| `{category}` | {count} | {count / summary['audited']:.1%} |")

    lines.extend(["", "## 逐题审计", ""])
    for index, item in enumerate(audit["items"], 1):
        metrics = item["artifact_metrics"]
        citation_text = "; ".join(
            f"{citation.get('filename')}, page {citation.get('page_number')}"
            for citation in item["citations"]
        ) or "无"
        lines.extend(
            [
                f"### {index}. {item['question_id']} — `{item['category']}`",
                "",
                f"- **问题类型**：{item.get('question_type') or '未知'}",
                f"- **问题**：{item['question']}",
                f"- **参考答案**：{item['reference_answer']}",
                f"- **模型答案**：{item['model_answer']}",
                f"- **Judge 原因**：{item['judge_reason']}",
                f"- **分类依据**：{item['classification_rationale']}",
                "- **命中指标**："
                f"candidate_gold_page_hit={metrics['candidate_gold_page_hit']}，"
                f"context_hit={metrics['context_hit']}，"
                f"context_evidence_span_hit={metrics['context_evidence_span_hit']}，"
                f"gold_page_rank={metrics['gold_page_rank']}",
                f"- **引用**：{citation_text}",
                "",
            ]
        )

    lines.extend(
        [
            "## 结论",
            "",
            "- 主要可归因于回答阶段的问题是 `reasoning_failure` 与 `refusal_failure`。",
            "- `evidence_not_sufficient` 说明候选页命中不等于最终上下文中保留了完整事实。",
            "- `other` 两题存在参考答案口径或 Judge 完整性问题，不宜据此向生产提示词加入单题规则。",
            "- 本报告是冻结结果的离线诊断，不构成运行时分类器或生产策略。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    audit = build_audit(read_jsonl(args.answers), read_references(args.dataset))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_output.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.json_output}")
    print(f"Markdown: {args.markdown_output}")


if __name__ == "__main__":
    main()
