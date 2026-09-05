"""Evaluate deterministic intent alignment on the frozen 15 answer failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from evidence_intent_alignment_v1 import (  # noqa: E402
    build_frozen_context_chunks_v1,
    classify_context_alignment_v1,
    extract_question_intent_v1,
)


DEFAULT_ANSWERS = ROOT / "reports/jina_full_baseline_input120_all100/answers.jsonl"
DEFAULT_AUDIT = ROOT / "reports/answer_failure_audit_v1.json"
DEFAULT_JSON = ROOT / "reports/evidence_intent_alignment_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/evidence_intent_alignment_v1.md"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(records: list[dict]) -> dict:
    categories = Counter(record["alignment"]["classification"] for record in records)
    aligned_count = sum(record["alignment"]["aligned_evidence_present"] for record in records)
    return {
        "questions": len(records), "aligned_evidence_present": aligned_count,
        "alignment_error_candidates": len(records) - aligned_count,
        "coarse_intent_alignment_is_primary_failure_signal": aligned_count < len(records) / 2,
        "classifications": dict(categories),
        "average_context_chunks": round(sum(record["context_chunk_count"] for record in records) / len(records), 2),
        "average_aligned_chunks": round(sum(len(record["alignment"]["aligned_chunk_ids"]) for record in records) / len(records), 2),
        "calculation_types": dict(Counter(record["intent"]["calculation_type"] for record in records)),
        "external_calls": {"llm": 0, "jina": 0, "judge": 0, "langsmith": 0, "retrieval": 0},
    }


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evidence Intent Alignment Shadow v1", "",
        "本报告只分析冻结 Jina baseline 的实际最终 context。Intent 来自问题；参考答案和人工 failure category 不参与对齐计算。未调用 LLM、Jina、Judge、LangSmith 或 Retrieval。", "",
        "## 汇总", "",
        f"- 题目：{summary['questions']}",
        f"- 包含 aligned evidence：{summary['aligned_evidence_present']}/{summary['questions']}",
        f"- Alignment error候选：{summary['alignment_error_candidates']}",
        f"- 粗粒度intent alignment是否为主要failure信号：`{summary['coarse_intent_alignment_is_primary_failure_signal']}`",
        f"- 分类：`{json.dumps(summary['classifications'], ensure_ascii=False)}`",
        f"- 平均 context chunks：{summary['average_context_chunks']}",
        f"- 平均 aligned chunks：{summary['average_aligned_chunks']}",
        f"- Calculation types：`{json.dumps(summary['calculation_types'], ensure_ascii=False)}`", "",
        "A 表示冻结 context 没有规则可识别的 metric evidence；B 表示存在 metric evidence，但 entity/period 未对齐；C 表示至少一个 chunk 三者均对齐而答案仍失败。C 只证明粗粒度 entity/period/metric 对齐，不证明 operands 完整或证据足够。", "",
        "## 逐题", "",
    ]
    for index, record in enumerate(payload["records"], 1):
        intent, alignment = record["intent"], record["alignment"]
        lines.extend([
            f"### {index}. {record['question_id']} — `{record['audit_failure_type']}`", "",
            f"**问题：** {record['question']}", "",
            f"- Entity intent：`{[item['value'] for item in intent['entity_candidates']]}`",
            f"- Period intent：`{[item['value'] for item in intent['period_candidates']]}`",
            f"- Metric intent：`{[item['value'] for item in intent['metric_candidates']]}`",
            f"- Calculation type：`{intent['calculation_type']}`",
            f"- Aligned evidence：`{alignment['aligned_evidence_present']}`",
            f"- 分类：`{alignment['classification']}`",
            f"- 原因：{alignment['reason']}", "",
            "**Top chunk alignment：**", "",
        ])
        ranked = sorted(alignment["chunk_alignments"], key=lambda item: (-item["alignment_score"], item["document"], item["page"]))
        for item in ranked[:8]:
            lines.append(
                f"- `{item['document']} p.{item['page']}` score={item['alignment_score']} "
                f"entity={item['entity_match']} period={item['period_match']} metric={item['metric_match']} aligned={item['aligned']}"
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
    answers = {row["financebench_id"]: row for row in load_jsonl(args.answers)}
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if len(audit.get("items", [])) != 15:
        raise ValueError("Expected the exact 15-item answer_failure_audit_v1 set")
    records = []
    for index, item in enumerate(audit["items"], 1):
        source = answers[item["question_id"]]
        intent = extract_question_intent_v1(source["question"])
        chunks = build_frozen_context_chunks_v1(source["evidence"], source.get("context_documents") or [])
        alignment = classify_context_alignment_v1(intent, chunks)
        records.append({
            "question_id": item["question_id"], "question": source["question"], "audit_failure_type": item["category"],
            "intent": intent, "context_chunk_count": len(chunks), "alignment": alignment,
        })
        print(f"[{index:02d}/15] {item['question_id']}: {alignment['classification']}", flush=True)
    payload = {"schema": "evidence_intent_alignment_shadow_v1", "summary": summarize(records), "records": records}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.output_json}\nMarkdown: {args.output_markdown}")


if __name__ == "__main__":
    main()
