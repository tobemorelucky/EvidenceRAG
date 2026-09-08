"""Offline replay of Evidence Focus Prompt Gating Shadow v1.

All baseline/focus answers and Judge verdicts are reused from completed frozen
experiments. This script makes no model, retrieval, rerank, or context calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from evidence_focus_prompt_router_v1 import route_evidence_focus_prompt  # noqa: E402


FINAL100 = ROOT / "reports/final100"
REGRESSION = ROOT / "reports/evidence_focus_prompt_regression_shadow_v1"
SAFETY = ROOT / "reports/evidence_focus_prompt_safety_regression_v1"
DEFAULT_OUTPUT = ROOT / "reports/evidence_focus_prompt_gating_shadow_v1"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_replay_inputs() -> tuple[list[dict], dict[str, dict]]:
    required = [
        FINAL100 / "answers.jsonl",
        FINAL100 / "judge_results.jsonl",
        REGRESSION / "results.jsonl",
        SAFETY / "results.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing completed shadow inputs: {missing}")
    final_answers = {record["question_id"]: record for record in load_jsonl(required[0])}
    final_judges = {record["question_id"]: record for record in load_jsonl(required[1])}
    focus_records = load_jsonl(required[2]) + load_jsonl(required[3])
    focus_by_id = {record["question_id"]: record for record in focus_records}
    if len(focus_records) != 90 or len(focus_by_id) != 90:
        raise ValueError("Expected 90 unique completed Focus records (32 + 58)")
    records = []
    for question_id, focus_record in focus_by_id.items():
        frozen = final_answers[question_id]
        if text_sha256(frozen["evidence"]) != focus_record["frozen_context_sha256"]:
            raise ValueError(f"Frozen context hash mismatch for {question_id}")
        focus = focus_record.get("evidence_focus", {})
        if focus.get("answer_status") != "ok" or focus.get("judge_status") != "ok":
            raise ValueError(f"Incomplete Focus answer/Judge for {question_id}")
        baseline_judge = final_judges[question_id]
        if baseline_judge.get("status") != "ok":
            raise ValueError(f"Incomplete baseline Judge for {question_id}")
        records.append(
            {
                "question_id": question_id,
                "question": focus_record["question"],
                "reference_answer": focus_record["reference_answer"],
                "context_chars": len(frozen["evidence"]),
                "candidate_evidence_chunks": len(frozen.get("context_documents") or []),
                "baseline": {
                    "answer": frozen["answer"],
                    "judge": baseline_judge["judge"],
                    "usage": frozen.get("usage", {}),
                },
                "evidence_focus": {
                    "answer": focus["answer"],
                    "judge": focus["judge"],
                    "usage": focus.get("usage", {}),
                },
            }
        )
    # Preserve final100 order without reading or rebuilding any context.
    order = {record["question_id"]: index for index, record in enumerate(load_jsonl(required[0]))}
    records.sort(key=lambda record: order[record["question_id"]])
    return records, final_answers


def replay(records: list[dict]) -> list[dict]:
    output = []
    for record in records:
        route = route_evidence_focus_prompt(
            record["question"],
            context_chars=record["context_chars"],
            candidate_evidence_chunks=record["candidate_evidence_chunks"],
        )
        selected_name = "evidence_focus" if route["use_evidence_focus"] else "baseline"
        selected = record[selected_name]
        baseline_correct = record["baseline"]["judge"].get("score") == 1
        selected_correct = selected["judge"].get("score") == 1
        transition = (
            "still_correct" if baseline_correct and selected_correct
            else "regressed" if baseline_correct
            else "newly_correct" if selected_correct
            else "still_incorrect"
        )
        output.append(
            {
                **record,
                "route": route,
                "selected_prompt": "finance_evidence_focus_v1" if route["use_evidence_focus"] else "clean_baseline_v1",
                "gated_answer": selected["answer"],
                "gated_judge": selected["judge"],
                "comparison": {
                    "transition": transition,
                    "answer_changed": record["baseline"]["answer"].strip() != selected["answer"].strip(),
                },
            }
        )
    return output


def summarize(records: list[dict]) -> dict:
    transition_ids = {name: [] for name in ("newly_correct", "regressed", "still_incorrect", "still_correct")}
    reason_stats: dict[str, dict] = {}
    for record in records:
        transition = record["comparison"]["transition"]
        transition_ids[transition].append(record["question_id"])
        for reason in record["route"]["reasons"]:
            stats = reason_stats.setdefault(reason, {"triggered": 0, "newly_correct": 0, "regressed": 0})
            stats["triggered"] += 1
            stats["newly_correct"] += int(transition == "newly_correct")
            stats["regressed"] += int(transition == "regressed")
    for stats in reason_stats.values():
        stats["net_gain"] = stats["newly_correct"] - stats["regressed"]
    baseline_correct = sum(record["baseline"]["judge"].get("score") == 1 for record in records)
    gated_correct = sum(record["gated_judge"].get("score") == 1 for record in records)
    gained = len(transition_ids["newly_correct"])
    regressed = len(transition_ids["regressed"])
    acceptance = gained - regressed >= 5 and regressed <= 2
    return {
        "experiment": "evidence_focus_prompt_gating_shadow_v1",
        "questions": len(records),
        "new_external_calls": {"answer": 0, "judge": 0, "retrieval": 0, "jina": 0},
        "strict_judge": {
            "baseline_correct": baseline_correct,
            "gated_correct": gated_correct,
            "baseline_accuracy": baseline_correct / len(records),
            "gated_accuracy": gated_correct / len(records),
        },
        "triggered": sum(record["route"]["use_evidence_focus"] for record in records),
        "not_triggered": sum(not record["route"]["use_evidence_focus"] for record in records),
        "transitions": {name: {"count": len(ids), "question_ids": ids} for name, ids in transition_ids.items()},
        "net_gain": gained - regressed,
        "trigger_rule_performance": reason_stats,
        "acceptance": {
            "net_gain_at_least_5": gained - regressed >= 5,
            "regressions_at_most_2": regressed <= 2,
            "passed": acceptance,
            "recommendation": "production_review" if acceptance else "keep_shadow",
        },
    }


def write_report(output: Path, records: list[dict], summary: dict) -> None:
    atomic_jsonl(output / "results.jsonl", records)
    atomic_json(output / "summary.json", summary)
    strict = summary["strict_judge"]
    transitions = summary["transitions"]
    lines = [
        "# Evidence Focus Prompt Gating Shadow v1", "",
        "本报告离线复用 final100 冻结 context 以及已完成的 baseline/Focus 答案和 Judge；没有重新调用任何模型、Retrieval、Query Rewrite、Jina 或 context builder。", "",
        "## 汇总", "",
        f"- 样本：**{summary['questions']}**（22 个原 answer failure + 68 个原正确）。",
        f"- 触发 / 未触发：**{summary['triggered']} / {summary['not_triggered']}**。",
        f"- Baseline Strict Judge：**{strict['baseline_correct']}/{summary['questions']}**。",
        f"- Gated Strict Judge：**{strict['gated_correct']}/{summary['questions']}**。",
        f"- 修复 / 回退 / 净提升：**{transitions['newly_correct']['count']} / {transitions['regressed']['count']} / {summary['net_gain']:+d}**。",
        f"- 验收：**{'通过' if summary['acceptance']['passed'] else '未通过，保持 shadow'}**。", "",
        "## 各触发规则收益", "",
        "| 规则 | 触发 | 新增正确 | 回退 | 净收益 |",
        "|---|---:|---:|---:|---:|",
    ]
    for reason, stats in summary["trigger_rule_performance"].items():
        lines.append(f"| {reason} | {stats['triggered']} | {stats['newly_correct']} | {stats['regressed']} | {stats['net_gain']:+d} |")
    lines.extend(["", "## 逐题结果", ""])
    for index, record in enumerate(records, 1):
        lines.extend([
            f"### {index}. {record['question_id']}", "",
            f"- Route：`{record['selected_prompt']}`",
            f"- Reasons：{', '.join(record['route']['reasons']) or '无'}",
            f"- Transition：`{record['comparison']['transition']}`",
            f"- Baseline Judge：{record['baseline']['judge'].get('verdict')} — {record['baseline']['judge'].get('reason', '')}",
            f"- Gated Judge：{record['gated_judge'].get('verdict')} — {record['gated_judge'].get('reason', '')}", "",
            "**问题**", "", record["question"], "",
            "**Baseline answer**", "", record["baseline"]["answer"], "",
            "**Gated answer**", "", record["gated_answer"], "",
        ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_records, _ = load_replay_inputs()
    records = replay(source_records)
    summary = summarize(records)
    write_report(args.output_dir.resolve(), records, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

