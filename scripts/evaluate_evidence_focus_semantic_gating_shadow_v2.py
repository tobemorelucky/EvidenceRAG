"""Offline replay of Evidence Focus Semantic Gating Shadow v2.

Uses the same 90 frozen baseline/Focus answers and Judge verdicts as the prior
gating experiment. No model, retrieval, rerank, or context calls are made.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from evidence_focus_semantic_router_v2 import route_evidence_focus_semantic  # noqa: E402
from evaluate_evidence_focus_prompt_gating_shadow_v1 import load_replay_inputs  # noqa: E402


DEFAULT_OUTPUT = ROOT / "reports/evidence_focus_semantic_gating_shadow_v2"


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


def replay(source_records: list[dict]) -> list[dict]:
    records = []
    for source in source_records:
        route = route_evidence_focus_semantic(source["question"])
        selected_name = "evidence_focus" if route["use_evidence_focus"] else "baseline"
        selected = source[selected_name]
        baseline_correct = source["baseline"]["judge"].get("score") == 1
        selected_correct = selected["judge"].get("score") == 1
        transition = (
            "still_correct" if baseline_correct and selected_correct
            else "regressed" if baseline_correct
            else "newly_correct" if selected_correct
            else "still_incorrect"
        )
        records.append(
            {
                **source,
                "route": route,
                "selected_prompt": "finance_evidence_focus_v1" if route["use_evidence_focus"] else "clean_baseline_v1",
                "gated_answer": selected["answer"],
                "gated_judge": selected["judge"],
                "comparison": {
                    "transition": transition,
                    "answer_changed": source["baseline"]["answer"].strip() != selected["answer"].strip(),
                },
            }
        )
    return records


def summarize(records: list[dict]) -> dict:
    transitions = {name: [] for name in ("newly_correct", "regressed", "still_incorrect", "still_correct")}
    rule_stats: dict[str, dict] = {}
    for record in records:
        transition = record["comparison"]["transition"]
        transitions[transition].append(record["question_id"])
        for reason in record["route"]["reasons"]:
            stats = rule_stats.setdefault(reason, {"triggered": 0, "covered_failures": 0, "covered_correct": 0, "fixed": 0, "regressed": 0})
            baseline_correct = record["baseline"]["judge"].get("score") == 1
            stats["triggered"] += 1
            stats["covered_correct"] += int(baseline_correct)
            stats["covered_failures"] += int(not baseline_correct)
            stats["fixed"] += int(transition == "newly_correct")
            stats["regressed"] += int(transition == "regressed")
    for stats in rule_stats.values():
        stats["net_gain"] = stats["fixed"] - stats["regressed"]
    triggered_records = [record for record in records if record["route"]["use_evidence_focus"]]
    triggered_failures = sum(record["baseline"]["judge"].get("score") == 0 for record in triggered_records)
    triggered_correct = len(triggered_records) - triggered_failures
    fixed = len(transitions["newly_correct"])
    regressed = len(transitions["regressed"])
    passed = fixed - regressed >= 5 and regressed <= 2
    baseline_correct = sum(record["baseline"]["judge"].get("score") == 1 for record in records)
    gated_correct = sum(record["gated_judge"].get("score") == 1 for record in records)
    return {
        "experiment": "evidence_focus_semantic_gating_shadow_v2",
        "questions": len(records),
        "new_external_calls": {"answer": 0, "judge": 0, "retrieval": 0, "jina": 0},
        "trigger": {
            "count": len(triggered_records),
            "rate": len(triggered_records) / len(records),
            "covered_baseline_failures": triggered_failures,
            "covered_baseline_correct": triggered_correct,
            "not_triggered": len(records) - len(triggered_records),
        },
        "strict_judge": {
            "baseline_correct": baseline_correct,
            "gated_correct": gated_correct,
            "baseline_accuracy": baseline_correct / len(records),
            "gated_accuracy": gated_correct / len(records),
        },
        "transitions": {name: {"count": len(ids), "question_ids": ids} for name, ids in transitions.items()},
        "fixed": fixed,
        "regressed": regressed,
        "net_gain": fixed - regressed,
        "rule_performance": rule_stats,
        "acceptance": {
            "net_gain_at_least_5": fixed - regressed >= 5,
            "regressions_at_most_2": regressed <= 2,
            "passed": passed,
            "recommendation": "production_review" if passed else "stop_semantic_gating_keep_shadow",
        },
    }


def write_report(output: Path, records: list[dict], summary: dict) -> None:
    atomic_jsonl(output / "results.jsonl", records)
    atomic_json(output / "summary.json", summary)
    trigger = summary["trigger"]
    strict = summary["strict_judge"]
    lines = [
        "# Evidence Focus Semantic Gating Shadow v2", "",
        "本实验离线复用 final100 冻结 context 和已有 baseline/Focus 答案及 Judge；没有重新调用 Retrieval、Query Rewrite、Jina、Answer 或 Judge。", "",
        "## 汇总", "",
        f"- 样本：**{summary['questions']}**（22 个原 answer failure + 68 个原正确）。",
        f"- 触发：**{trigger['count']}**；未触发：**{trigger['not_triggered']}**。",
        f"- 触发覆盖原错误 / 原正确：**{trigger['covered_baseline_failures']} / {trigger['covered_baseline_correct']}**。",
        f"- Baseline / Gated Strict Judge：**{strict['baseline_correct']} / {strict['gated_correct']}**。",
        f"- 修复 / 回退 / 净提升：**{summary['fixed']} / {summary['regressed']} / {summary['net_gain']:+d}**。",
        f"- 验收：**{'通过，进入生产评审' if summary['acceptance']['passed'] else '未通过，停止该方向并保持 shadow'}**。", "",
        "## 规则收益（规则可重叠）", "",
        "| 规则 | 触发 | 覆盖错误 | 覆盖正确 | 修复 | 回退 | 净收益 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for reason, stats in summary["rule_performance"].items():
        lines.append(
            f"| {reason} | {stats['triggered']} | {stats['covered_failures']} | {stats['covered_correct']} | "
            f"{stats['fixed']} | {stats['regressed']} | {stats['net_gain']:+d} |"
        )
    lines.extend(["", "## 逐题结果", ""])
    for index, record in enumerate(records, 1):
        lines.extend([
            f"### {index}. {record['question_id']}", "",
            f"- Prompt：`{record['selected_prompt']}`",
            f"- Reasons：{', '.join(record['route']['reasons']) or '普通 lookup / 无推理信号'}",
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

