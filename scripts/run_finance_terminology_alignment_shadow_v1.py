"""Run a three-question prompt-only A/B over frozen final100 evidence.

This script does not import or call retrieval, Jina, Judge, or LangSmith.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

SOURCE_ANSWERS = ROOT / "reports" / "final100" / "answers.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "finance_terminology_alignment_shadow_v1"
PROMPT_MODES = ("clean_baseline_v1", "finance_terminology_alignment_v1")

CASES = (
    {
        "question_id": "financebench_id_04735",
        "question": (
            "Adobe FY2015 的经营现金流比率是多少？经营现金流比率定义为"
            "“经营活动现金流 ÷ 流动负债合计”。请保留两位小数，并引用文件和页码。"
        ),
        "required_number_groups": (("0.66",),),
        "direction_terms": (),
    },
    {
        "question_id": "financebench_id_03856",
        "question": (
            "把财年改为 FY2017，仍使用同一公式重新计算 Adobe 的经营现金流比率。"
            "经营现金流比率定义为“经营活动现金流 ÷ 流动负债合计”。"
            "请保留两位小数，并引用文件和页码。"
        ),
        "required_number_groups": (("0.83",),),
        "direction_terms": (),
    },
    {
        "question_id": "financebench_id_00438",
        "question": (
            "截至 FY2022，Adobe 的经营利润率是否呈改善趋势？"
            "请计算 FY2021 和 FY2022 的经营利润率、说明变化方向，并引用文件和页码。"
        ),
        "required_number_groups": (("36.8", "36.76"), ("34.6", "34.63")),
        "direction_terms": ("下降", "降低", "下滑", "declin", "decreas", "not improv", "没有改善", "未改善"),
    },
)

REFUSAL_PATTERNS = (
    r"无法(?:直接)?(?:计算|回答|确认|确定)",
    r"证据不足",
    r"信息不足",
    r"缺少(?:所需|必要|足够)",
    r"cannot (?:calculate|answer|determine|confirm)",
    r"insufficient (?:evidence|information)",
    r"not enough (?:evidence|information)",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_frozen_cases(source_path: Path = SOURCE_ANSWERS) -> list[dict]:
    source = {row["question_id"]: row for row in _load_jsonl(source_path)}
    records = []
    for case in CASES:
        frozen = source.get(case["question_id"])
        if not frozen or frozen.get("status") != "ok" or not frozen.get("evidence"):
            raise ValueError(f"Missing completed frozen evidence for {case['question_id']}")
        records.append({
            **case,
            "evidence": str(frozen["evidence"]),
            "evidence_sha256": _sha256_text(str(frozen["evidence"])),
            "source_context_metrics": dict(frozen.get("context_metrics") or {}),
            "results": {},
        })
    return records


def assess_answer(answer: str, case: dict) -> dict:
    normalized = str(answer or "").replace(",", "").casefold()
    required_groups = case.get("required_number_groups") or ()
    number_groups_found = [
        any(str(candidate).casefold() in normalized for candidate in alternatives)
        for alternatives in required_groups
    ]
    unnecessary_refusal = any(re.search(pattern, normalized, re.IGNORECASE) for pattern in REFUSAL_PATTERNS)
    direction_terms = case.get("direction_terms") or ()
    direction_ok = not direction_terms or any(term.casefold() in normalized for term in direction_terms)
    correct = all(number_groups_found) and direction_ok and not unnecessary_refusal
    return {
        "required_number_hit": all(number_groups_found),
        "required_number_groups_found": number_groups_found,
        "direction_ok": direction_ok,
        "unnecessary_refusal": unnecessary_refusal,
        "acceptance_pass": correct,
    }


def _usage_total(usage: dict) -> int:
    return int(usage.get("total_tokens") or usage.get("total_token_count") or 0)


def build_summary(payload: dict) -> dict:
    comparisons = []
    token_delta = []
    for record in payload["records"]:
        baseline = record["results"].get("clean_baseline_v1") or {}
        experiment = record["results"].get("finance_terminology_alignment_v1") or {}
        baseline_pass = bool((baseline.get("assessment") or {}).get("acceptance_pass"))
        experiment_pass = bool((experiment.get("assessment") or {}).get("acceptance_pass"))
        if baseline.get("status") == "ok" and experiment.get("status") == "ok":
            token_delta.append(_usage_total(experiment.get("usage") or {}) - _usage_total(baseline.get("usage") or {}))
        comparisons.append({
            "question_id": record["question_id"],
            "baseline_pass": baseline_pass,
            "alignment_pass": experiment_pass,
            "recovered": not baseline_pass and experiment_pass,
            "regression": baseline_pass and not experiment_pass,
            "baseline_unnecessary_refusal": bool((baseline.get("assessment") or {}).get("unnecessary_refusal")),
            "alignment_unnecessary_refusal": bool((experiment.get("assessment") or {}).get("unnecessary_refusal")),
        })
    return {
        "questions": len(comparisons),
        "baseline_correct": sum(item["baseline_pass"] for item in comparisons),
        "alignment_correct": sum(item["alignment_pass"] for item in comparisons),
        "recovered": sum(item["recovered"] for item in comparisons),
        "regressions": sum(item["regression"] for item in comparisons),
        "baseline_unnecessary_refusals": sum(item["baseline_unnecessary_refusal"] for item in comparisons),
        "alignment_unnecessary_refusals": sum(item["alignment_unnecessary_refusal"] for item in comparisons),
        "average_total_token_delta": round(sum(token_delta) / len(token_delta), 2) if token_delta else None,
        "acceptance": {
            "fy2015_restored_to_0_66": comparisons[0]["alignment_pass"],
            "fy2017_remains_0_83": comparisons[1]["alignment_pass"],
            "fy2022_direction_consistent": comparisons[2]["alignment_pass"],
            "no_regression": not any(item["regression"] for item in comparisons),
        },
        "comparisons": comparisons,
    }


def write_outputs(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    temporary = output_dir / "state.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(state_path)
    completed = [
        {key: value for key, value in record.items() if key != "evidence"}
        for record in payload["records"]
    ]
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in completed),
        encoding="utf-8",
    )
    summary = build_summary(payload)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    lines = [
        "# Finance Terminology Alignment v1 Shadow",
        "",
        "三题严格复用 final100 已冻结 evidence；未调用 Retrieval、Jina、Judge 或 LangSmith。",
        "",
        "## 汇总",
        "",
        f"- Clean baseline：{summary['baseline_correct']}/3",
        f"- Terminology alignment：{summary['alignment_correct']}/3",
        f"- 恢复：{summary['recovered']}；回退：{summary['regressions']}",
        f"- 平均总 token 变化：{summary['average_total_token_delta']}",
        "",
    ]
    for index, record in enumerate(payload["records"], 1):
        lines.extend([f"## {index}. {record['question_id']}", "", record["question"], ""])
        for mode in PROMPT_MODES:
            result = record["results"].get(mode) or {}
            lines.extend([
                f"### {mode}", "",
                result.get("answer") or f"状态：{result.get('status', 'pending')}", "",
                f"- assessment: `{json.dumps(result.get('assessment') or {}, ensure_ascii=False)}`",
                f"- usage: `{json.dumps(result.get('usage') or {}, ensure_ascii=False)}`",
                f"- latency_ms: `{result.get('latency_ms')}`",
                "",
            ])
    (output_dir / "answers.md").write_text("\n".join(lines), encoding="utf-8")


def _manifest(source_path: Path) -> dict:
    return {
        "experiment": "finance_terminology_alignment_shadow_v1",
        "source": str(source_path.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "question_ids": [case["question_id"] for case in CASES],
        "prompt_modes": list(PROMPT_MODES),
        "retrieval_calls": 0,
        "jina_calls": 0,
        "judge_calls": 0,
        "langsmith_calls": 0,
        "answer_calls_expected": len(CASES) * len(PROMPT_MODES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_ANSWERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true", help="Validate and freeze inputs without calling the answer model")
    args = parser.parse_args()
    source_path = args.source.resolve()
    manifest = _manifest(source_path)
    state_path = args.output_dir / "state.json"
    if state_path.exists():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if payload.get("manifest") != manifest:
            raise ValueError("Existing checkpoint does not match the frozen source or experiment manifest")
    else:
        payload = {"manifest": manifest, "records": load_frozen_cases(source_path), "complete": False}
    write_outputs(args.output_dir, payload)
    if args.prepare_only:
        print(f"Prepared {len(payload['records'])} frozen cases: {args.output_dir}")
        return

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    os.environ.update({
        "LANGSMITH_TRACING": "false",
        "LANGSMITH_TRACING_V2": "false",
        "LANGCHAIN_TRACING_V2": "false",
    })
    from answer_generator import generate_answer

    for record in payload["records"]:
        if _sha256_text(record["evidence"]) != record["evidence_sha256"]:
            raise ValueError(f"Frozen evidence drift for {record['question_id']}")
        for mode in PROMPT_MODES:
            if (record["results"].get(mode) or {}).get("status") == "ok":
                continue
            started = time.perf_counter()
            try:
                answer, usage = generate_answer(
                    record["question"], record["evidence"], history=[], task_policy="",
                    profile="finance", prompt_mode=mode,
                )
                result = {
                    "status": "ok",
                    "answer": answer,
                    "assessment": assess_answer(answer, record),
                    "usage": usage,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            except Exception as exc:
                result = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            record["results"][mode] = result
            write_outputs(args.output_dir, payload)
            print(f"[{record['question_id']}] {mode}: {result['status']}", flush=True)
    payload["complete"] = all(
        (record["results"].get(mode) or {}).get("status") == "ok"
        for record in payload["records"] for mode in PROMPT_MODES
    )
    write_outputs(args.output_dir, payload)
    print(f"Complete={payload['complete']} report={args.output_dir / 'answers.md'}")


if __name__ == "__main__":
    main()
