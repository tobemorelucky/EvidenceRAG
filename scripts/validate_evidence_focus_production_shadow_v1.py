"""Offline five-question routing/trace smoke for production-shadow wiring."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from answer_generator import resolve_answer_prompt_route  # noqa: E402


OUTPUT = ROOT / "reports/evidence_focus_production_shadow_v1"
CASES = [
    ("calculation", "Calculate the operating cash flow ratio for FY2022.", "finance", "finance_evidence_focus_v1"),
    ("trend", "Did operating margin improve from FY2021 to FY2022?", "finance", "finance_evidence_focus_v1"),
    ("lookup", "What was revenue in FY2022?", "finance", "clean_baseline_v1"),
    ("chinese_calculation", "请计算 FY2022 的流动比率。", "finance", "finance_evidence_focus_v1"),
    ("non_finance", "Compare Python list and tuple syntax.", "general", "baseline"),
]


def main() -> None:
    os.environ["FINANCE_EVIDENCE_FOCUS_ROUTER_ENABLED"] = "true"
    results = []
    for name, question, profile, expected in CASES:
        trace = resolve_answer_prompt_route(question, profile, "baseline")
        results.append({
            "case": name,
            "question": question,
            "profile": profile,
            "expected_prompt": expected,
            "trace": trace,
            "passed": trace["selected_prompt"] == expected,
        })
    payload = {"cases": len(results), "passed": sum(row["passed"] for row in results), "results": results}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "manual_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    lines = [
        "# Evidence Focus Production Shadow v1", "",
        "五个问题仅验证线上 Prompt 路由与 trace，不执行 Retrieval、Jina 或 LLM。", "",
    ]
    for row in results:
        lines.extend([
            f"## {row['case']}", "",
            f"- Question: {row['question']}",
            f"- Profile: `{row['profile']}`",
            f"- Selected: `{row['trace']['selected_prompt']}`",
            f"- Reason: `{row['trace']['router_reason']}`",
            f"- Passed: `{row['passed']}`", "",
        ])
    (OUTPUT / "manual_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

