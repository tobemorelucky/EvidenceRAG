"""Offline contract evaluation for 20 human-authored multi-turn scenarios."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.conversation_understanding import understand_conversation  # noqa: E402
from backend.memory.context_builder import build_answer_context  # noqa: E402
from backend.memory.conversation_state import ConversationState  # noqa: E402


CASES = ROOT / "tests/fixtures/conversation_aware_rag_v1_cases.json"
OUTPUT = ROOT / "reports/conversation_aware_rag_v1_shadow"


class FrozenDecisionModel:
    """Returns the human-authored expected JSON; it is not an intent heuristic."""

    def __init__(self, decision: dict):
        self.decision = decision
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return SimpleNamespace(content=json.dumps(self.decision), usage_metadata={})


def evaluate_cases(cases: list[dict]) -> list[dict]:
    records = []
    for case in cases:
        state = ConversationState(case["name"])
        for role, content in case["history"]:
            state.append(role, content)
        model = FrozenDecisionModel(case["decision"])
        decision, trace = understand_conversation(case["user_input"], state, model=model)
        actual = decision.to_dict()
        answer_context = build_answer_context(state, case["user_input"], "[frozen test evidence]", actual)
        passed = actual == case["decision"] and model.calls == 1
        records.append({
            "name": case["name"], "user_input": case["user_input"], "expected": case["decision"],
            "actual": actual, "passed": passed, "understanding_trace": trace,
            "answer_history_turns": len(answer_context["conversation_history"]),
            "evidence_preserved": answer_context["evidence"] == "[frozen test evidence]",
        })
    return records


def main() -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if len(cases) != 20:
        raise ValueError(f"Expected 20 human-authored cases, found {len(cases)}")
    records = evaluate_cases(cases)
    summary = {
        "cases": len(records), "passed": sum(record["passed"] for record in records),
        "failed": sum(not record["passed"] for record in records), "external_model_calls": 0,
        "scope": "schema, first/multi-turn routing contract, history inclusion, and context composition; not live LLM quality",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Conversation-aware RAG v1 Shadow", "", "## Offline result", "",
        f"- Human-authored multi-turn scenarios: **{summary['cases']}**",
        f"- Contract passes: **{summary['passed']}/{summary['cases']}**",
        "- External LLM/Jina/retrieval calls: **0**", "",
        "These tests validate orchestration contracts with frozen decisions. They do not estimate live conversation-understanding accuracy.", "",
        "| Scenario | Need RAG | Retrieval | Rewrite | Previous context | Result |", "|---|---:|---:|---:|---:|---:|"]
    for record in records:
        value = record["actual"]
        lines.append(f"| {record['name']} | {value['need_rag']} | {value['need_retrieval']} | {value['need_rewrite']} | {value['need_previous_context']} | {'pass' if record['passed'] else 'fail'} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
