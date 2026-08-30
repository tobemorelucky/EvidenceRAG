"""Build an offline evidence-flow audit from frozen RAG Core v2 artifacts.

Gold pages and reference answers are used only by this reporting script.  Nothing
from this module is imported by the online retrieval path.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWERS = ROOT / "reports" / "evidencerag-rag-core-v2-skills-all100-final_answers.jsonl"
DEFAULT_JUDGE = ROOT / "reports" / "evidencerag-rag-core-v2-skills-all100-final_judge.jsonl"
DEFAULT_CORE_JUDGE = ROOT / "reports" / "evidencerag-rag-core-v2-all100-final_judge.jsonl"
DEFAULT_DIAGNOSTIC = ROOT / "reports" / "evidencerag-rag-core-v2-skills-all100-final-evidence-diagnostic.json"
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_JSON = ROOT / "reports" / "rag_core_v3_error_flow_audit.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "rag_core_v3_error_flow_audit.md"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json"
ERROR_STAGES = (
    "candidate_miss",
    "candidate_hit_selection_loss",
    "selected_hit_context_loss",
    "context_hit_answer_wrong",
    "skill_execution_wrong",
    "benchmark_definition_disagreement",
    "generation_variance_or_uncertain",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _by_id(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("financebench_id") or ""): item for item in records}


def _citation_pages(answer: dict[str, Any]) -> list[dict[str, Any]]:
    pages = []
    seen = set()
    for citation in answer.get("citations") or []:
        item = {
            "filename": str(citation.get("filename") or ""),
            "page_number": citation.get("page_number"),
        }
        key = (item["filename"], item["page_number"])
        if key not in seen:
            seen.add(key)
            pages.append(item)
    return pages


def _skill_state(answer: dict[str, Any]) -> tuple[str, bool, bool]:
    trace = answer.get("rag_trace") or {}
    router = trace.get("skill_router") or {}
    skill = str(router.get("selected_skill") or "none")
    applied = skill not in {"", "none"}
    authoritative = applied and int(trace.get("answer_input_tokens") or 0) == 0
    return skill, applied, authoritative


def _classify(
    diagnostic: dict[str, Any],
    judge: dict[str, Any],
    answer: dict[str, Any],
    core_judge: dict[str, Any],
) -> tuple[str, str]:
    correct = int(judge.get("score") or 0) == 1
    candidate_hit = bool(diagnostic.get("candidate_page_hit"))
    selected_hit = bool(diagnostic.get("selected_page_hit"))
    context_hit = bool(diagnostic.get("context_page_hit"))
    skill, applied, authoritative = _skill_state(answer)
    core_score = int(core_judge.get("score") or 0)

    if correct:
        return "generation_variance_or_uncertain", "judge_correct_no_error"
    if not candidate_hit:
        return "candidate_miss", "gold_page_absent_from_candidate_pool"
    if not selected_hit:
        return "candidate_hit_selection_loss", "gold_page_candidate_not_selected"
    if not context_hit:
        return "selected_hit_context_loss", "selected_gold_page_absent_from_answer_context"
    if authoritative and applied:
        return "skill_execution_wrong", f"authoritative_{skill}_answer_judged_incorrect"
    if not applied and core_score == 1:
        return "generation_variance_or_uncertain", "same_retrieval_core_correct_skills_run_incorrect"
    return "context_hit_answer_wrong", "gold_page_in_context_but_answer_judged_incorrect"


def _question_family(question: str) -> str:
    """Coarse labels used only to balance the frozen correct regression fixture."""
    text = question.casefold()
    if any(token in text for token in ("calculate", "what is the change", "what was the change", "ratio", "percent change")):
        return "calculation"
    if any(token in text for token in ("increase or decrease", "higher or lower", "compare", "versus", " vs ")):
        return "multi_period"
    if any(token in text for token in ("why", "what drove", "reason", "describe", "explain")):
        return "narrative"
    if any(token in text for token in ("which", "largest", "smallest", "most", "least")):
        return "selection"
    if any(char.isdigit() for char in question):
        return "table_or_lookup"
    return "lookup"


def _take_evenly(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(records, key=lambda row: row["financebench_id"]):
        buckets[_question_family(item["question"])].append(item)
    preferred = ["lookup", "narrative", "table_or_lookup", "calculation", "multi_period", "selection"]
    chosen = []
    while len(chosen) < limit:
        added = False
        for family in preferred:
            if buckets[family]:
                item = buckets[family].pop(0)
                chosen.append({"financebench_id": item["financebench_id"], "family": family})
                added = True
                if len(chosen) >= limit:
                    break
        if not added:
            break
    return chosen


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    answers = _by_id(_read_jsonl(args.answers))
    judges = _by_id(_read_jsonl(args.judge))
    core_judges = _by_id(_read_jsonl(args.core_judge))
    diagnostic_payload = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    diagnostics = _by_id(diagnostic_payload.get("records") or [])
    dataset = pd.read_csv(args.dataset).fillna("")
    dataset_rows = _by_id(dataset.to_dict("records"))

    records = []
    for financebench_id, dataset_row in dataset_rows.items():
        answer = answers.get(financebench_id, {})
        judge = judges.get(financebench_id, {})
        diagnostic = diagnostics.get(financebench_id, {})
        skill, skill_applied, skill_authoritative = _skill_state(answer)
        error_stage, detail = _classify(
            diagnostic, judge, answer, core_judges.get(financebench_id, {})
        )
        records.append({
            "financebench_id": financebench_id,
            "question": str(dataset_row.get("question") or ""),
            "judge": str(judge.get("verdict") or "missing"),
            "judge_score": int(judge.get("score") or 0),
            "candidate_gold_hit": bool(diagnostic.get("candidate_page_hit")),
            "selected_gold_hit": bool(diagnostic.get("selected_page_hit")),
            "context_gold_hit": bool(diagnostic.get("context_page_hit")),
            "skill_name": skill,
            "skill_applied": skill_applied,
            "skill_authoritative": skill_authoritative,
            "cited_documents_pages": _citation_pages(answer),
            "error_stage": error_stage,
            "error_detail": detail,
        })

    records.sort(key=lambda item: item["financebench_id"])
    counts = Counter(item["error_stage"] for item in records)
    incorrect_counts = Counter(item["error_stage"] for item in records if not item["judge_score"])
    all_stage_counts = {stage: counts[stage] for stage in ERROR_STAGES}
    incorrect_stage_counts = {stage: incorrect_counts[stage] for stage in ERROR_STAGES}
    fixture = {
        "source_commit": "7cfd9cd5acf41fa36fc10e3d8f22326f3748a38e",
        "selection_loss10": [
            {"financebench_id": item["financebench_id"]}
            for item in records if item["error_stage"] == "candidate_hit_selection_loss"
        ][:10],
        "candidate_miss10": [
            {"financebench_id": item["financebench_id"]}
            for item in records if item["error_stage"] == "candidate_miss"
        ][:10],
        "correct_regression10": _take_evenly(
            [item for item in records if item["judge_score"] == 1], 10
        ),
    }
    return {
        "source": {
            "answers": str(args.answers),
            "judge": str(args.judge),
            "diagnostic": str(args.diagnostic),
            "dataset": str(args.dataset),
        },
        "questions": len(records),
        "all_stage_counts": all_stage_counts,
        "incorrect_stage_counts": incorrect_stage_counts,
        "records": records,
        "fixture": fixture,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# RAG Core v3 Error Flow Audit",
        "",
        "> Offline evaluation only. Gold pages and reference data are never used by online retrieval or ranking.",
        "",
        f"- Questions: {payload['questions']}",
        "- Frozen source commit: `7cfd9cd5acf41fa36fc10e3d8f22326f3748a38e`",
        "",
        "## Incorrect-question flow",
        "",
        "| Stage | Count |",
        "|---|---:|",
    ]
    for name, count in payload["incorrect_stage_counts"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend([
        "",
        "## All questions",
        "",
        "Correct questions remain in `generation_variance_or_uncertain` with detail `judge_correct_no_error`; this avoids inventing an error cause.",
        "",
        "| ID | Judge | Candidate | Selected | Context | Skill | Error stage | Citations | Question |",
        "|---|---|---:|---:|---:|---|---|---|---|",
    ])
    for item in payload["records"]:
        citations = ", ".join(
            f"{entry['filename']}#p{entry['page_number']}"
            for entry in item["cited_documents_pages"]
        )
        question = item["question"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item['financebench_id']} | {item['judge']} | "
            f"{int(item['candidate_gold_hit'])} | {int(item['selected_gold_hit'])} | "
            f"{int(item['context_gold_hit'])} | {item['skill_name']} | "
            f"`{item['error_stage']}` | {citations} | {question} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--core-judge", type=Path, default=DEFAULT_CORE_JUDGE)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--fixture-output", type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args()

    payload = build_audit(args)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.fixture_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_output.write_text(_markdown(payload), encoding="utf-8")
    args.fixture_output.write_text(json.dumps(payload["fixture"], indent=2), encoding="utf-8")
    print(json.dumps({
        "questions": payload["questions"],
        "incorrect_stage_counts": payload["incorrect_stage_counts"],
        "json": str(args.json_output),
        "markdown": str(args.markdown_output),
        "fixture": str(args.fixture_output),
    }, indent=2))


if __name__ == "__main__":
    main()
