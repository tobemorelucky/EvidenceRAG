"""Generate a complete offline Markdown analysis for the frozen final100 run.

No retrieval, reranking, answer generation, Judge, LangSmith, or external API is
imported or called. The FinanceBench CSV is read only because the frozen run
artifacts do not duplicate reference answers or question reasoning metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports/final100"
DEFAULT_DATASET = ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_OUTPUT = DEFAULT_INPUT / "final100_answer_analysis.md"
DEFAULT_SUMMARY = DEFAULT_INPUT / "final100_answer_analysis_summary.json"
DEFAULT_ATTRIBUTION = ROOT / "reports/answer_failure_attribution_v2/failure_cases.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["financebench_id"]: row for row in csv.DictReader(stream)}


def records_by_id(records: list[dict], source: str) -> dict[str, dict]:
    output = {str(record.get("question_id") or ""): record for record in records}
    output.pop("", None)
    if len(output) != len(records):
        raise ValueError(f"Duplicate or missing question_id in {source}")
    return output


def load_attribution(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    payload = load_json(path)
    rows = payload if isinstance(payload, list) else payload.get("failure_cases", [])
    return {str(row.get("id") or row.get("question_id")): row for row in rows}


def require_inputs(input_dir: Path, dataset_path: Path) -> dict[str, Path]:
    paths = {
        "summary": input_dir / "final_summary.json",
        "answers": input_dir / "answers.jsonl",
        "judges": input_dir / "judge_results.jsonl",
        "retrieval": input_dir / "retrieval_snapshot.json",
        "jina": input_dir / "jina_results.json",
        "dataset": dataset_path,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required offline inputs: {missing}")
    return paths


def load_inputs(input_dir: Path, dataset_path: Path, attribution_path: Path | None) -> dict:
    paths = require_inputs(input_dir, dataset_path)
    summary = load_json(paths["summary"])
    answers_list = load_jsonl(paths["answers"])
    judges_list = load_jsonl(paths["judges"])
    retrieval_payload = load_json(paths["retrieval"])
    jina_payload = load_json(paths["jina"])
    retrieval_list = list(retrieval_payload.get("records") or [])
    jina_list = list(jina_payload.get("records") or [])
    dataset = load_dataset(paths["dataset"])
    answers = records_by_id(answers_list, "answers.jsonl")
    judges = records_by_id(judges_list, "judge_results.jsonl")
    retrieval = records_by_id(retrieval_list, "retrieval_snapshot.json")
    jina = records_by_id(jina_list, "jina_results.json")
    expected = int(summary.get("questions") or 0)
    counts = {"answers": len(answers), "judges": len(judges), "retrieval": len(retrieval), "jina": len(jina)}
    if expected != 100 or any(value != expected for value in counts.values()):
        raise ValueError(f"Expected complete final100 artifacts; summary={expected}, records={counts}")
    ids = set(answers)
    for name, values in (("judges", judges), ("retrieval", retrieval), ("jina", jina), ("dataset", dataset)):
        missing = ids - set(values)
        if missing:
            raise ValueError(f"{name} missing {len(missing)} final100 IDs")
    order = [str(record["question_id"]) for record in retrieval_list]
    return {
        "summary": summary,
        "answers": answers,
        "judges": judges,
        "retrieval": retrieval,
        "jina": jina,
        "dataset": dataset,
        "attribution": load_attribution(attribution_path),
        "order": order,
        "paths": paths,
    }


def bool_label(value: bool) -> str:
    return "Yes" if value else "No"


def markdown_cell(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def selected_evidence_table(answer_record: dict) -> list[str]:
    documents = list(answer_record.get("context_documents") or [])
    if not documents:
        return ["No selected context documents were recorded."]
    lines = [
        "| Rank | Document | Page | Chunk ID | Level | Table ID | Score |",
        "|---:|---|---:|---|---|---|---:|",
    ]
    for rank, document in enumerate(documents, 1):
        score = document.get("score", document.get("rrf_score"))
        score_text = f"{float(score):.6f}" if isinstance(score, (int, float)) else markdown_cell(score)
        lines.append(
            f"| {rank} | {markdown_cell(document.get('filename'))} | {markdown_cell(document.get('page_number'))} | "
            f"{markdown_cell(document.get('chunk_id') or document.get('id'))} | "
            f"{markdown_cell(document.get('chunk_level'))} | {markdown_cell(document.get('table_id'))} | {score_text} |"
        )
    return lines


def question_flags(dataset_row: dict) -> tuple[bool, bool]:
    reasoning = str(dataset_row.get("question_reasoning") or "")
    is_calculation = "Numerical reasoning" in reasoning
    is_reasoning = "Logical reasoning" in reasoning
    return is_calculation, is_reasoning


def stage_failure_category(answer: dict, retrieval: dict, jina: dict) -> str:
    candidate_hit = bool((retrieval.get("retrieval_metrics") or {}).get("candidate_gold_page_hit"))
    jina_hit = bool((jina.get("rerank_metrics") or {}).get("context_hit"))
    context_hit = bool((answer.get("context_metrics") or {}).get("context_hit"))
    if not candidate_hit:
        return "retrieval_miss"
    if not jina_hit:
        return "rerank_miss"
    if not context_hit:
        return "context_loss"
    return "answer_failure"


def attributed_failure_category(
    question_id: str,
    answer: dict,
    retrieval: dict,
    jina: dict,
    attribution: dict[str, dict],
) -> tuple[str, str]:
    for source in (answer, retrieval, jina):
        for key in ("answer_failure", "failure_type", "failure_category", "attribution"):
            if source.get(key):
                return str(source[key]), f"existing field `{key}`"
    row = attribution.get(question_id)
    if row:
        name = str(row.get("category_name") or row.get("category") or "Other")
        return name, "answer_failure_attribution_v2"
    return stage_failure_category(answer, retrieval, jina), "final100 funnel fields"


def statistics_bucket(category: str) -> str:
    normalized = category.casefold()
    if "retrieval" in normalized:
        return "Retrieval failure"
    if "rerank" in normalized or "context_loss" in normalized or "evidence" in normalized:
        return "Evidence selection failure"
    if "calculation" in normalized:
        return "Calculation error"
    if "reasoning" in normalized or "period" in normalized or "entity" in normalized or "metric" in normalized:
        return "Reasoning error"
    if "refusal" in normalized or "uncertainty" in normalized:
        return "Refusal"
    return "Other"


def build_case(
    question_id: str,
    inputs: dict,
) -> dict:
    answer = inputs["answers"][question_id]
    judge_record = inputs["judges"][question_id]
    retrieval = inputs["retrieval"][question_id]
    jina = inputs["jina"][question_id]
    dataset = inputs["dataset"][question_id]
    judge = dict(judge_record.get("judge") or {})
    calculation, reasoning = question_flags(dataset)
    failure_category, attribution_source = (None, None)
    if judge.get("score") != 1:
        failure_category, attribution_source = attributed_failure_category(
            question_id, answer, retrieval, jina, inputs["attribution"],
        )
    return {
        "question_id": question_id,
        "question": str(answer.get("question") or dataset.get("question") or retrieval.get("question") or ""),
        "reference_answer": str(dataset.get("answer") or ""),
        "model_answer": str(answer.get("answer") or ""),
        "answer_record": answer,
        "judge": judge,
        "question_type": str(dataset.get("question_type") or "unknown"),
        "question_reasoning": str(dataset.get("question_reasoning") or "unknown"),
        "is_calculation": calculation,
        "is_reasoning": reasoning,
        "failure_category": failure_category,
        "attribution_source": attribution_source,
    }


def render_case(case: dict, *, incorrect: bool) -> list[str]:
    heading = "Retrieved Evidence" if incorrect else "Evidence"
    lines = [
        f"## {case['question_id']}", "",
        "### Question", "", case["question"], "",
        "### Reference Answer", "", case["reference_answer"], "",
        "### Model Answer", "", case["model_answer"], "",
        f"### {heading}", "",
        *selected_evidence_table(case["answer_record"]), "",
    ]
    if incorrect:
        lines.extend([
            "### Failure Category", "",
            f"- Category: **{case['failure_category']}**",
            f"- Source: {case['attribution_source']}", "",
        ])
    lines.extend([
        "### Evaluation", "",
        f"- Judge result: **{case['judge'].get('verdict', 'unknown')}** (score={case['judge'].get('score')})",
        f"- Judge reason: {case['judge'].get('reason', '')}",
        f"- Question type: `{case['question_type']}`",
        f"- Question reasoning metadata: `{case['question_reasoning']}`",
        f"- Calculation: **{bool_label(case['is_calculation'])}**",
        f"- Reasoning: **{bool_label(case['is_reasoning'])}**", "",
    ])
    return lines


def build_report(inputs: dict) -> tuple[str, dict]:
    cases = [build_case(question_id, inputs) for question_id in inputs["order"]]
    correct = [case for case in cases if case["judge"].get("score") == 1]
    incorrect = [case for case in cases if case["judge"].get("score") != 1]
    summary = inputs["summary"]
    strict = dict(summary.get("strict_judge") or {})
    funnel = dict(summary.get("funnel") or {})
    failure_counts = Counter(statistics_bucket(str(case["failure_category"])) for case in incorrect)
    lines = [
        "# FinanceBench Final100 Experiment Analysis", "",
        "## 1. Overall Result", "",
        f"- Total questions: **{summary.get('questions')}**",
        f"- Strict Judge: **{strict.get('correct')}/{strict.get('total')} ({float(strict.get('accuracy') or 0):.2%})**",
        f"- Candidate hit: **{float(funnel.get('rrf_candidate_hit') or 0):.2%}**",
        f"- Context hit: **{float(funnel.get('context_hit') or 0):.2%}**",
        f"- Retrieval miss: **{funnel.get('retrieval_miss')}**",
        f"- Rerank miss: **{funnel.get('rerank_miss')}**",
        f"- Context loss: **{funnel.get('context_loss')}**",
        f"- Answer failure: **{funnel.get('answer_failure')}**", "",
        "# 2. Correct Answers Analysis", "",
    ]
    for case in correct:
        lines.extend(render_case(case, incorrect=False))
    lines.extend(["# 3. Incorrect Answers Analysis", ""])
    for case in incorrect:
        lines.extend(render_case(case, incorrect=True))
    lines.extend([
        "# 4. Failure Statistics", "",
        "| Failure class | Count | Share of incorrect answers |",
        "|---|---:|---:|",
    ])
    for category in (
        "Retrieval failure", "Evidence selection failure", "Calculation error",
        "Reasoning error", "Refusal", "Other",
    ):
        count = failure_counts.get(category, 0)
        lines.append(f"| {category} | {count} | {count / len(incorrect):.2%} |")
    lines.extend([
        "", "# 5. Key Findings", "",
        "- 当前系统的主要瓶颈已经不只是宽召回；大量错误发生在相关 evidence 已进入 context 之后的 answer utilization 阶段。",
        "- Query Rewrite + Jina80 将 candidate hit 从 85% 提升到 94%，将 context hit 从 73% 提升到 81%，但 Strict Judge 仍为 68%。",
        "- 后续 Evidence Focus Prompt shadow 实验修复了部分 calculation、reasoning direction 和 unnecessary refusal 类型的 answer failure。",
        "- Semantic Gating 比全量启用 Evidence Focus 更安全，但该结论来自后续 shadow 结果，不会改写本次 final100 基线分数。", "",
    ])
    report = "\n".join(lines)
    validation = validate_report(report, cases, correct, incorrect)
    output_summary = {
        "questions": len(cases),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "strict_accuracy": len(correct) / len(cases),
        "funnel": funnel,
        "failure_statistics": {category: failure_counts.get(category, 0) for category in (
            "Retrieval failure", "Evidence selection failure", "Calculation error",
            "Reasoning error", "Refusal", "Other",
        )},
        "validation": validation,
        "external_calls": 0,
    }
    return report, output_summary


def validate_report(report: str, cases: list[dict], correct: list[dict], incorrect: list[dict]) -> dict:
    if len(cases) != 100 or len(correct) + len(incorrect) != 100:
        raise ValueError("Report must contain exactly 100 classified questions")
    ids = [case["question_id"] for case in cases]
    if len(set(ids)) != 100:
        raise ValueError("Report contains duplicate question IDs")
    missing_sections = []
    for case in cases:
        marker = f"## {case['question_id']}\n"
        if report.count(marker) != 1:
            missing_sections.append(case["question_id"])
        if not case["question"].strip() or not case["reference_answer"].strip() or not case["model_answer"].strip():
            missing_sections.append(case["question_id"])
    if missing_sections:
        raise ValueError(f"Incomplete question sections: {sorted(set(missing_sections))}")
    return {
        "all_100_questions_present_once": True,
        "correct_plus_incorrect_equals_100": True,
        "question_reference_model_answer_present": True,
    }


def write_outputs(report: str, summary: dict, output_path: Path, summary_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--attribution", type=Path, default=DEFAULT_ATTRIBUTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    inputs = load_inputs(args.input_dir.resolve(), args.dataset.resolve(), args.attribution.resolve())
    report, summary = build_report(inputs)
    write_outputs(report, summary, args.output.resolve(), args.summary_output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

