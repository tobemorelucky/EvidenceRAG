"""Run the DeepSeek-V4-Flash-only Conversation Understanding v2 shadow."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.conversation_understanding import (  # noqa: E402
    conversation_memory_enabled,
    create_understanding_model,
    understand_conversation,
)
from backend.memory.conversation_state import ConversationState  # noqa: E402


MODEL = "deepseek-v4-flash-ga-260731"
CASES = ROOT / "tests/fixtures/conversation_aware_rag_v2_cases.json"
OUTPUT = ROOT / "reports/conversation_aware_rag_v2_shadow"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    temporary.replace(path)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9%$]+", " ", str(text or "").casefold()).strip()


def standalone_quality(query: str, required_terms: list[str], need_rewrite: bool) -> dict:
    normalized = normalize(query)
    matched = [term for term in required_terms if normalize(term) in normalized]
    retention = len(matched) / len(required_terms) if required_terms else 1.0
    unresolved = bool(need_rewrite and re.search(r"\b(?:it|its|that|this|they|them|those|former|latter)\b", normalized))
    nonempty = bool(normalized)
    return {
        "required_terms": required_terms, "matched_terms": matched,
        "term_retention": retention, "unresolved_reference": unresolved,
        "quality_pass": nonempty and retention >= 0.75 and not unresolved,
    }


class RecordingModel:
    def __init__(self, model):
        self.model = model
        self.raw_content = None

    def invoke(self, messages):
        response = self.model.invoke(messages)
        self.raw_content = response.content
        return response


def run(cases: list[dict], output: Path, limit: int = 0) -> None:
    if conversation_memory_enabled():
        raise ValueError("ENABLE_CONVERSATION_MEMORY must remain false for this shadow")
    if not os.getenv("ARK_API_KEY") or not os.getenv("BASE_URL"):
        raise ValueError("ARK_API_KEY and BASE_URL are required")
    os.environ["CONVERSATION_UNDERSTANDING_MODEL"] = MODEL
    selected_cases = cases[:limit] if limit else cases
    path = output / "results.jsonl"
    saved = {record["id"]: record for record in load_jsonl(path)}
    model = create_understanding_model()
    ordered_ids = [case["id"] for case in selected_cases]
    for index, case in enumerate(selected_cases, 1):
        existing = saved.get(case["id"], {})
        if existing.get("status") == "ok":
            print(f"[{index}/{len(selected_cases)}] {case['id']} cached", flush=True)
            continue
        state = ConversationState(case["id"])
        for role, content in case["history"]:
            state.append(role, content)
        recorder = RecordingModel(model)
        started = time.perf_counter()
        record = {"id": case["id"], "category": case["category"], "user_input": case["user_input"], "expected": case["expected"]}
        try:
            decision, trace = understand_conversation(case["user_input"], state, model=recorder)
            actual = decision.to_dict()
            record.update(
                status="ok", raw_output=recorder.raw_content, understanding=actual, trace=trace,
                need_rag_correct=actual["need_rag"] == case["expected"]["need_rag"],
                need_rewrite_correct=actual["need_rewrite"] == case["expected"]["need_rewrite"],
                standalone_quality=standalone_quality(actual["standalone_query"], case["standalone_must_include"], case["expected"]["need_rewrite"]),
            )
        except Exception as exc:
            record.update(status="invalid_json", raw_output=recorder.raw_content, error_type=type(exc).__name__, error=str(exc)[:500])
        record["latency_ms"] = (time.perf_counter() - started) * 1000
        saved[case["id"]] = record
        atomic_jsonl(path, [saved[item] for item in ordered_ids if item in saved])
        print(f"[{index}/{len(selected_cases)}] {case['id']} {record['status']}", flush=True)


def summarize(cases: list[dict], records: list[dict]) -> dict:
    by_id = {record["id"]: record for record in records}
    if len(by_id) != len(cases) or any(case["id"] not in by_id for case in cases):
        raise ValueError(f"Report requires {len(cases)} complete records")
    valid = [by_id[case["id"]] for case in cases if by_id[case["id"]].get("status") == "ok"]
    categories = {}
    for category in sorted({case["category"] for case in cases}):
        category_cases = [case for case in cases if case["category"] == category]
        subset = [by_id[case["id"]] for case in category_cases]
        legal = [record for record in subset if record.get("status") == "ok"]
        query_subset = [by_id[case["id"]] for case in category_cases if case["expected"]["need_retrieval"]]
        categories[category] = {
            "cases": len(subset), "json_legal": len(legal),
            "need_rag_accuracy": sum(bool(record.get("need_rag_correct")) for record in subset) / len(subset),
            "need_rewrite_accuracy": sum(bool(record.get("need_rewrite_correct")) for record in subset) / len(subset),
            "standalone_query_cases": len(query_subset),
            "standalone_quality_pass_rate": (
                sum(bool((record.get("standalone_quality") or {}).get("quality_pass")) for record in query_subset) / len(query_subset)
                if query_subset else None
            ),
        }
    errors = []
    for case in cases:
        record = by_id[case["id"]]
        reasons = []
        if record.get("status") != "ok": reasons.append("invalid_json")
        if not record.get("need_rag_correct"): reasons.append("need_rag")
        if not record.get("need_rewrite_correct"): reasons.append("need_rewrite")
        if case["expected"]["need_retrieval"] and not (record.get("standalone_quality") or {}).get("quality_pass"):
            reasons.append("standalone_query")
        if reasons:
            errors.append({"id": case["id"], "category": case["category"], "reasons": reasons, "expected": case["expected"], "actual": record.get("understanding"), "raw_output": record.get("raw_output"), "quality": record.get("standalone_quality")})
    usages = [record.get("trace", {}).get("usage", {}) for record in valid]
    query_records = [by_id[case["id"]] for case in cases if case["expected"]["need_retrieval"]]
    return {
        "model": MODEL, "cases": len(cases), "json_legal": len(valid), "json_legal_rate": len(valid) / len(cases),
        "need_rag_accuracy": sum(bool(record.get("need_rag_correct")) for record in records) / len(cases),
        "need_rewrite_accuracy": sum(bool(record.get("need_rewrite_correct")) for record in records) / len(cases),
        "need_retrieval_accuracy": sum(
            bool((record.get("understanding") or {}).get("need_retrieval") == case["expected"]["need_retrieval"])
            for case, record in ((case, by_id[case["id"]]) for case in cases)
        ) / len(cases),
        "need_previous_context_accuracy": sum(
            bool((record.get("understanding") or {}).get("need_previous_context") == case["expected"]["need_previous_context"])
            for case, record in ((case, by_id[case["id"]]) for case in cases)
        ) / len(cases),
        "standalone_query_evaluated_cases": len(query_records),
        "standalone_query_quality_pass_rate": sum(bool((record.get("standalone_quality") or {}).get("quality_pass")) for record in query_records) / len(query_records),
        "standalone_query_mean_term_retention": sum(float((record.get("standalone_quality") or {}).get("term_retention") or 0) for record in query_records) / len(query_records),
        "errors": len(errors), "error_cases": errors, "categories": categories,
        "cost": {
            key: sum(int(usage.get(key) or 0) for usage in usages)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        },
        "latency_ms_per_case": sum(float(record.get("latency_ms") or 0) for record in records) / len(cases),
        "evaluation_note": "Standalone-query quality is a deterministic required-term/self-containment proxy, not an LLM Judge score.",
    }


def write_report(cases: list[dict], output: Path) -> dict:
    records = load_jsonl(output / "results.jsonl")
    summary = summarize(cases, records)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Conversation-aware RAG v2 Shadow", "", "## Result", "",
        f"- Model: `{summary['model']}`", f"- JSON legal: **{summary['json_legal']}/{summary['cases']} ({summary['json_legal_rate']:.1%})**",
        f"- `need_rag` accuracy: **{summary['need_rag_accuracy']:.1%}**",
        f"- `need_rewrite` accuracy: **{summary['need_rewrite_accuracy']:.1%}**",
        f"- `need_retrieval` accuracy: **{summary['need_retrieval_accuracy']:.1%}**",
        f"- `need_previous_context` accuracy: **{summary['need_previous_context_accuracy']:.1%}**",
        f"- Standalone-query quality pass: **{summary['standalone_query_quality_pass_rate']:.1%} ({summary['standalone_query_evaluated_cases']} retrieval-required cases)**",
        f"- Standalone required-term retention: **{summary['standalone_query_mean_term_retention']:.1%}**",
        f"- Errors requiring review: **{summary['errors']}**", "",
        "Standalone-query quality is measured without another model: >=75% human-required term retention and no unresolved pronoun when rewrite is required.", "",
        "## By category", "", "| Category | Cases | JSON | Need RAG | Need rewrite | Query quality |", "|---|---:|---:|---:|---:|---:|"]
    for category, value in summary["categories"].items():
        query_quality = f"{value['standalone_quality_pass_rate']:.1%}" if value["standalone_quality_pass_rate"] is not None else "N/A"
        lines.append(f"| {category} | {value['cases']} | {value['json_legal']} | {value['need_rag_accuracy']:.1%} | {value['need_rewrite_accuracy']:.1%} | {query_quality} |")
    lines += ["", "## Error cases", ""]
    if not summary["error_cases"]:
        lines.append("None.")
    for error in summary["error_cases"]:
        lines += [f"### {error['id']} — {error['category']}", "", f"- Failed checks: {', '.join(error['reasons'])}",
            f"- Expected: `{json.dumps(error['expected'], ensure_ascii=False)}`",
            f"- Actual: `{json.dumps(error['actual'], ensure_ascii=False)}`",
            f"- Query quality: `{json.dumps(error['quality'], ensure_ascii=False)}`", ""]
    lines += ["## Cost", "", f"- Total tokens: **{summary['cost']['total_tokens']:,}**", f"- Mean latency: **{summary['latency_ms_per_case']:.0f} ms/case**", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "report", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="0 runs all 50 cases")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if len(cases) != 50 or len({case["id"] for case in cases}) != 50:
        raise ValueError("Expected exactly 50 unique human-authored scenarios")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"run", "all"}:
        run(cases, args.output_dir, args.limit)
    if args.stage in {"report", "all"}:
        if args.limit:
            raise ValueError("Full report cannot be generated from a limited run")
        print(json.dumps(write_report(cases, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
