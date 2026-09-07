"""Evaluate Conversation Understanding v3 on the frozen 50-scenario corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.conversation_understanding import conversation_memory_enabled  # noqa: E402
from backend.conversation_understanding_v3 import create_understanding_model_v3, understand_conversation_v3  # noqa: E402
from backend.memory.conversation_state import ConversationState  # noqa: E402
from scripts.run_conversation_aware_rag_v2_shadow import standalone_quality  # noqa: E402


MODEL = "deepseek-v4-flash-ga-260731"
CASES = ROOT / "tests/fixtures/conversation_aware_rag_v2_cases.json"
LABELS = ROOT / "tests/fixtures/conversation_aware_rag_v3_labels.json"
V2_SUMMARY = ROOT / "reports/conversation_aware_rag_v2_shadow/summary.json"
V2_RESULTS = ROOT / "reports/conversation_aware_rag_v2_shadow/results.jsonl"
OUTPUT = ROOT / "reports/conversation_aware_rag_v3_schema_shadow"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    temporary.replace(path)


def load_cases() -> list[dict]:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    ids = {case["id"] for case in cases}
    if len(cases) != 50 or len(ids) != 50 or set(labels) != ids:
        raise ValueError("v3 must use exactly the original 50 frozen scenarios with 50 matching labels")
    enriched = []
    for case in cases:
        depends, resolution, scope = labels[case["id"]]
        enriched.append({
            **case,
            "expected_v3": {
                "need_rag": case["expected"]["need_rag"],
                "need_retrieval": case["expected"]["need_retrieval"],
                "depends_on_history": depends,
                "query_resolution_needed": resolution,
                "required_memory_scope": scope,
            },
        })
    return enriched


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
    selected = cases[:limit] if limit else cases
    path = output / "results.jsonl"
    saved = {record["id"]: record for record in load_jsonl(path)}
    model = create_understanding_model_v3()
    ordered_ids = [case["id"] for case in selected]
    for index, case in enumerate(selected, 1):
        existing = saved.get(case["id"], {})
        if existing.get("status") == "ok":
            print(f"[{index}/{len(selected)}] {case['id']} cached", flush=True)
            continue
        state = ConversationState(case["id"])
        for role, content in case["history"]:
            state.append(role, content)
        recorder = RecordingModel(model)
        started = time.perf_counter()
        record = {"id": case["id"], "category": case["category"], "user_input": case["user_input"], "expected": case["expected_v3"]}
        try:
            decision, trace = understand_conversation_v3(case["user_input"], state, recorder)
            actual = decision.to_dict()
            quality = standalone_quality(actual["standalone_query"], case["standalone_must_include"], case["expected_v3"]["query_resolution_needed"])
            record.update(
                status="ok", raw_output=recorder.raw_content, understanding=actual, trace=trace,
                depends_on_history_correct=actual["depends_on_history"] == case["expected_v3"]["depends_on_history"],
                query_resolution_correct=actual["query_resolution_needed"] == case["expected_v3"]["query_resolution_needed"],
                need_rag_correct=actual["need_rag"] == case["expected_v3"]["need_rag"],
                need_retrieval_correct=actual["need_retrieval"] == case["expected_v3"]["need_retrieval"],
                standalone_quality=quality,
            )
        except Exception as exc:
            record.update(status="invalid_json", raw_output=recorder.raw_content, error_type=type(exc).__name__, error=str(exc)[:500])
        record["latency_ms"] = (time.perf_counter() - started) * 1000
        saved[case["id"]] = record
        atomic_jsonl(path, [saved[item] for item in ordered_ids if item in saved])
        print(f"[{index}/{len(selected)}] {case['id']} {record['status']}", flush=True)


def accuracy(records: list[dict], key: str) -> float:
    return sum(bool(record.get(key)) for record in records) / len(records)


def summarize(cases: list[dict], records: list[dict], v2: dict) -> dict:
    by_id = {record["id"]: record for record in records}
    if len(by_id) != len(cases) or any(case["id"] not in by_id for case in cases):
        raise ValueError("Report requires all 50 frozen scenarios")
    ordered = [by_id[case["id"]] for case in cases]
    valid = [record for record in ordered if record.get("status") == "ok"]
    query_records = [by_id[case["id"]] for case in cases if case["expected_v3"]["need_retrieval"]]
    categories = {}
    for category in sorted({case["category"] for case in cases}):
        subset = [by_id[case["id"]] for case in cases if case["category"] == category]
        categories[category] = {
            "cases": len(subset),
            "depends_on_history_accuracy": accuracy(subset, "depends_on_history_correct"),
            "query_resolution_accuracy": accuracy(subset, "query_resolution_correct"),
        }
    errors = []
    for case in cases:
        record = by_id[case["id"]]
        reasons = []
        if record.get("status") != "ok": reasons.append("invalid_json")
        if not record.get("need_rag_correct"): reasons.append("need_rag")
        if not record.get("need_retrieval_correct"): reasons.append("need_retrieval")
        if not record.get("depends_on_history_correct"): reasons.append("depends_on_history")
        if not record.get("query_resolution_correct"): reasons.append("query_resolution_needed")
        if case["expected_v3"]["need_retrieval"] and not (record.get("standalone_quality") or {}).get("quality_pass"):
            reasons.append("standalone_query")
        if reasons:
            errors.append({"id": case["id"], "category": case["category"], "reasons": reasons, "expected": case["expected_v3"], "actual": record.get("understanding"), "quality": record.get("standalone_quality")})
    usages = [record.get("trace", {}).get("usage", {}) for record in valid]
    current = {
        "json_legal_rate": len(valid) / len(cases),
        "depends_on_history_accuracy": accuracy(ordered, "depends_on_history_correct"),
        "query_resolution_accuracy": accuracy(ordered, "query_resolution_correct"),
        "standalone_query_quality_pass_rate": sum(bool((record.get("standalone_quality") or {}).get("quality_pass")) for record in query_records) / len(query_records),
        "standalone_query_mean_term_retention": sum(float((record.get("standalone_quality") or {}).get("term_retention") or 0) for record in query_records) / len(query_records),
    }
    return {
        "model": MODEL, "cases": len(cases), **current,
        "need_rag_accuracy": accuracy(ordered, "need_rag_correct"),
        "need_retrieval_accuracy": accuracy(ordered, "need_retrieval_correct"),
        "standalone_query_evaluated_cases": len(query_records),
        "categories": categories, "errors": len(errors), "error_cases": errors,
        "v2_comparison": {
            "json_legal_rate": {"v2": v2["json_legal_rate"], "v3": current["json_legal_rate"], "delta": current["json_legal_rate"] - v2["json_legal_rate"]},
            "need_rag": {"v2": v2["need_rag_accuracy"], "v3": accuracy(ordered, "need_rag_correct"), "delta": accuracy(ordered, "need_rag_correct") - v2["need_rag_accuracy"]},
            "need_retrieval": {"v2": v2["need_retrieval_accuracy"], "v3": accuracy(ordered, "need_retrieval_correct"), "delta": accuracy(ordered, "need_retrieval_correct") - v2["need_retrieval_accuracy"]},
            "history_dependency": {"v2_field": "need_previous_context", "v2": v2["need_previous_context_accuracy"], "v3_field": "depends_on_history", "v3": current["depends_on_history_accuracy"], "delta": current["depends_on_history_accuracy"] - v2["need_previous_context_accuracy"]},
            "query_resolution": {"v2_field": "need_rewrite", "v2": v2["need_rewrite_accuracy"], "v3_field": "query_resolution_needed", "v3": current["query_resolution_accuracy"], "delta": current["query_resolution_accuracy"] - v2["need_rewrite_accuracy"], "note": "Labels were redefined to measure utterance resolution already performed, not whether another rewrite call is needed."},
            "standalone_query_quality": {"v2": v2["standalone_query_quality_pass_rate"], "v3": current["standalone_query_quality_pass_rate"], "delta": current["standalone_query_quality_pass_rate"] - v2["standalone_query_quality_pass_rate"]},
        },
        "cost": {key: sum(int(usage.get(key) or 0) for usage in usages) for key in ("input_tokens", "output_tokens", "total_tokens")},
        "latency_ms_per_case": sum(float(record.get("latency_ms") or 0) for record in ordered) / len(ordered),
    }


def write_report(cases: list[dict], output: Path) -> dict:
    v2 = json.loads(V2_SUMMARY.read_text(encoding="utf-8"))
    records = load_jsonl(output / "results.jsonl")
    for case in cases:
        record = next(item for item in records if item["id"] == case["id"])
        if record.get("status") == "ok":
            record["standalone_quality"] = standalone_quality(
                record["understanding"]["standalone_query"], case["standalone_must_include"],
                case["expected_v3"]["query_resolution_needed"],
            )
    atomic_jsonl(output / "results.jsonl", records)
    v2_records = {record["id"]: record for record in load_jsonl(V2_RESULTS)}
    v2_query_cases = [case for case in cases if case["expected_v3"]["need_retrieval"]]
    v2_query_quality = sum(
        standalone_quality(
            v2_records[case["id"]]["understanding"]["standalone_query"],
            case["standalone_must_include"], True,
        )["quality_pass"]
        for case in v2_query_cases
    ) / len(v2_query_cases)
    v2 = {**v2, "standalone_query_quality_pass_rate": v2_query_quality}
    summary = summarize(cases, records, v2)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = summary["v2_comparison"]
    lines = ["# Conversation-aware RAG v3 Schema Shadow", "", "## Result", "",
        f"- JSON legal rate: **{summary['json_legal_rate']:.1%}**",
        f"- `depends_on_history` accuracy: **{summary['depends_on_history_accuracy']:.1%}**",
        f"- `query_resolution_needed` accuracy: **{summary['query_resolution_accuracy']:.1%}**",
        f"- Standalone-query quality: **{summary['standalone_query_quality_pass_rate']:.1%}**",
        f"- Required-term retention: **{summary['standalone_query_mean_term_retention']:.1%}**", "",
        "## v2 → v3", "", "| Measure | v2 | v3 | Delta |", "|---|---:|---:|---:|",
        f"| JSON legal | {comparison['json_legal_rate']['v2']:.1%} | {comparison['json_legal_rate']['v3']:.1%} | {comparison['json_legal_rate']['delta']:+.1%} |",
        f"| Need RAG | {comparison['need_rag']['v2']:.1%} | {comparison['need_rag']['v3']:.1%} | {comparison['need_rag']['delta']:+.1%} |",
        f"| Need retrieval | {comparison['need_retrieval']['v2']:.1%} | {comparison['need_retrieval']['v3']:.1%} | {comparison['need_retrieval']['delta']:+.1%} |",
        f"| History dependency | {comparison['history_dependency']['v2']:.1%} | {comparison['history_dependency']['v3']:.1%} | {comparison['history_dependency']['delta']:+.1%} |",
        f"| Query resolution | {comparison['query_resolution']['v2']:.1%} | {comparison['query_resolution']['v3']:.1%} | {comparison['query_resolution']['delta']:+.1%} |",
        f"| Standalone query | {comparison['standalone_query_quality']['v2']:.1%} | {comparison['standalone_query_quality']['v3']:.1%} | {comparison['standalone_query_quality']['delta']:+.1%} |", "",
        "The query-resolution comparison uses redefined v3 labels. It measures whether history resolution was required and already performed, rather than whether an additional rewrite call remains necessary.", "",
        "## By category", "", "| Category | Cases | History dependency | Query resolution |", "|---|---:|---:|---:|"]
    for category, value in summary["categories"].items():
        lines.append(f"| {category} | {value['cases']} | {value['depends_on_history_accuracy']:.1%} | {value['query_resolution_accuracy']:.1%} |")
    lines += ["", "## Errors", ""]
    if not summary["error_cases"]:
        lines.append("None.")
    for error in summary["error_cases"]:
        lines += [f"### {error['id']} — {error['category']}", "", f"- Checks: {', '.join(error['reasons'])}", f"- Expected: `{json.dumps(error['expected'], ensure_ascii=False)}`", f"- Actual: `{json.dumps(error['actual'], ensure_ascii=False)}`", ""]
    lines += ["## Cost", "", f"- Total tokens: **{summary['cost']['total_tokens']:,}**", f"- Mean latency: **{summary['latency_ms_per_case']:.0f} ms/case**", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "report", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    cases = load_cases()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"run", "all"}:
        run(cases, args.output_dir, args.limit)
    if args.stage in {"report", "all"}:
        if args.limit:
            raise ValueError("Full report cannot be generated from a limited run")
        print(json.dumps(write_report(cases, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
