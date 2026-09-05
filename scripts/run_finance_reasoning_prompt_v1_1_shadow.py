"""Compare three answer prompts on the frozen answer-failure audit contexts.

The completed baseline answers are reused from the Jina all100 state.  Only the two
finance prompt variants generate new answers.  Retrieval, reranking, Judge, and
LangSmith are never called.
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
CONFIG = ROOT / "configs/experiments/finance_reasoning_prompt_v1_1.json"
DEFAULT_OUTPUT = ROOT / "reports/finance_reasoning_prompt_v1_1_shadow15"
REFUSAL_PATTERNS = (
    "cannot determine",
    "cannot calculate",
    "can't determine",
    "can't calculate",
    "insufficient information",
    "insufficient evidence",
    "not enough information",
    "unable to determine",
    "unable to calculate",
    "not meaningful",
    "无法计算",
    "无法确定",
    "无法判断",
    "信息不足",
    "证据不足",
    "缺少",
)
STOPWORDS = {
    "about", "after", "answer", "based", "because", "before", "being", "between",
    "company", "data", "during", "from", "have", "million", "question", "states",
    "than", "that", "their", "there", "these", "this", "those", "using", "were",
    "what", "which", "while", "with", "would", "year",
}
DIRECTIONS = {
    "increase": ("increase", "increased", "improving", "accelerate", "accelerated"),
    "decrease": ("decrease", "decreased", "deteriorat", "decelerate", "decelerated"),
}
CRITICAL_FACT_TERMS = {"negative", "positive", "zero"}


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_numbers(text: str) -> set[str]:
    values = set()
    for raw in re.findall(r"(?<![A-Za-z])\(?-?\$?\d[\d,]*(?:\.\d+)?%?", text or ""):
        value = raw.replace("$", "").replace(",", "").replace("(", "-").replace(")", "")
        value = value.rstrip("%")
        try:
            number = float(value)
        except ValueError:
            continue
        if number.is_integer() and 1900 <= number <= 2100:
            continue
        values.add(f"{number:.8f}".rstrip("0").rstrip("."))
    return values


def content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z&'-]{3,}", (text or "").lower())
        if token not in STOPWORDS
    }


def leading_polarity(text: str) -> str | None:
    beginning = re.sub(r"[*_#]", "", (text or "").strip().lower())[:180]
    yes = re.search(r"\byes\b", beginning)
    no = re.search(r"\bno\b|\bnot\b", beginning)
    if yes and (not no or yes.start() < no.start()):
        return "yes"
    if no:
        return "no"
    return None


def expected_polarity(reference: str) -> str | None:
    beginning = (reference or "").strip().lower()
    if beginning.startswith("yes"):
        return "yes"
    if beginning.startswith("no"):
        return "no"
    return None


def expected_direction(reference: str) -> str | None:
    lowered = (reference or "").lower()
    for direction, variants in DIRECTIONS.items():
        if any(variant in lowered for variant in variants):
            return direction
    return None


def direction_in_conclusion(answer: str) -> str | None:
    beginning = (answer or "").lower()[:300]
    found = []
    for direction, variants in DIRECTIONS.items():
        positions = [beginning.find(value) for value in variants if value in beginning]
        if positions:
            found.append((min(positions), direction))
    return min(found)[1] if found else None


def selection_choice(text: str) -> str | None:
    lowered = (text or "").lower()[:500]
    if "most" not in lowered:
        return None
    from_match = re.search(r"\bmost\b.{0,80}\bfrom\s+([a-z&]+(?:\s+activities)?)", lowered)
    if from_match:
        return from_match.group(1).replace(" activities", "").strip()
    subject_match = re.search(
        r"\b([a-z&]+(?:\s+activities)?)\s+(?:brought|brings|provided).{0,50}\bmost\b",
        lowered,
    )
    return subject_match.group(1).replace(" activities", "").strip() if subject_match else None


def proper_names(text: str) -> set[str]:
    ignored = {"As", "FY", "No", "The", "Yes"}
    return {
        token.lower()
        for token in re.findall(r"\b[A-Z][A-Za-z&]{2,}\b", text or "")
        if token not in ignored
    }


def diagnose_resolution(category: str, reference: str, answer: str) -> dict:
    """Return a conservative, deterministic diagnostic rather than a Judge score."""
    lowered = (answer or "").lower()
    refusal = any(pattern in lowered for pattern in REFUSAL_PATTERNS)
    reference_numbers = normalized_numbers(reference)
    answer_numbers = normalized_numbers(answer)
    numeric_coverage = (
        len(reference_numbers & answer_numbers) / len(reference_numbers)
        if reference_numbers else None
    )
    reference_terms = content_terms(reference)
    term_coverage = len(reference_terms & content_terms(answer)) / len(reference_terms) if reference_terms else 1.0
    leading_term_coverage = (
        len(reference_terms & content_terms((answer or "")[:600])) / len(reference_terms)
        if reference_terms else 1.0
    )
    polarity = expected_polarity(reference)
    polarity_match = polarity is None or leading_polarity(answer) == polarity
    direction = expected_direction(reference)
    direction_match = direction is None or direction_in_conclusion(answer) == direction
    reference_choice = selection_choice(reference)
    answer_choice = selection_choice(answer)
    selection_match = reference_choice is None or answer_choice == reference_choice
    required_fact_terms = CRITICAL_FACT_TERMS & content_terms(reference)
    critical_fact_match = required_fact_terms <= content_terms((answer or "")[:600])
    required_names = proper_names(reference)
    proper_name_match = required_names <= content_terms(answer)

    diagnostics = {
        "refusal_detected": refusal,
        "reference_numbers": sorted(reference_numbers),
        "matched_reference_numbers": sorted(reference_numbers & answer_numbers),
        "numeric_coverage": numeric_coverage,
        "reference_term_coverage": round(term_coverage, 4),
        "leading_reference_term_coverage": round(leading_term_coverage, 4),
        "expected_polarity": polarity,
        "polarity_match": polarity_match,
        "expected_direction": direction,
        "direction_match": direction_match,
        "reference_selection_choice": reference_choice,
        "answer_selection_choice": answer_choice,
        "selection_match": selection_match,
        "required_critical_fact_terms": sorted(required_fact_terms),
        "critical_fact_match": critical_fact_match,
        "required_proper_names": sorted(required_names),
        "proper_name_match": proper_name_match,
    }
    if category in {"evidence_not_sufficient", "other"}:
        return {"status": "not_applicable", "reason": "Failure is not cleanly answer-prompt-actionable.", **diagnostics}

    number_ok = numeric_coverage is None or numeric_coverage >= 0.8
    if category == "calculation_failure":
        resolved = not refusal and bool(reference_numbers) and numeric_coverage == 1.0
    elif category == "refusal_failure":
        resolved = (
            not refusal and polarity_match and direction_match and selection_match
            and critical_fact_match and proper_name_match and number_ok and leading_term_coverage >= 0.4
        )
    elif category == "terminology_failure":
        resolved = (
            not refusal and number_ok and direction_match and selection_match
            and critical_fact_match and proper_name_match and leading_term_coverage >= 0.4
        )
    else:
        resolved = (
            not refusal and polarity_match and direction_match and selection_match
            and critical_fact_match and proper_name_match and number_ok and leading_term_coverage >= 0.4
        )
    return {
        "status": "likely_resolved" if resolved else "not_resolved",
        "reason": "Deterministic reference-alignment diagnostic; not an official Judge result.",
        **diagnostics,
    }


def write_outputs(directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "state.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / "state.json")
    rows = [
        {"question_id": record["question_id"], "failure_type": record["failure_type"], **result}
        for record in payload["records"]
        for result in record["results"].values()
        if result.get("status") == "ok"
    ]
    (directory / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (directory / "summary.json").write_text(
        json.dumps(build_summary(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / "answers.md").write_text(render_markdown(payload), encoding="utf-8")


def build_summary(payload: dict) -> dict:
    modes = payload["manifest"]["prompt_modes"]
    summary = {"questions": len(payload["records"]), "modes": {}}
    for mode in modes:
        results = [record["results"].get(mode, {}) for record in payload["records"]]
        completed = [result for result in results if result.get("status") == "ok"]
        actionable = [result for result in completed if result.get("resolution", {}).get("status") != "not_applicable"]
        likely = [result for result in actionable if result["resolution"]["status"] == "likely_resolved"]
        input_tokens = [int(result.get("usage", {}).get("input_tokens") or 0) for result in completed]
        latencies = [float(result.get("latency_ms") or 0) for result in completed]
        summary["modes"][mode] = {
            "completed": len(completed),
            "actionable": len(actionable),
            "likely_resolved": len(likely),
            "diagnostic_resolution_rate": len(likely) / len(actionable) if actionable else None,
            "total_input_tokens": sum(input_tokens),
            "average_input_tokens": sum(input_tokens) / len(input_tokens) if input_tokens else None,
            "average_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        }
    baseline_tokens = summary["modes"].get("baseline", {}).get("average_input_tokens") or 0
    for values in summary["modes"].values():
        average_tokens = values["average_input_tokens"]
        values["average_input_token_increase_vs_baseline"] = (
            average_tokens - baseline_tokens if average_tokens is not None else None
        )
    summary["diagnostic_note"] = "Resolution is deterministic reference alignment, not a Judge score."
    return summary


def render_markdown(payload: dict) -> str:
    summary = build_summary(payload)
    lines = [
        "# Finance Reasoning Prompt v1.1 Shadow",
        "",
        "固定复用 Jina full baseline 的15题Evidence。Baseline直接复用已有答案；仅v1与v1.1新增生成。",
        "没有Retrieval、RRF、Jina、Judge或LangSmith调用。`likely_resolved`仅为确定性参考答案对齐诊断。",
        "",
        "## 汇总",
        "",
        "| Prompt mode | 完成 | 可诊断 | Likely resolved | 输入token/题 | 相对Baseline | 延迟ms/题 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, values in summary["modes"].items():
        average_tokens = values["average_input_tokens"]
        token_increase = values["average_input_token_increase_vs_baseline"]
        average_latency = values["average_latency_ms"]
        lines.append(
            f"| `{mode}` | {values['completed']} | {values['actionable']} | {values['likely_resolved']} | "
            f"{average_tokens:.1f} | {token_increase:+.1f} | {average_latency:.1f} |"
            if average_tokens is not None and token_increase is not None and average_latency is not None
            else f"| `{mode}` | {values['completed']} | {values['actionable']} | {values['likely_resolved']} | pending | pending | pending |"
        )
    lines.extend(["", "## 逐题答案", ""])
    for index, record in enumerate(payload["records"], 1):
        lines.extend([
            f"### {index}. {record['question_id']} — `{record['failure_type']}`",
            "",
            f"**问题：** {record['question']}",
            "",
            f"**参考答案：** {record['reference_answer']}",
            "",
        ])
        for mode in payload["manifest"]["prompt_modes"]:
            result = record["results"].get(mode, {})
            resolution = result.get("resolution", {})
            lines.extend([
                f"#### {mode}",
                "",
                result.get("final_answer", "尚未完成"),
                "",
                f"- 诊断：`{resolution.get('status', 'pending')}` — {resolution.get('reason', '')}",
                f"- Token：`{json.dumps(result.get('usage', {}), ensure_ascii=False)}`",
                f"- 延迟：`{result.get('latency_ms')} ms`",
                "",
            ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_path = ROOT / config["source_state"]
    audit_path = ROOT / config["audit_json"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    frozen = {record["financebench_id"]: record for record in source["records"]}
    question_ids = [item["question_id"] for item in audit["items"]]
    if len(question_ids) != 15 or any(question_id not in frozen for question_id in question_ids):
        raise ValueError("Expected all 15 audited questions in the frozen state")

    manifest = {
        "experiment": config["name"],
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "source_state_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "question_ids": question_ids,
        "prompt_modes": config["prompt_modes"],
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "judge_calls": 0,
    }
    audit_items = {item["question_id"]: item for item in audit["items"]}
    payload = {"manifest": manifest, "records": []}
    for question_id in question_ids:
        item = audit_items[question_id]
        source_record = frozen[question_id]
        baseline_answer = source_record["answer"]
        baseline = {
            "prompt_mode": "baseline",
            "source": "reused_frozen_jina_full_baseline",
            "final_answer": baseline_answer,
            "usage": source_record.get("usage", {}),
            "latency_ms": round(float(source_record.get("latency_ms", {}).get("answer") or 0), 2),
            "resolution": {
                "status": "not_resolved" if item["category"] not in {"evidence_not_sufficient", "other"} else "not_applicable",
                "reason": "Frozen baseline has an existing strict Judge=false result; no new Judge call was made.",
            },
            "status": "ok",
        }
        payload["records"].append({
            "question_id": question_id,
            "question_type": item.get("question_type"),
            "failure_type": item["category"],
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "evidence_sha256": digest_text(source_record["evidence"]),
            "results": {"baseline": baseline},
        })

    state_path = args.output_dir / "state.json"
    if state_path.exists():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if payload.get("manifest") != manifest:
            raise ValueError("Shadow checkpoint drift; use a new output directory")
        for record in payload["records"]:
            record["results"]["baseline"]["resolution"] = {
                "status": (
                    "not_applicable"
                    if record["failure_type"] in {"evidence_not_sufficient", "other"}
                    else "not_resolved"
                ),
                "reason": "Frozen baseline has an existing strict Judge=false result; no new Judge call was made.",
            }
            for mode in ("finance_reasoning", "finance_reasoning_v1_1"):
                result = record["results"].get(mode)
                if result and result.get("status") == "ok":
                    result["resolution"] = diagnose_resolution(
                        record["failure_type"], record["reference_answer"], result["final_answer"]
                    )

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    from runtime_profile import apply_runtime_profile
    apply_runtime_profile(config["answer"]["profile"])
    os.environ.update({
        "MODEL": config["answer"]["model"],
        "ANSWER_TEMPERATURE": str(config["answer"]["temperature"]),
        "ANSWER_MAX_COMPLETION_TOKENS": str(config["answer"]["max_completion_tokens"]),
        "ANSWER_THINKING_MODE": config["answer"]["thinking"],
        "ANSWER_TIMEOUT_SECONDS": str(config["answer"]["timeout_seconds"]),
        "ANSWER_MAX_RETRIES": str(config["answer"]["max_retries"]),
        "LANGSMITH_TRACING": "false",
        "LANGSMITH_TRACING_V2": "false",
        "LANGCHAIN_TRACING_V2": "false",
    })
    from answer_generator import generate_answer

    write_outputs(args.output_dir, payload)
    for record in payload["records"]:
        source_record = frozen[record["question_id"]]
        evidence = source_record["evidence"]
        if digest_text(evidence) != record["evidence_sha256"]:
            raise ValueError("Frozen evidence drift")
        for mode in ("finance_reasoning", "finance_reasoning_v1_1"):
            if record["results"].get(mode, {}).get("status") == "ok":
                continue
            started = time.perf_counter()
            answer, usage = generate_answer(
                record["question"], evidence, [], "", config["answer"]["profile"], mode
            )
            if not answer.strip():
                raise RuntimeError("Empty answer")
            record["results"][mode] = {
                "prompt_mode": mode,
                "source": "new_shadow_generation",
                "final_answer": answer,
                "usage": usage,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "resolution": diagnose_resolution(record["failure_type"], record["reference_answer"], answer),
                "status": "ok",
            }
            write_outputs(args.output_dir, payload)
            print(f"[{record['question_id']}] {mode} ok", flush=True)
    payload["complete"] = True
    write_outputs(args.output_dir, payload)
    print(json.dumps(build_summary(payload), ensure_ascii=False, indent=2))
    print(f"Report: {args.output_dir / 'answers.md'}")


if __name__ == "__main__":
    main()
