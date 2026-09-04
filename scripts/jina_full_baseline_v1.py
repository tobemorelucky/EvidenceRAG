"""Pure helpers for the independent Jina full baseline. No runtime/API imports."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/experiments/jina_full_baseline_v1.json"
PROFILES = {
    "jina_full_baseline_v1": PROFILE,
    "jina_full_baseline_input120_v1": ROOT / "configs/experiments/jina_full_baseline_input120_v1.json",
    "jina_full_baseline_input80_v1": ROOT / "configs/experiments/jina_full_baseline_input80_v1.json",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


PARAMETER_PATHS = {
    "DENSE_TOP_K": ("retrieval", "dense_top_k"),
    "BM25_TOP_K": ("retrieval", "bm25_top_k"),
    "RRF_TOP_K": ("retrieval", "rrf_top_k"),
    "JINA_INPUT_K": ("reranker", "input_k"),
    "JINA_OUTPUT_K": ("reranker", "output_k"),
}


def _positive_int(name, value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def read_profile(environ=None, profile_name="jina_full_baseline_v1"):
    """Load the experimental profile and resolve all depth controls from env."""
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown Jina baseline profile: {profile_name}")
    profile = json.loads(PROFILES[profile_name].read_text(encoding="utf-8"))
    environ = os.environ if environ is None else environ
    defaults = profile.get("environment", {})
    for name, (section, key) in PARAMETER_PATHS.items():
        default = defaults.get(name, profile[section].get(key))
        profile[section][key] = _positive_int(name, environ.get(name, default))
        profile["environment"][name] = profile[section][key]
    if profile["name"] != profile_name or profile["skills"] or profile["planner"] or profile["langsmith"]:
        raise ValueError("v1 is a pure static Jina baseline; use a separately versioned profile for Skills/Agent")
    if profile["reranker"]["fallback"]:
        raise ValueError("Jina-only contract violated")
    if profile["reranker"]["input_k"] > profile["retrieval"]["rrf_top_k"]:
        raise ValueError("JINA_INPUT_K cannot exceed RRF_TOP_K")
    if profile["reranker"]["output_k"] > profile["reranker"]["input_k"]:
        raise ValueError("JINA_OUTPUT_K cannot exceed JINA_INPUT_K")
    profile["context"]["top_k"] = profile["reranker"]["output_k"]
    return profile


def validate_candidates(record, question, expected_count=120):
    chunks = record["chunks"]
    if record["question"] != question or len(chunks) != expected_count:
        raise ValueError(f"Question mismatch or incomplete Top{expected_count}")
    if len({c["chunk_id"] for c in chunks}) != expected_count or [c["rrf_rank"] for c in chunks] != list(range(1, expected_count + 1)):
        raise ValueError("Candidate identity/order mismatch")
    if any(not c["text"].strip() or int(c["page_number"]) < 0 or not c["filename"] for c in chunks):
        raise ValueError("Missing source text or invalid source page")
    if record["candidate_sha256"] != digest(chunks):
        raise ValueError("Candidate fingerprint mismatch")


def rrf_merge(dense, bm25, top_k, rank_constant=60):
    """Experimental independent Dense/BM25 RRF; production retrieval is untouched."""
    merged = {}
    for route, rows in (("dense", dense), ("bm25", bm25)):
        for rank, row in enumerate(rows, 1):
            key = row.get("chunk_id") or str(row.get("id"))
            if not key:
                raise ValueError(f"{route} candidate has no stable identity")
            item = merged.setdefault(key, {**row, "dense_rank": None, "bm25_rank": None, "rrf_score": 0.0})
            item[f"{route}_rank"] = rank
            item["rrf_score"] += 1.0 / (rank_constant + rank)
    ordered = sorted(
        merged.values(),
        key=lambda item: (-item["rrf_score"], item["dense_rank"] or 10**9, item["bm25_rank"] or 10**9, item.get("chunk_id", "")),
    )[:top_k]
    return [{**item, "score": item["rrf_score"], "rrf_rank": rank} for rank, item in enumerate(ordered, 1)]


def build_context(ordered, config):
    """Only source headers + sequential raw text; no page/field re-selection."""
    parts, citations, documents = [], [], []
    remaining = config["max_chars"]
    for chunk in ordered[:config["top_k"]]:
        header = f"Source: {chunk['filename']} | Page: {int(chunk['page_number'])}\n"
        separator = 2 if parts else 0
        room = remaining - separator - len(header)
        if room <= 0:
            break
        body = chunk["text"][:room]
        if not body:
            continue
        parts.append(header + body)
        remaining -= separator + len(header) + len(body)
        documents.append({**chunk, "text": body})
        citations.append({"filename": chunk["filename"], "page_number": int(chunk["page_number"]),
                          "chunk_id": chunk["chunk_id"], "included_chars": len(body), "truncated": len(body) < len(chunk["text"])})
    return "\n\n".join(parts), citations, documents


def cached_jina(source, cache, model, input_k=120):
    if input_k != len(source.get("chunks", [])):
        return None
    if cache.get("manifest", {}).get("backends", {}).get("jina", {}).get("model") != model:
        return None
    if cache["manifest"]["backends"]["jina"].get("endpoint") != "https://api.jina.ai/v1/rerank":
        return None
    for item in cache.get("records", []):
        route = item.get("routes", {}).get("jina", {})
        if item.get("question") == source["question"] and item.get("candidate_sha256") == source["candidate_sha256"] and route.get("status") == "ok":
            return route
    return None


def write_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def export_reports(directory, state, references):
    records = state["records"]
    answers = [r for r in records if r.get("answer_status") == "ok"]
    judged = [r for r in records if r.get("judge_status") == "ok"]
    for name, values in (("answers.jsonl", answers), ("judge.jsonl", [{"financebench_id": r["financebench_id"], **r["judge"]} for r in judged])):
        (directory / name).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")
    correct = sum(r["judge"]["score"] == 1 for r in judged)
    def total_usage(items, field):
        return sum(int((item.get("usage") or {}).get(field, 0)) for item in items)
    successful_jina = [r for r in records if r.get("jina", {}).get("status") == "ok"]
    summary = {"planned": len(records), "answers_completed": len(answers), "judge_completed": len(judged),
        "strict_correct": correct, "strict_accuracy": correct / len(records) if records and len(judged) == len(records) else None,
        "jina_cached_questions": sum(bool(r.get("jina_cache_hit")) for r in successful_jina),
        "jina_successful_new_questions": sum(not r.get("jina_cache_hit") for r in successful_jina),
        "jina_new_reported_tokens": sum(int((r["jina"].get("trace", {}).get("usage") or {}).get("total_tokens", 0)) for r in successful_jina if not r.get("jina_cache_hit")),
        "answer_usage": {k: total_usage(answers, k) for k in ("input_tokens", "output_tokens", "total_tokens")},
        "judge_usage": {k: total_usage([r["judge"] for r in judged], k) for k in ("input_tokens", "output_tokens", "total_tokens")},
        "errors": sum(bool(r.get("last_error")) for r in records),
        "cost_note": "Successful response usage only; cached Jina tokens not charged again; failed-request billing may be unknown"}
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Jina Full Baseline v1：问题、参考答案、模型答案与Judge", "",
        f"- 计划{len(records)}题；回答完成{len(answers)}；Judge完成{len(judged)}；正确{correct}。",
        f"- Strict accuracy：{summary['strict_accuracy'] if summary['strict_accuracy'] is not None else '尚未完整，暂不计算'}；未完成不算错误。",
        f"- 成本与状态：`{json.dumps(summary, ensure_ascii=False)}`",
        "- 纯RAG：RRF Top120 → Jina → 原文Top8（含来源≤28000字符）→ 既有clean-baseline提示词 → 独立strict Judge。",
        "- Skills/Agent/Planner/LangSmith关闭；Jina失败不降级。页码为内部0-based，不减1。",
        "- 本基线是当前架构的强参照，不是理论上限；30题shadow context hit不能当成完整答案正确率。", ""]
    for i, r in enumerate(records, 1):
        row = references[r["financebench_id"]]
        refs = "; ".join(f"{c['filename']}, page {c['page_number']}" for c in r.get("citations", [])) or "未生成"
        lines += [f"## {i}. {r['financebench_id']}", "", "### 问题", "", row["question"], "",
                  "### 参考答案", "", row.get("answer", ""), "", "### 模型答案", "", r.get("answer") or "尚未完成", "",
                  "### 引用", "", refs, "", "### Judge", "", json.dumps(r.get("judge") or {"status": "pending"}, ensure_ascii=False), "",
                  "### 指标", "", json.dumps({"retrieval": r.get("retrieval_metrics"), "usage": r.get("usage"),
                    "jina_usage": r.get("jina", {}).get("trace", {}).get("usage"), "jina_cache_hit": r.get("jina_cache_hit"),
                    "context_chars": len(r.get("evidence", "")), "latency_ms": r.get("latency_ms"), "error": r.get("last_error")}, ensure_ascii=False), ""]
    (directory / "answers.md").write_text("\n".join(lines), encoding="utf-8")
