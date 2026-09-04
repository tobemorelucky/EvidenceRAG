"""RRF Top120 frozen diagnostic30 -> independent identity/Jina/local BGE ranks.

Never import production retrieval, Assembly or Packing. Gold is evaluated only
after backend ranking. Resume completed routes; failed routes stay explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.shadow_rerankers_v1 import IdentityReranker, JinaReranker, BGEReranker, validate_order

GROUPS = ("candidate_miss10", "selection_loss10", "correct_regression10")
DEFAULT_INPUT = ROOT / "reports/reranker_shadow_v1_rrf_top120.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def scoring_manifest(manifest):
    """Request pacing is operational, not a model/input/context change."""
    result = json.loads(json.dumps(manifest))
    for config in result.get("backends", {}).values():
        if isinstance(config, dict):
            config.pop("interval_seconds", None)
    return result


def filename(value):
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name if name.endswith(".pdf") else name + ".pdf"


def page(chunk):
    return filename(chunk["filename"]), int(chunk["page_number"])


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def fixture_rows():
    fixture = json.loads((ROOT / "tests/fixtures/rag_core_v3_diagnostic_ids.json").read_text(encoding="utf-8"))
    groups = {r["financebench_id"] if isinstance(r, dict) else r: group for group in GROUPS for r in fixture[group]}
    with (ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = {r["financebench_id"]: r for r in csv.DictReader(f) if r["financebench_id"] in groups}
    if len(rows) != 30 or Counter(groups.values()) != Counter({g: 10 for g in GROUPS}):
        raise ValueError("Expected fixed 10+10+10 diagnostic set")
    return rows, groups


def validate_snapshot(payload, rows, groups):
    if payload.get("schema") != "rrf_top120_shadow_v1" or payload.get("retrieval", {}).get("method") != "MilvusManager.hybrid_retrieve":
        raise ValueError("Not a verified RRF snapshot; Dense Primary/Top60 are not substitutes")
    records = payload["records"]
    if len(records) != 30 or {r["question_id"] for r in records} != set(rows):
        raise ValueError("Snapshot must contain exactly the frozen diagnostic30")
    for r in records:
        if r["question"] != rows[r["question_id"]]["question"] or r["group"] != groups[r["question_id"]]:
            raise ValueError("Question/group drift")
        chunks = r["chunks"]
        if len(chunks) != 120 or len({c["chunk_id"] for c in chunks}) != 120:
            raise ValueError("Exactly 120 unique chunk IDs required per query")
        if any(not c["chunk_id"] or not c["text"].strip() or int(c["page_number"]) < 0 for c in chunks):
            raise ValueError("Missing text/identity or invalid internal page")
        if [c["rrf_rank"] for c in chunks] != list(range(1, 121)) or digest(chunks) != r["candidate_sha256"]:
            raise ValueError("Frozen chunk content/order drift")
    return records


def metrics(row, ordered, context_chunks=8, budget=28000):
    gold = json.loads(row.get("evidence") or "[]")
    gold_pages = {(filename(g["doc_name"]), int(g["evidence_page_num"])) for g in gold}
    # Benchmark has no chunk IDs. This is explicitly a same-page literal-span
    # proxy, not a fabricated official chunk annotation or numeric token match.
    def span_hit(chunk, text):
        for g in gold:
            if page(chunk) != (filename(g["doc_name"]), int(g["evidence_page_num"])):
                continue
            lines = [normalize(line) for line in g.get("evidence_text", "").splitlines() if len(normalize(line)) >= 40]
            if any(line in normalize(text) for line in lines):
                return True
        return False
    unique_pages = list(dict.fromkeys(page(c) for c in ordered))
    gold_chunk_rank = next((i for i, c in enumerate(ordered, 1) if span_hit(c, c["text"])), None)
    gold_page_rank = next((i for i, p in enumerate(unique_pages, 1) if p in gold_pages), None)
    parts, context_pages, remaining = [], [], budget
    context_span_hit = False
    for chunk in ordered[:context_chunks]:
        separator = 2 if parts else 0
        if remaining <= separator:
            break
        text = chunk["text"][:remaining - separator]
        parts.append(text)
        context_pages.append(page(chunk))
        remaining -= len(text) + separator
        context_span_hit |= span_hit(chunk, text)
    return {"gold_chunk_rank": gold_chunk_rank, "gold_page_rank": gold_page_rank,
            **{f"page_hit_at_{k}": bool(gold_pages & set(unique_pages[:k])) for k in (5, 10, 20)},
            "context_hit": bool(gold_pages & set(context_pages)),
            "context_all_gold_pages_hit": bool(gold_pages) and gold_pages <= set(context_pages),
            "context_evidence_span_hit": context_span_hit, "context_chars": len("\n\n".join(parts)),
            "context_pages": [list(p) for p in dict.fromkeys(context_pages)],
            "candidate_gold_page_hit": bool(gold_pages & set(unique_pages))}


def summarize(records, backends):
    summary = {}
    for backend in backends:
        valid = [r for r in records if r["routes"].get(backend, {}).get("status") == "ok"]
        scores = [r["routes"][backend]["metrics"] for r in valid]
        item = {"completed": len(valid), "statuses": dict(Counter(r["routes"].get(backend, {}).get("status", "not_run") for r in records))}
        for key in ("candidate_gold_page_hit", "page_hit_at_5", "page_hit_at_10", "page_hit_at_20", "context_hit", "context_evidence_span_hit"):
            item[key] = fmean(m[key] for m in scores) if scores else None
        for key in ("gold_chunk_rank", "gold_page_rank"):
            ranks = [m[key] for m in scores if m[key] is not None]
            item[key] = {"hit_questions": len(ranks), "mean_on_hits": fmean(ranks) if ranks else None}
        paired = [r for r in valid if r["routes"].get("identity", {}).get("status") == "ok"]
        item["context_gains_vs_identity"] = [r["question_id"] for r in paired if r["routes"][backend]["metrics"]["context_hit"] and not r["routes"]["identity"]["metrics"]["context_hit"]]
        item["context_regressions_vs_identity"] = [r["question_id"] for r in paired if not r["routes"][backend]["metrics"]["context_hit"] and r["routes"]["identity"]["metrics"]["context_hit"]]
        item["groups"] = {g: {"completed": sum(r["group"] == g for r in valid),
                            "context_hit": fmean(r["routes"][backend]["metrics"]["context_hit"] for r in valid if r["group"] == g)
                            if any(r["group"] == g for r in valid) else None} for g in GROUPS}
        routes = [r["routes"][backend] for r in valid]
        item["runtime"] = {
            "successful_input_chars": sum(r.get("input_chars", 0) for r in routes),
            "reported_total_tokens": sum((r.get("trace", {}).get("usage") or {}).get("total_tokens", 0) for r in routes),
            "truncated_pairs": sum(r.get("trace", {}).get("truncated_pairs", 0) for r in routes),
            "mean_route_latency_ms_including_pacing_and_cold_start": fmean(r.get("latency_ms", 0) for r in routes) if routes else None,
            "prior_errors": [error for r in records for error in r["routes"].get(backend, {}).get("prior_errors", [])],
        }
        summary[backend] = item
    return summary


def finalize_report(payload, frozen, rows):
    """Recompute metrics locally and compare only a common completed subset."""
    by_id = {r["question_id"]: r for r in frozen}
    records = payload["records"]
    if len({r["question_id"] for r in records}) != len(records):
        raise ValueError("Duplicate checkpoint question")
    verified = 0
    for record in records:
        source = by_id[record["question_id"]]
        if record["candidate_sha256"] != source["candidate_sha256"]:
            raise ValueError("Checkpoint candidate fingerprint drift")
        for route in record["routes"].values():
            if route["status"] != "ok":
                continue
            ranked = validate_order(route["ranked"], len(source["chunks"]))
            if ranked != route["ranked"] or metrics(rows[record["question_id"]], [source["chunks"][r["index"]] for r in ranked]) != route["metrics"]:
                raise ValueError("Saved ranking/metrics mismatch")
            verified += 1
    backends = list(payload["manifest"]["backends"])
    common = [r for r in records if all(r["routes"].get(b, {}).get("status") == "ok" for b in backends)]
    payload["summary"] = summarize(records, backends)
    payload["common_subset"] = {"question_ids": [r["question_id"] for r in common],
                                "summary": summarize(common, backends)}
    payload["verification"] = {"snapshot_questions": len(frozen), "snapshot_chunks": sum(len(r["chunks"]) for r in frozen),
                               "completed_routes_recomputed": verified, "candidate_hashes_and_rankings_match": True}
    return payload


def markdown(payload):
    lines = ["# Reranker Shadow Evaluation v1", "",
        "固定30题，每题同一RRF Top120原始chunks；只改变重排。无LLM/Judge/LangSmith，不调用生产Assembly/Packing。", "",
        "- gold page使用FinanceBench内部0-based页码，不减1。",
        "- gold chunk rank是同gold页内≥40字符参考原文行完整匹配的代理；FinanceBench不提供chunk ID真值。无匹配记null，不伪装成120/121。",
        "- gold page rank及page hit@K按重排后首次出现的唯一页面排序。",
        "- context hit为固定Top8 chunks、最多28000字符shadow投影中出现gold页。不是生产Assembly/Packing context，也不证明答案/事实正确。context_evidence_span_hit另检查截断后实际原文。",
        "- Jina/BGE均重排全部120项，失败单独记录，不降级或用identity冒充成功。BGE token截断数单独记录。", "",
        "| Backend | 成功/30 | Candidate hit | Page@5 | Page@10 | Page@20 | Shadow context hit |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    def pct(v):
        return "N/A" if v is None else f"{v:.2%}"
    for name, s in payload["summary"].items():
        lines.append(f"| {name} | {s['completed']}/30 | {pct(s['candidate_gold_page_hit'])} | {pct(s['page_hit_at_5'])} | {pct(s['page_hit_at_10'])} | {pct(s['page_hit_at_20'])} | {pct(s['context_hit'])} |")
    lines += ["", "缺失/失败不算无命中，也不用于宣称整体改善；只在相同完成题上比较。完整逐题排名、内容哈希与失败状态见JSON。", ""]
    if "common_subset" in payload:
        lines += [f"## 公平对比：共同完成的{len(payload['common_subset']['question_ids'])}题", "",
                  "| Backend | Candidate hit | Gold chunk rank均值* | Gold page rank均值 | Page@5 | Page@10 | Page@20 | Shadow context hit |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, s in payload["common_subset"]["summary"].items():
            chunk_rank = s["gold_chunk_rank"]["mean_on_hits"]
            page_rank = s["gold_page_rank"]["mean_on_hits"]
            chunk_rank = "N/A" if chunk_rank is None else f"{chunk_rank:.2f}"
            page_rank = "N/A" if page_rank is None else f"{page_rank:.2f}"
            lines.append(f"| {name} | {pct(s['candidate_gold_page_hit'])} | {chunk_rank} | {page_rank} | {pct(s['page_hit_at_5'])} | {pct(s['page_hit_at_10'])} | {pct(s['page_hit_at_20'])} | {pct(s['context_hit'])} |")
        lines += ["", "排名均值仅对命中题计算；chunk为原文匹配代理，不是官方chunk标注。", ""]
    for name, s in payload["summary"].items():
        lines += [f"## {name}", "", f"- Status: `{s['statuses']}`",
                  f"- Gold chunk rank代理：`{s['gold_chunk_rank']}`；Gold page rank：`{s['gold_page_rank']}`（均值仅对命中题计算，未命中仍为null）。",
                  f"- 成本/截断/历史错误：`{s['runtime']}`（reported_total_tokens仅为后端返回值，0不代表本地未处理token）。",
                  f"- Context新增命中：{s['context_gains_vs_identity']}",
                  f"- Context回退：{s['context_regressions_vs_identity']}",
                  f"- 分组：`{s['groups']}`", ""]
    lines += ["## 逐题", "", "| ID | Backend | Status | Gold chunk rank* | Gold page rank | Context hit |",
              "|---|---|---|---:|---:|---|"]
    for r in payload["records"]:
        for b, route in r["routes"].items():
            m = route.get("metrics", {})
            lines.append(f"| {r['question_id']} | {b} | {route['status']} | {m.get('gold_chunk_rank')} | {m.get('gold_page_rank')} | {m.get('context_hit')} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/reranker_shadow_v1.json")
    parser.add_argument("--backends", nargs="+", choices=["identity", "jina", "bge"], default=["identity", "jina", "bge"])
    parser.add_argument("--bge-batch-size", type=int, default=4)
    parser.add_argument("--bge-max-length", type=int, default=1024)
    parser.add_argument("--jina-interval-seconds", type=float, default=8)
    parser.add_argument("--summarize-only", action="store_true", help="Validate cached results and render report; no model/API requests")
    args = parser.parse_args()
    if not 0 <= args.jina_interval_seconds <= 60:
        parser.error("Jina interval must be between 0 and 60 seconds")
    # Fail before model initialization or paid API calls when input is absent.
    if not args.input.exists():
        parser.error(f"Missing frozen RRF Top120 snapshot: {args.input}. Do not substitute Dense Primary or Top60.")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LANGCHAIN_TRACING_V2="false", LANGSMITH_TRACING="false")
    rows, groups = fixture_rows()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    frozen = validate_snapshot(source, rows, groups)
    if args.summarize_only:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if hashlib.sha256(args.input.read_bytes()).hexdigest() != payload["manifest"]["input_sha256"]:
            raise ValueError("Checkpoint belongs to another snapshot")
        payload = finalize_report(payload, frozen, rows)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        args.output.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
        print(json.dumps(payload["verification"], ensure_ascii=False), flush=True)
        print(f"Report: {args.output.with_suffix('.md')}", flush=True)
        return
    print(f"[setup] frozen_questions=30 chunks=3600 input_chars={sum(len(c['text']) for r in frozen for c in r['chunks'])} no_generation_or_judge", flush=True)
    from dotenv import dotenv_values
    env = {**dotenv_values(ROOT / ".env"), **os.environ}
    adapters = {"identity": IdentityReranker()}
    unavailable = {}
    if "jina" in args.backends:
        key = env.get("RERANK_API_KEY") or env.get("JINA_API_KEY")
        if key:
            host = env.get("RERANK_BINDING_HOST") or "https://api.jina.ai"
            endpoint = host.rstrip("/") if host.rstrip("/").endswith("/v1/rerank") else host.rstrip("/") + "/v1/rerank"
            adapters["jina"] = JinaReranker(key, env.get("RERANK_MODEL") or "jina-reranker-v3", endpoint,
                                           interval=args.jina_interval_seconds)
        else:
            unavailable["jina"] = "credential_not_configured"
    if "bge" in args.backends:
        try:
            adapters["bge"] = BGEReranker(ROOT / (env.get("LOCAL_RERANK_MODEL_PATH") or "models/bge-reranker-v2-m3"),
                env.get("LOCAL_RERANK_DEVICE") or "auto", args.bge_batch_size, args.bge_max_length)
        except FileNotFoundError:
            unavailable["bge"] = "local_weights_missing"
    manifest = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "dataset_sha256": hashlib.sha256((ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv").read_bytes()).hexdigest(),
                "backends": {b: adapters[b].config if b in adapters else unavailable[b] for b in args.backends},
                "context_chunks": 8, "context_budget": 28000}
    saved = {}
    if args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        if scoring_manifest(old["manifest"]) != scoring_manifest(manifest):
            raise ValueError("Existing output belongs to another manifest; choose a new --output")
        saved = {r["question_id"]: r for r in old["records"]}
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(frozen, 1):
        record = saved.get(r["question_id"], {"question_id": r["question_id"], "question": r["question"], "group": r["group"], "candidate_sha256": r["candidate_sha256"], "routes": {}})
        texts = [c["text"] for c in r["chunks"]]
        for backend in args.backends:
            if record["routes"].get(backend, {}).get("status") == "ok":
                continue
            print(f"[{i:02d}/30] {backend} starting chunks=120 chars={sum(map(len, texts))}", flush=True)
            started = time.perf_counter()
            if backend in unavailable:
                route = {"status": "unavailable", "reason": unavailable[backend]}
            else:
                try:
                    ranked, trace = adapters[backend].rank(r["question"], texts)
                    ranked = validate_order(ranked, 120)
                    ordered = [r["chunks"][v["index"]] for v in ranked]
                    route = {"status": "ok", "ranked": ranked, "trace": trace, "metrics": metrics(rows[r["question_id"]], ordered)}
                except Exception as exc:
                    route = {"status": "error", "error_type": type(exc).__name__}
                    if backend == "jina" and isinstance(exc, RuntimeError):
                        route["reason"] = str(exc) # sanitized by adapter
                    # Don't repeat 30 paid/network failures or repeatedly reload
                    # a broken local model. A new invocation may retry errors.
                    unavailable[backend] = "circuit_open_after_error"
            route["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            route["input_chars"] = sum(map(len, texts))
            if backend == "jina":
                route["request_interval_seconds"] = args.jina_interval_seconds
            prior_route = record["routes"].get(backend, {})
            if prior_route.get("status") == "error":
                route["prior_errors"] = prior_route.get("prior_errors", []) + [
                    {k: prior_route.get(k) for k in ("status", "error_type", "reason", "latency_ms")}]
            record["routes"][backend] = route
            print(f"[{i:02d}/30] {backend} {route['status']} {route['latency_ms']}ms", flush=True)
            saved[record["question_id"]] = record
            partial = [saved[x["question_id"]] for x in frozen if x["question_id"] in saved]
            payload = {"manifest": manifest, "summary": summarize(partial, args.backends), "records": partial}
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != manifest["input_sha256"]:
        raise AssertionError("Frozen input changed")
    payload = {"manifest": manifest, "summary": summarize(records, args.backends), "records": records}
    payload = finalize_report(payload, frozen, rows)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    if any(payload["summary"][b]["completed"] != 30 for b in args.backends
           if not (b == "jina" and manifest["backends"][b] == "credential_not_configured")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
