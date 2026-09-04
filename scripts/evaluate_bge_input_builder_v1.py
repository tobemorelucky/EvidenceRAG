"""Fixed30 frozen RRF -> local BGE input-view experiment. No network backends.

--prepare-only constructs auditable views using tokenizer.json, no torch/model.
--summarize-only validates cached results, no tokenizer/model initialization.
Default resumes preparation and then scores only unfinished local BGE questions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bge_reranker_input_builder_v1 import LocalPairTokenizer, VERSION, build_input, render
from scripts.evaluate_reranker_shadow_v1 import (
    DEFAULT_INPUT, GROUPS, digest, fixture_rows, validate_snapshot, finalize_report,
    metrics, normalize, filename, summarize,
)
from scripts.shadow_rerankers_v1 import BGEReranker, validate_order

BACKENDS = ("identity", "bge_raw", "jina_cached", "bge_input_v1")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_resources(min_available_gib=4):
    import psutil
    available = psutil.virtual_memory().available / 1024**3
    if available < min_available_gib:
        raise RuntimeError(f"Available RAM {available:.2f} GiB < {min_available_gib} GiB; checkpoint saved. Close unused applications manually, then resume. No processes were stopped.")
    return round(available, 3)


def visible_gold_span(row, chunk, visible):
    # Diagnostic ONLY: builder never receives this row or benchmark annotations.
    for gold in json.loads(row.get("evidence") or "[]"):
        if (filename(chunk["filename"]), int(chunk["page_number"])) != (filename(gold["doc_name"]), int(gold["evidence_page_num"])):
            continue
        lines = [normalize(x) for x in gold.get("evidence_text", "").splitlines() if len(normalize(x)) >= 40]
        if any(line in normalize(visible) for line in lines):
            return True
    return False


def validate_prepared(record, source):
    if record["candidate_sha256"] != source["candidate_sha256"] or len(record["inputs"]) != 120:
        raise ValueError("Prepared candidate set drift")
    views = record["inputs"]
    if digest(views) != record["inputs_sha256"]:
        raise ValueError("Prepared view fingerprint drift")
    for i, (view, chunk) in enumerate(zip(views, source["chunks"])):
        if view["index"] != i or view["chunk_id"] != chunk["chunk_id"]:
            raise ValueError("Prepared view identity/order drift")
        spans = view["source_spans"]
        if not spans or any(not 0 <= a < b <= len(chunk["text"]) for a, b in spans):
            raise ValueError("Invalid source offsets")
        if render(chunk["text"], spans) != view["text"]:
            raise ValueError("View is not verbatim source spans")
        if view["input_pair_tokens"] > 1024:
            raise ValueError("Prepared view exceeds frozen BGE window")
        if not view["changed"] and view["text"] != chunk["text"]:
            raise ValueError("Short chunk must remain unchanged")


def finish(payload, frozen, rows):
    sources = {r["question_id"]: r for r in frozen}
    if len({r["question_id"] for r in payload["records"]}) != len(payload["records"]):
        raise ValueError("Duplicate prepared record")
    for record in payload["records"]:
        validate_prepared(record, sources[record["question_id"]])
    payload = finalize_report(payload, frozen, rows)
    views = [v for r in payload["records"] for v in r["inputs"]]
    payload["input_statistics"] = {
        "prepared_questions": len(payload["records"]), "pairs": len(views),
        "changed_pairs": sum(v["changed"] for v in views),
        "original_over_budget_pairs": sum(v["original_pair_tokens"] > 1024 for v in views),
        "built_over_budget_pairs": sum(v["input_pair_tokens"] > 1024 for v in views),
        "mean_original_pair_tokens": fmean(v["original_pair_tokens"] for v in views) if views else None,
        "mean_built_pair_tokens": fmean(v["input_pair_tokens"] for v in views) if views else None,
        "source_chars_omitted_from_reranker_only": sum(v["omitted_source_chars"] for v in views),
        "gold_span_visible_before": sum(v["offline_gold_span_visible_raw"] for v in views),
        "gold_span_visible_after": sum(v["offline_gold_span_visible_built"] for v in views),
        "gold_span_recovered_pairs": sum(v["offline_gold_span_visible_built"] and not v["offline_gold_span_visible_raw"] for v in views),
        "gold_span_lost_pairs": sum(v["offline_gold_span_visible_raw"] and not v["offline_gold_span_visible_built"] for v in views),
    }
    paired = [r for r in payload["records"] if r["routes"].get("bge_input_v1", {}).get("status") == "ok"]
    def delta(group_rows):
        before = lambda r: r["routes"]["bge_raw"]["metrics"]["context_hit"]
        after = lambda r: r["routes"]["bge_input_v1"]["metrics"]["context_hit"]
        return {"completed": len(group_rows),
                "gains": [r["question_id"] for r in group_rows if after(r) and not before(r)],
                "regressions": [r["question_id"] for r in group_rows if before(r) and not after(r)]}
    payload["comparison_vs_raw_bge"] = {"all": delta(paired), **{g: delta([r for r in paired if r["group"] == g]) for g in GROUPS}}
    # A zero can mean "not run"; always retain completion counts and null rates.
    payload["network_calls_this_experiment"] = 0
    return payload


def report(payload):
    lines = ["# BGE Reranker Input Builder v1 — shadow", "",
        "固定同一30题RRF Top120；只改本地BGE输入表示。未调用Jina、LLM、Judge、LangSmith。", "",
        "短输入逐字不变；超长输入保留小段原文前缀及问题词匹配的原文窗口/相邻行，按原文顺序拼接。无公司、指标、答案或ID规则。",
        "模型/精度/窗口/batch保持旧BGE配置（CUDA FP16、1024、4）；每题仍120对，不做多窗口多次评分。",
        "最终context仍使用重排后的完整原始chunks，Top8且≤28000字符，不使用压缩后的reranker文本。", "",
        f"- Snapshot SHA256: `{payload['manifest']['input_sha256']}`",
        f"- 状态：{payload.get('run_status', 'prepared')}；完整评测前不得宣称接近Jina。",
        f"- 输入统计：`{json.dumps(payload['input_statistics'], ensure_ascii=False)}`", "",
        "## 指标", "", "| Backend | 完成 | Candidate hit | Page@5 | Page@10 | Page@20 | Context hit | Context span hit | Gold page rank* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def pct(x):
        return "N/A" if x is None else f"{x:.2%}"
    for name, s in payload["summary"].items():
        rank = s["gold_page_rank"]["mean_on_hits"]
        lines.append(f"| {name} | {s['completed']}/30 | {pct(s['candidate_gold_page_hit'])} | {pct(s['page_hit_at_5'])} | {pct(s['page_hit_at_10'])} | {pct(s['page_hit_at_20'])} | {pct(s['context_hit'])} | {pct(s['context_evidence_span_hit'])} | {rank if rank is not None else 'N/A'} |")
    lines += ["", "identity/bge_raw/jina_cached均来自已有同快照结果，未重新调用。未完成题不按失败/零命中计算；公平对比只使用JSON common_subset中的共同完成题。",
        "Gold page rank均值仅统计命中题；gold chunk rank是≥40字符原文匹配代理，完整排名见JSON。context hit不是答案正确率。",
        "输入中的gold可见性字段仅在构造后离线评估，重复chunk对不是独立题目；零tokenizer截断不代表零证据丢失。", "",
        "## 分组与逐题回退", "", f"`{json.dumps(payload['comparison_vs_raw_bge'], ensure_ascii=False)}`", "",
        "| Group | Backend | 完成 | Context hit |", "|---|---|---:|---:|"]
    for name, s in payload["summary"].items():
        for group, g in s["groups"].items():
            lines.append(f"| {group} | {name} | {g['completed']}/10 | {pct(g['context_hit'])} |")
    lines += ["", "## 逐题", "", "| ID | 改变输入数 | Raw BGE context | Input v1 context | Raw page rank | v1 page rank |", "|---|---:|---|---|---:|---:|"]
    for r in payload["records"]:
        a = r["routes"]["bge_raw"]["metrics"]
        b = r["routes"].get("bge_input_v1", {}).get("metrics", {})
        lines.append(f"| {r['question_id']} | {sum(v['changed'] for v in r['inputs'])} | {a['context_hit']} | {b.get('context_hit', 'not run')} | {a['gold_page_rank']} | {b.get('gold_page_rank', 'not run')} |")
    lines += ["", "## 代价与边界", "",
        f"- 本轮本地BGE运行统计：`{payload['summary']['bge_input_v1']['runtime']}`",
        "- 每题input_builder_ms和ranking latency分开记录；模型冷启动不能与热启动直接比较。",
        "- 此30题已反复用于诊断，属于开发集，不是未见测试集；本轮不按单题反馈调窗口/权重。",
        "- 若提升有限，说明截断不是唯一原因；不能据此扩大生产context、修改Prompt或自动接入。", ""]
    return "\n".join(lines)


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    path.with_suffix(".md").write_text(report(payload), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--baseline", type=Path, default=ROOT / "reports/reranker_shadow_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/bge_reranker_input_builder_v1.json")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.output.resolve() in {args.input.resolve(), args.baseline.resolve()}:
        parser.error("Output must not overwrite frozen input/baseline")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LANGCHAIN_TRACING_V2="false", LANGSMITH_TRACING="false",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    rows, groups = fixture_rows()
    frozen = validate_snapshot(json.loads(args.input.read_text(encoding="utf-8")), rows, groups)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline["manifest"]["input_sha256"] != file_hash(args.input):
        raise ValueError("Baseline uses a different candidate snapshot")
    baseline = finalize_report(baseline, frozen, rows)
    if any(baseline["summary"][b]["completed"] != 30 for b in ("identity", "bge", "jina")):
        raise ValueError("Need the completed shared baseline; never call a remote backend to fill it")
    model_config = baseline["manifest"]["backends"]["bge"]
    model_path = Path(model_config["path"])
    adapter = BGEReranker(model_path, model_config["device"], model_config["batch_size"], model_config["max_length"])
    if adapter.config != model_config or model_config["max_length"] != 1024 or model_config["batch_size"] != 4:
        raise ValueError("Model/config changed since frozen BGE baseline")
    old_records = {r["question_id"]: r for r in baseline["records"]}
    if any(r["routes"]["bge"]["trace"]["device"] != "cuda" for r in baseline["records"]):
        raise ValueError("This prototype expects the frozen CUDA baseline")
    manifest = {"input_sha256": file_hash(args.input), "baseline_sha256": file_hash(args.baseline),
                "dataset_sha256": file_hash(ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv"),
                "builder_version": VERSION, "builder_sha256": file_hash(ROOT / "scripts/bge_reranker_input_builder_v1.py"),
                "evaluator_sha256": file_hash(Path(__file__)),
                "tokenizer_sha256": file_hash(model_path / "tokenizer.json"),
                "backends": {b: (model_config if b == "bge_input_v1" else {"source": "cached_only"}) for b in BACKENDS},
                "context_chunks": 8, "context_budget": 28000}
    payload = {"manifest": manifest, "records": []}
    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload["manifest"] != manifest:
            raise ValueError("Checkpoint manifest changed; use a new --output, do not overwrite")
        finish(payload, frozen, rows)
    if args.summarize_only:
        if not args.output.exists():
            parser.error("No checkpoint to summarize")
        save(args.output, finish(payload, frozen, rows))
        return
    saved = {r["question_id"]: r for r in payload["records"]}
    tokenizer = None
    for i, source in enumerate(frozen, 1):
        if source["question_id"] in saved:
            continue
        if tokenizer is None:
            check_resources(1)
            tokenizer = LocalPairTokenizer(model_path / "tokenizer.json")
        started = time.perf_counter()
        inputs = []
        for index, chunk in enumerate(source["chunks"]):
            view = build_input(source["question"], chunk["text"], tokenizer)
            raw_visible = tokenizer.baseline_visible_text(source["question"], chunk["text"], 1024) if view["changed"] else chunk["text"]
            view.update(index=index, chunk_id=chunk["chunk_id"],
                        offline_gold_span_visible_raw=visible_gold_span(rows[source["question_id"]], chunk, raw_visible),
                        offline_gold_span_visible_built=visible_gold_span(rows[source["question_id"]], chunk, view["text"]))
            inputs.append(view)
        record = {"question_id": source["question_id"], "question": source["question"], "group": source["group"],
                  "candidate_sha256": source["candidate_sha256"], "inputs": inputs, "inputs_sha256": digest(inputs),
                  "input_builder_ms": round((time.perf_counter() - started) * 1000, 2),
                  "routes": {new: copy.deepcopy(old_records[source["question_id"]]["routes"][old])
                             for new, old in (("identity", "identity"), ("bge_raw", "bge"), ("jina_cached", "jina"))}}
        payload["records"].append(record)
        print(f"[prepare {i:02d}/30] changed={sum(v['changed'] for v in inputs)}/120 builder_ms={record['input_builder_ms']}", flush=True)
        payload["run_status"] = "prepared_only"
        save(args.output, finish(payload, frozen, rows))
    del tokenizer
    if args.prepare_only:
        print(f"Prepared: {args.output}; no model inference", flush=True)
        return
    try:
        if any(r["routes"].get("bge_input_v1", {}).get("status") != "ok" for r in payload["records"]):
            available = check_resources(4)
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; refusing CPU/config drift")
            free, _ = torch.cuda.mem_get_info()
            if free < 4 * 1024**3:
                raise RuntimeError("Less than 4 GiB free VRAM; no model loaded")
            torch.set_num_threads(2)
            adapter.device = "cuda"
            print(f"[setup] local BGE only; available_ram={available}GiB; batch=4 max_length=1024", flush=True)
        for i, record in enumerate(payload["records"], 1):
            if record["routes"].get("bge_input_v1", {}).get("status") == "ok":
                continue
            check_resources(1)
            started = time.perf_counter()
            print(f"[rank {i:02d}/30] local BGE 120 views", flush=True)
            ranked, trace = adapter.rank(record["question"], [v["text"] for v in record["inputs"]])
            ranked = validate_order(ranked, 120)
            if trace["truncated_pairs"]:
                raise ValueError("HF tokenizer still truncated built inputs; reject this run")
            source = next(s for s in frozen if s["question_id"] == record["question_id"])
            record["routes"]["bge_input_v1"] = {
                "status": "ok", "ranked": ranked, "trace": trace,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "input_chars": sum(len(v["text"]) for v in record["inputs"]),
                "metrics": metrics(rows[record["question_id"]], [source["chunks"][v["index"]] for v in ranked])}
            payload["run_status"] = "running"
            save(args.output, finish(payload, frozen, rows))
            print(f"[rank {i:02d}/30] done truncated=0", flush=True)
    except Exception as exc:
        payload["run_status"] = f"stopped: {type(exc).__name__}: {exc}"
        save(args.output, finish(payload, frozen, rows))
        print(payload["run_status"], flush=True)
        raise SystemExit(2) from None
    if file_hash(args.input) != manifest["input_sha256"] or file_hash(args.baseline) != manifest["baseline_sha256"]:
        raise ValueError("Frozen artifacts changed during evaluation")
    payload["run_status"] = "complete"
    save(args.output, finish(payload, frozen, rows))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
