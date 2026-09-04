"""Generate baseline/finance-reasoning answers over three frozen contexts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
CONFIG = ROOT / "configs/experiments/finance_reasoning_prompt_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/finance_reasoning_prompt_v1_smoke3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_sample(records: list[dict], size: int, seed: int) -> list[dict]:
    eligible = [record for record in records if record.get("answer_status") == "ok" and record.get("evidence")]
    if len(eligible) < size:
        raise ValueError("Not enough completed frozen contexts")
    return random.Random(seed).sample(eligible, size)


def write_outputs(directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "state.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / "state.json")
    completed = [result for record in payload["records"] for result in record["results"].values() if result.get("status") == "ok"]
    (directory / "results.jsonl").write_text(
        "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in completed), encoding="utf-8"
    )
    lines = ["# Finance Reasoning Prompt v1：固定Context三题Smoke", "",
        "同一题的baseline与finance_reasoning使用完全相同的已冻结Jina context；无Retrieval、Jina或Judge调用。", ""]
    for index, record in enumerate(payload["records"], 1):
        lines += [f"## {index}. {record['question_id']} ({record['question_type']})", "", record["question"], ""]
        for mode in payload["manifest"]["prompt_modes"]:
            result = record["results"].get(mode, {})
            lines += [f"### {mode}", "", result.get("final_answer", "尚未完成"), "",
                f"`usage={json.dumps(result.get('usage', {}), ensure_ascii=False)}`  `latency_ms={result.get('latency_ms')}`", ""]
    (directory / "answers.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    seed = config["random_seed"] if args.seed is None else args.seed
    source_path = ROOT / config["source_state"]
    dataset_path = ROOT / config["dataset"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not source.get("complete") or len(source.get("records", [])) != 100:
        raise ValueError("Expected the completed frozen Jina all100 state")
    selected = select_sample(source["records"], config["sample_size"], seed)
    with dataset_path.open(encoding="utf-8-sig", newline="") as stream:
        dataset = {row["financebench_id"]: row for row in csv.DictReader(stream)}
    manifest = {"experiment": config["name"], "config_sha256": sha256(CONFIG),
        "source_state_sha256": sha256(source_path), "dataset_sha256": sha256(dataset_path),
        "seed": seed, "question_ids": [record["financebench_id"] for record in selected],
        "prompt_modes": config["prompt_modes"], "retrieval_calls": 0, "reranker_calls": 0, "judge_calls": 0}
    payload = {"manifest": manifest, "records": [{"question_id": record["financebench_id"],
        "question_type": dataset[record["financebench_id"]]["question_type"], "question": record["question"],
        "reference_answer": dataset[record["financebench_id"]]["answer"],
        "evidence_sha256": hashlib.sha256(record["evidence"].encode()).hexdigest(), "results": {}}
        for record in selected]}
    state_path = args.output_dir / "state.json"
    if state_path.exists():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if payload.get("manifest") != manifest:
            raise ValueError("Smoke checkpoint configuration drift; use a new output directory")

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    from runtime_profile import apply_runtime_profile
    apply_runtime_profile(config["answer"]["profile"])
    os.environ.update({"MODEL": config["answer"]["model"], "ANSWER_TEMPERATURE": str(config["answer"]["temperature"]),
        "ANSWER_MAX_COMPLETION_TOKENS": str(config["answer"]["max_completion_tokens"]),
        "ANSWER_THINKING_MODE": config["answer"]["thinking"], "ANSWER_TIMEOUT_SECONDS": str(config["answer"]["timeout_seconds"]),
        "ANSWER_MAX_RETRIES": str(config["answer"]["max_retries"]), "LANGSMITH_TRACING": "false",
        "LANGSMITH_TRACING_V2": "false", "LANGCHAIN_TRACING_V2": "false"})
    from answer_generator import generate_answer
    frozen = {record["financebench_id"]: record for record in selected}
    write_outputs(args.output_dir, payload)
    for record in payload["records"]:
        evidence = frozen[record["question_id"]]["evidence"]
        if hashlib.sha256(evidence.encode()).hexdigest() != record["evidence_sha256"]:
            raise ValueError("Frozen context drift")
        for mode in config["prompt_modes"]:
            if record["results"].get(mode, {}).get("status") == "ok":
                continue
            started = time.perf_counter()
            answer, usage = generate_answer(record["question"], evidence, [], "", config["answer"]["profile"], mode)
            if not answer.strip():
                raise RuntimeError("Empty answer")
            record["results"][mode] = {"question_id": record["question_id"], "question_type": record["question_type"],
                "prompt_mode": mode, "final_answer": answer, "usage": usage,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2), "status": "ok"}
            write_outputs(args.output_dir, payload)
            print(f"[{record['question_id']}] {mode} ok", flush=True)
    payload["complete"] = True
    write_outputs(args.output_dir, payload)
    print(f"Completed 3x2 generations: {args.output_dir / 'answers.md'}")


if __name__ == "__main__":
    main()
