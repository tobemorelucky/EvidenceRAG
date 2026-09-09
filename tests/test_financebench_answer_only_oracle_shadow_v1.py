from __future__ import annotations

import csv
import builtins
import json
from pathlib import Path

import pytest

from scripts.run_financebench_answer_only_oracle_shadow_v1 import (
    build_oracle_context,
    build_result_record,
    load_dataset,
    validate_frozen_final100,
    validate_results,
)


def _row(index: int) -> dict[str, str]:
    question_id = f"financebench_id_{index:05d}"
    evidence = [
        {
            "doc_name": "ACME_2025_10K",
            "evidence_page_num": 7,
            "evidence_text": "Revenue was $125 million in 2025.",
            "evidence_text_full_page": "A longer page containing Revenue was $125 million in 2025.",
        }
    ]
    return {
        "financebench_id": question_id,
        "question": "What was revenue in 2025?",
        "answer": "$125 million",
        "evidence": json.dumps(evidence),
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_loads_exactly_100_unique_questions_and_nonempty_gold_evidence(tmp_path: Path):
    dataset = tmp_path / "financebench.csv"
    rows = [_row(index) for index in range(100)]
    with dataset.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_dataset(dataset)

    assert len(loaded) == 100
    assert len({row["financebench_id"] for row in loaded}) == 100
    assert "Revenue was $125 million" in build_oracle_context(loaded[0])
    assert "A longer page" not in build_oracle_context(loaded[0])


def test_validates_frozen_artifacts_without_importing_or_calling_retrieval(tmp_path: Path, monkeypatch):
    records = [{"question_id": f"financebench_id_{index:05d}"} for index in range(100)]
    (tmp_path / "retrieval_snapshot.json").write_text(
        json.dumps({"records": records}), encoding="utf-8"
    )
    (tmp_path / "jina_results.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    _write_jsonl(tmp_path / "answers.jsonl", records)
    _write_jsonl(tmp_path / "judge_results.jsonl", records)

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"embedding", "milvus_client", "query_rewriter", "shadow_rerankers_v1"}:
            raise AssertionError("retrieval must not be imported or called")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    manifest = validate_frozen_final100(tmp_path)

    assert len(manifest["question_ids"]) == 100
    assert all(item["records"] == 100 for item in manifest["artifacts"].values())


def test_result_fields_are_complete_for_all_100_questions():
    rows = [_row(index) for index in range(100)]
    records = [
        build_result_record(
            row,
            build_oracle_context(row),
            "$125 million [source: ACME_2025_10K.pdf, page 7]",
            0.25,
            {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
            "other",
        )
        for row in rows
    ]

    validate_results(rows, records)
    required = {
        "financebench_id", "question", "gold_answer", "oracle_context",
        "model_answer", "latency", "token_usage",
    }
    assert all(required <= set(record) for record in records)


def test_empty_gold_evidence_is_rejected():
    row = _row(1)
    row["evidence"] = "[]"
    with pytest.raises(ValueError, match="Gold evidence is empty"):
        build_oracle_context(row)
