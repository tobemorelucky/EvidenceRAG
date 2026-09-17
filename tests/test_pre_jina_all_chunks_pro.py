from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.pre_jina_all_chunks_pro_common import (
    ANSWER_MODELS,
    EXPECTED_CHUNKS,
    FLASH_MODEL,
    build_all_chunks_context,
    build_summary,
    prepare_output,
    run_answers,
    select_half,
)


def _chunks() -> list[dict]:
    return [
        {
            "id": index,
            "chunk_id": f"chunk-{index}",
            "filename": "ACME_2025_10K.pdf",
            "page_number": index,
            "rrf_rank": index + 1,
            "text": f"FULL-TEXT-{index}",
        }
        for index in range(EXPECTED_CHUNKS)
    ]


def test_first_and_last_splits_are_disjoint_and_complete():
    items = list(range(100))

    first = select_half(items, "first50")
    last = select_half(items, "last50")

    assert first == list(range(50))
    assert last == list(range(50, 100))
    assert set(first).isdisjoint(last)


def test_context_contains_all_120_chunks_without_truncation():
    context, trace = build_all_chunks_context(_chunks())

    assert trace["chunk_count"] == 120
    assert trace["truncated"] is False
    assert "FULL-TEXT-0" in context
    assert "FULL-TEXT-119" in context
    assert context.count("[RRF Chunk ") == 120
    assert len(trace["chunk_ids"]) == 120


def test_fresh_output_archives_paid_results(tmp_path: Path):
    output = tmp_path / "experiment"
    output.mkdir()
    (output / "answers.jsonl").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (output / "judge_results.jsonl").write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    archive = prepare_output(output, resume=False)

    assert archive is not None
    assert (archive / "answers.jsonl").exists()
    assert (archive / "judge_results.jsonl").exists()
    assert not (output / "answers.jsonl").exists()


def test_resume_preserves_checkpoint(tmp_path: Path):
    output = tmp_path / "experiment"
    output.mkdir()
    checkpoint = output / "answers.jsonl"
    checkpoint.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    assert prepare_output(output, resume=True) is None
    assert checkpoint.exists()


def test_repository_root_is_available_for_shared_judge_import():
    from scripts.pre_jina_all_chunks_pro_common import ROOT

    assert str(ROOT) in sys.path
    from scripts.financebench_judge_common import JUDGE_PROMPT

    assert "Reference answer" in JUDGE_PROMPT


def test_max_new_limits_paid_answer_attempts(tmp_path: Path):
    rows = [
        {"financebench_id": f"id-{index}", "question": f"q-{index}", "answer": "a"}
        for index in range(4)
    ]
    snapshot = {row["financebench_id"]: {"chunks": _chunks()} for row in rows}
    calls: list[str] = []

    def fake_answer(_model, question: str, _context: str):
        calls.append(question)
        return "answer", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    with patch("scripts.pre_jina_all_chunks_pro_common.create_answer_model", return_value=object()), patch(
        "scripts.pre_jina_all_chunks_pro_common.invoke_answer", side_effect=fake_answer
    ):
        records = run_answers(rows, snapshot, tmp_path, "first50", max_new=2)

    assert calls == ["q-0", "q-1"]
    assert [record["financebench_id"] for record in records] == ["id-0", "id-1"]


def test_flash_answer_model_is_recorded_without_changing_pro_judge(tmp_path: Path):
    rows = [{"financebench_id": "id-0", "question": "q", "answer": "a"}]
    snapshot = {"id-0": {"chunks": _chunks()}}

    with patch("scripts.pre_jina_all_chunks_pro_common.create_answer_model", return_value=object()) as factory, patch(
        "scripts.pre_jina_all_chunks_pro_common.invoke_answer",
        return_value=("answer", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
    ):
        answers = run_answers(
            rows,
            snapshot,
            tmp_path,
            "first50",
            answer_model=FLASH_MODEL,
        )

    summary = build_summary("first50", answers, [], answer_model=FLASH_MODEL)
    factory.assert_called_once_with(FLASH_MODEL)
    assert answers[0]["answer_model"] == FLASH_MODEL
    assert summary["answer_model"] == FLASH_MODEL
    assert summary["judge_model"] == "deepseek-v4-pro-ga-260813"


def test_answer_model_cli_is_restricted_to_flash_and_pro():
    assert ANSWER_MODELS == {
        "pro": "deepseek-v4-pro-ga-260813",
        "flash": "deepseek-v4-flash-ga-260731",
    }
