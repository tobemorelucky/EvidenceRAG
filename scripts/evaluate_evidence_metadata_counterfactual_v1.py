"""Evidence Metadata Counterfactual Replay v1 on frozen diagnostic30.

Candidate chunk IDs come from the existing Evidence Block v1 diagnostic JSON.
The script performs read-only PostgreSQL lookups and never reruns retrieval or
calls an external service. Both routes use the same candidate units, ranking
weights, character budget, and packing implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from database import SessionLocal  # noqa: E402
from evidence_assembly_v5 import EvidenceUnit, build_evidence_units  # noqa: E402
from evidence_ranking_v1 import rank_evidence_units_v1, score_evidence_unit, select_ranked_evidence_v1  # noqa: E402
from milvus_client import MilvusManager  # noqa: E402
from models import DocumentPage, DocumentTable, ParentChunk  # noqa: E402
from scripts.audit_evidence_selection_failure_v1 import (  # noqa: E402
    _filename,
    _numbers,
    _periods,
    _required_numbers,
    evidence_coverage,
)


DEFAULT_RANKING = ROOT / "reports" / "evidence_ranking_v1_diagnostic30.json"
DEFAULT_BLOCKS = ROOT / "reports" / "evidence_block_v1_diagnostic30.json"
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_JSON = ROOT / "reports" / "evidence_metadata_counterfactual_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "evidence_metadata_counterfactual_v1.md"
MAX_CONTEXT_CHARS = 28000
GROUPS = ("selection_loss10", "correct_regression10", "candidate_miss10")

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_VALUE_RE = re.compile(r"(?:[$€£¥]\s*)?\(?-?\d[\d,]*(?:\.\d+)?%?\)?")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "their", "this", "to", "was", "were", "what", "when",
    "which", "who", "with", "would",
}


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _field(item: object, name: str, default=None):
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def _terms(value: object) -> list[str]:
    return list(dict.fromkeys(
        token.casefold() for token in _TOKEN_RE.findall(str(value or ""))
        if token.casefold() not in _STOP
    ))


def _years(value: object) -> list[str]:
    return list(dict.fromkeys(_YEAR_RE.findall(str(value or ""))))


def _values(value: object) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in _VALUE_RE.finditer(str(value or ""))))


def _row_values(row: object, columns: list[str]) -> list[str]:
    if isinstance(row, dict):
        ordered = [row.get(column) for column in columns if column in row]
        ordered.extend(value for key, value in row.items() if key not in columns and not str(key).startswith("_"))
        return [_clean(value) for value in ordered if _clean(value)]
    if isinstance(row, list):
        return [_clean(value) for value in row if _clean(value)]
    return [_clean(row)] if _clean(row) else []


def _page_dict(row: DocumentPage) -> dict:
    return {
        "document_id": row.document_id,
        "page_id": row.page_id,
        "filename": row.filename,
        "doc_name": row.doc_name,
        "page_number": row.page_number,
        "company": row.company,
        "report_year": row.report_year,
        "financial_document_type": row.financial_document_type,
        "page_text": row.page_text,
        "table_text": row.table_text,
    }


def _table_dict(row: DocumentTable) -> dict:
    return {
        "table_id": row.table_id,
        "document_id": row.document_id,
        "page_id": row.page_id,
        "filename": row.filename,
        "doc_name": row.doc_name,
        "page_number": row.page_number,
        "title": row.title,
        "caption": row.caption,
        "before_context": row.before_context,
        "after_context": row.after_context,
        "columns": list(row.columns or []),
        "rows": list(row.rows or []),
        "unit": row.unit,
        "scale": row.scale,
        "quality_score": row.quality_score,
    }


def _chunk_dict(row: ParentChunk | dict, page: dict, rank: int) -> dict:
    return {
        "chunk_id": _field(row, "chunk_id", ""),
        "text": _field(row, "text", ""),
        "filename": _field(row, "filename", ""),
        "page_number": int(_field(row, "page_number", 0) or 0),
        "chunk_idx": int(_field(row, "chunk_idx", 0) or 0),
        "chunk_level": int(_field(row, "chunk_level", 3) or 3),
        "document_id": page.get("document_id"),
        "company": page.get("company"),
        "merged_rank": rank,
    }


def _load_dataset(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["financebench_id"]: row for row in csv.DictReader(handle)}


def _gold(row: dict) -> list[dict]:
    return [{
        "filename": _filename(item.get("doc_name")),
        "page_number": int(item.get("evidence_page_num") or 0),
        "evidence_text": str(item.get("evidence_text") or ""),
    } for item in json.loads(row.get("evidence") or "[]")]


def _chunk_ids(block_record: dict) -> list[str]:
    result = []
    for block in (block_record.get("evidence_block_v1") or {}).get("block_score_trace") or []:
        for chunk_id in block.get("source_chunk_ids") or []:
            if chunk_id and chunk_id not in result:
                result.append(chunk_id)
    return result


def _raw_retrieval_rank(trace: dict) -> int:
    score = float((trace.get("ranking_v1_features") or {}).get("retrieval_score") or 0.0)
    if score <= 0:
        raise ValueError("Ranking trace lacks a positive retrieval_score")
    rank = max(1, round(1.0 / score))
    if not math.isclose(score, 1.0 / rank, abs_tol=1e-6):
        raise ValueError(f"Cannot recover original retrieval rank from score={score}")
    return rank


def _feature_distance(expected: dict, actual: dict) -> float:
    return sum(
        abs(float(expected.get(name) or 0.0) - float(actual.get(name) or 0.0))
        for name in ("query_lexical_overlap", "numeric_presence", "period_match", "unit_completeness")
    )


def _temporary_text_unit(question: str, chunk: ParentChunk | dict, page: dict, rank: int) -> dict:
    text = str(_field(chunk, "text", "") or "")
    return EvidenceUnit(
        document_id=str(page.get("document_id") or ""),
        page_id=str(page.get("page_id") or ""),
        source_type="text",
        entity=_clean(page.get("company") or page.get("doc_name") or _field(chunk, "filename", "")),
        period=_years(text),
        metric=None,
        value=_values(text) or None,
        unit=None,
        source_text=text,
        metadata={
            "filename": _field(chunk, "filename", ""),
            "page_number": int(_field(chunk, "page_number", 0) or 0),
            "chunk_id": _field(chunk, "chunk_id", ""),
            "retrieval_rank": rank,
        },
    ).to_dict()


def reconstruct_frozen_chunks(
    question: str,
    rank_trace: list[dict],
    selected_units: list[dict],
    chunk_ids: list[str],
    chunks_by_id: dict[str, ParentChunk | dict],
    pages_by_id: dict[str, dict],
) -> tuple[list[dict], dict]:
    """Recover the exact Top120 chunk set and deterministic retrieval order."""
    text_targets = [item for item in rank_trace if item.get("source_type") == "text"]
    target_by_raw_rank = {_raw_retrieval_rank(item): item for item in text_targets}
    if len(target_by_raw_rank) != len(text_targets):
        raise RuntimeError("Text trace retrieval ranks are not unique")
    selected_chunk_by_raw_rank = {}
    for unit in selected_units:
        if unit.get("source_type") != "text":
            continue
        chunk_id = str((unit.get("metadata") or {}).get("chunk_id") or "")
        raw_rank = int((unit.get("metadata") or {}).get("retrieval_rank") or 0)
        if chunk_id and raw_rank:
            selected_chunk_by_raw_rank[raw_rank] = chunk_id

    page_id_by_key = {
        (_filename(page.get("filename")), int(page.get("page_number") or 0)): page_id
        for page_id, page in pages_by_id.items()
    }
    candidates_by_page: dict[str, list[ParentChunk | dict]] = defaultdict(list)
    for chunk_id in chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        page_id = page_id_by_key.get((
            _filename(_field(chunk, "filename", "")), int(_field(chunk, "page_number", 0) or 0)
        ))
        if page_id:
            candidates_by_page[page_id].append(chunk)

    assigned: dict[int, ParentChunk | dict] = {}
    used_ids = set()
    for raw_rank, chunk_id in selected_chunk_by_raw_rank.items():
        chunk = chunks_by_id.get(chunk_id)
        if chunk is not None:
            assigned[raw_rank] = chunk
            used_ids.add(str(_field(chunk, "chunk_id", "")))

    matching_distances = []
    for raw_rank in sorted(target_by_raw_rank):
        if raw_rank in assigned:
            continue
        target = target_by_raw_rank[raw_rank]
        page_id = str(target.get("page_id") or "")
        page = pages_by_id.get(page_id) or {}
        choices = [
            item for item in candidates_by_page.get(page_id, [])
            if str(_field(item, "chunk_id", "")) not in used_ids
        ]
        if not choices:
            raise RuntimeError(f"No frozen chunk remains for page={page_id}, raw_rank={raw_rank}")
        scored = []
        for chunk in choices:
            unit = _temporary_text_unit(question, chunk, page, raw_rank)
            features = score_evidence_unit(question, unit)
            scored.append((
                _feature_distance(target.get("ranking_v1_features") or {}, features),
                int(_field(chunk, "chunk_idx", 0) or 0),
                str(_field(chunk, "chunk_id", "")),
                chunk,
            ))
        distance, _, _, chosen = min(scored)
        assigned[raw_rank] = chosen
        used_ids.add(str(_field(chosen, "chunk_id", "")))
        matching_distances.append(distance)

    if len(assigned) != len(text_targets) or len(used_ids) != len(chunk_ids):
        raise RuntimeError(
            f"Frozen chunk reconstruction mismatch assigned={len(assigned)} trace={len(text_targets)} "
            f"used={len(used_ids)} ids={len(chunk_ids)}"
        )
    chunks = []
    for rank in sorted(assigned):
        target = target_by_raw_rank[rank]
        page = pages_by_id.get(str(target.get("page_id") or "")) or {}
        chunks.append(_chunk_dict(assigned[rank], page, rank))
    return chunks, {
        "frozen_chunk_count": len(chunk_ids),
        "reconstructed_chunk_count": len(chunks),
        "selected_exact_chunk_assignments": len(selected_chunk_by_raw_rank),
        "feature_matched_chunk_assignments": len(chunks) - len(selected_chunk_by_raw_rank),
        "average_feature_assignment_distance": round(statistics.fmean(matching_distances), 8) if matching_distances else 0.0,
        "max_feature_assignment_distance": round(max(matching_distances), 8) if matching_distances else 0.0,
    }


def _nearby_texts(units: list[dict]) -> dict[str, str]:
    by_page: dict[str, list[str]] = defaultdict(list)
    for unit in units:
        if unit.get("source_type") == "text":
            by_page[str(unit.get("page_id") or "")].append(str(unit.get("source_text") or ""))
    return {page_id: "\n".join(texts) for page_id, texts in by_page.items()}


def extract_counterfactual_metadata(
    question: str,
    unit: dict,
    page: dict | None,
    *,
    nearby_text: str = "",
) -> tuple[dict, dict]:
    """Re-extract generic metadata with a source-priority contract."""
    page = page or {}
    source_text = str(unit.get("source_text") or "")
    source_type = str(unit.get("source_type") or "")
    header = "\n".join(
        line for line in source_text.splitlines() if line.casefold().startswith(("header:", "table title:"))
    )
    page_lines = [line.strip() for line in str(page.get("page_text") or "").splitlines() if line.strip()]
    page_title = "\n".join(page_lines[:4])[:600]
    period_sources = {
        "table_header": _years(header) if source_type == "table" else [],
        "local_text": _years(source_text),
        "page_title": _years(page_title),
        "nearby_chunk": _years(nearby_text),
        "document_metadata": [str(page.get("report_year"))] if int(page.get("report_year") or 0) else [],
    }
    priority = ("table_header", "local_text", "page_title", "nearby_chunk", "document_metadata")
    period = []
    applied_period_sources = []
    for source in priority:
        values = period_sources[source]
        if values:
            period = list(dict.fromkeys(values))
            applied_period_sources.append(source)
            break

    entity_sources = {
        "document_metadata": _clean(page.get("company")),
        "page_text": "",
        "nearby_chunk": "",
        "existing_unit": _clean(unit.get("entity")),
    }
    entity = entity_sources["document_metadata"] or entity_sources["existing_unit"]
    entity_source = "document_metadata" if entity_sources["document_metadata"] else "existing_unit"

    current_metric = _clean(unit.get("metric"))
    query_terms = _terms(question)
    local_terms = set(_terms(source_text))
    lexical_metric = " ".join(term for term in query_terms if term in local_terms)
    if source_type == "table" and current_metric:
        metric, metric_source = current_metric, "table_row_label"
    elif current_metric:
        metric, metric_source = current_metric, "chunk_section"
    elif lexical_metric:
        metric, metric_source = lexical_metric, "chunk_lexical_match"
    else:
        metric, metric_source = None, "missing"

    corrected = {
        **unit,
        "entity": entity,
        "period": period,
        "metric": metric,
        "metadata": {**(unit.get("metadata") or {}), "counterfactual_metadata": True},
    }
    trace = {
        "entity": {"value": entity, "source": entity_source, "confidence": 1.0 if entity_source == "document_metadata" else 0.5, "candidates": entity_sources},
        "period": {
            "value": period,
            "source": applied_period_sources[0] if applied_period_sources else "missing",
            "confidence": {"table_header": 1.0, "local_text": 0.95, "page_title": 0.8, "nearby_chunk": 0.6, "document_metadata": 0.45}.get(applied_period_sources[0], 0.0) if applied_period_sources else 0.0,
            "candidates": period_sources,
        },
        "metric": {"value": metric, "source": metric_source, "confidence": 0.9 if metric_source in {"table_row_label", "chunk_section"} else (0.6 if metric_source == "chunk_lexical_match" else 0.0)},
    }
    return corrected, trace


def _rank_trace_match(rebuilt: list[dict], frozen: list[dict]) -> dict:
    comparable = min(len(rebuilt), len(frozen))
    exact = 0
    score_matches = 0
    for left, right in zip(rebuilt, frozen):
        if left.get("source_type") == right.get("source_type") and left.get("page_id") == right.get("page_id"):
            exact += 1
        if math.isclose(float(left.get("ranking_v1_score") or 0), float(right.get("ranking_v1_score") or 0), abs_tol=1e-7):
            score_matches += 1
    return {
        "rebuilt_candidate_count": len(rebuilt),
        "frozen_candidate_count": len(frozen),
        "rank_page_type_match_rate": round(exact / max(1, comparable), 4),
        "rank_score_match_rate": round(score_matches / max(1, comparable), 4),
        "valid": len(rebuilt) == len(frozen) and exact / max(1, comparable) >= 0.99 and score_matches / max(1, comparable) >= 0.99,
    }


def _selected_unit_match(rebuilt: list[dict], frozen: list[dict]) -> dict:
    comparable = min(len(rebuilt), len(frozen))
    exact = 0
    for left, right in zip(rebuilt, frozen):
        left_meta = left.get("metadata") or {}
        right_meta = right.get("metadata") or {}
        identity = (
            left.get("source_type") == right.get("source_type")
            and left.get("page_id") == right.get("page_id")
            and left_meta.get("chunk_id") == right_meta.get("chunk_id")
            and left_meta.get("table_id") == right_meta.get("table_id")
            and left_meta.get("row_index") == right_meta.get("row_index")
            and str(left.get("source_text") or "") == str(right.get("source_text") or "")
        )
        exact += int(identity)
    rate = exact / max(1, comparable)
    return {
        "rebuilt_selected_unit_count": len(rebuilt),
        "frozen_selected_unit_count": len(frozen),
        "selected_identity_text_match_rate": round(rate, 4),
        "selected_units_valid": len(rebuilt) == len(frozen) and rate >= 0.99,
    }


def _contains_all(required: list[str], evidence: str, extractor) -> bool | None:
    if not required:
        return None
    return set(required) <= set(extractor(evidence))


def _context_metrics(row: dict, context: str, units: list[dict]) -> dict:
    gold = _gold(row)
    gold_pages = {(item["filename"], item["page_number"]) for item in gold}
    selected_pages = {
        (_filename((unit.get("metadata") or {}).get("filename")), int((unit.get("metadata") or {}).get("page_number") or 0))
        for unit in units
    }
    required_numbers = _required_numbers(row)
    required_periods = _periods(row.get("question"))
    return {
        "selected_gold_page_hit": bool(gold_pages & selected_pages),
        "answer_evidence_coverage": evidence_coverage(gold, context),
        "required_numbers": required_numbers,
        "required_number_hit": _contains_all(required_numbers, context, _numbers),
        "required_periods": required_periods,
        "required_period_hit": _contains_all(required_periods, context, _periods),
        "context_chars": len(context),
        "selected_unit_count": len(units),
    }


def _gold_retention(gold: list[dict], candidate_context: str, selected_context: str) -> dict:
    candidate = evidence_coverage(gold, candidate_context)
    selected = evidence_coverage(gold, selected_context)
    denominator = int(candidate["matched_lines"] or 0)
    return {
        "candidate_matched_lines": denominator,
        "selected_matched_lines": int(selected["matched_lines"] or 0),
        "ratio": round(int(selected["matched_lines"] or 0) / denominator, 4) if denominator else None,
    }


def _mean(values: list[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.fmean(usable), 4) if usable else None


def _rate(values: list[bool | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return round(sum(bool(value) for value in usable) / len(usable), 4) if usable else None


def _route_summary(records: list[dict], route: str) -> dict:
    values = [record["routes"][route] for record in records]
    return {
        "selected_gold_page_hit": _rate([value["metrics"]["selected_gold_page_hit"] for value in values]),
        "gold_evidence_retention": _mean([value["gold_evidence_retention"]["ratio"] for value in values]),
        "evidence_coverage": _mean([value["metrics"]["answer_evidence_coverage"]["ratio"] for value in values]),
        "required_number_hit": _rate([value["metrics"]["required_number_hit"] for value in values]),
        "required_period_hit": _rate([value["metrics"]["required_period_hit"] for value in values]),
        "average_context_chars": _mean([value["metrics"]["context_chars"] for value in values]),
    }


def _summary_for(records: list[dict]) -> dict:
    current = _route_summary(records, "current_ranking")
    counter = _route_summary(records, "counterfactual_metadata_ranking")
    metrics = ("selected_gold_page_hit", "gold_evidence_retention", "evidence_coverage", "required_number_hit", "required_period_hit")
    return {
        "questions": len(records),
        "current_ranking": current,
        "counterfactual_metadata_ranking": counter,
        "delta": {metric: round((counter[metric] or 0.0) - (current[metric] or 0.0), 4) for metric in metrics},
        "coverage_gains": [record["financebench_id"] for record in records if (
            record["routes"]["counterfactual_metadata_ranking"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0
        ) > (
            record["routes"]["current_ranking"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0
        )],
        "coverage_regressions": [record["financebench_id"] for record in records if (
            record["routes"]["counterfactual_metadata_ranking"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0
        ) < (
            record["routes"]["current_ranking"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0
        )],
    }


def summarize(records: list[dict]) -> dict:
    overall = _summary_for(records)
    overall["frozen_top120_chunks"] = sum(
        record["replay_validation"]["frozen_chunk_count"] for record in records
    )
    overall["saved_candidate_evidence_units"] = sum(record["candidate_unit_count"] for record in records)
    overall["groups"] = {
        group: _summary_for([record for record in records if record["group"] == group])
        for group in GROUPS
    }
    validation = [record["replay_validation"] for record in records]
    overall["replay_validation"] = {
        "valid_questions": sum(item["valid"] for item in validation),
        "all_valid": all(item["valid"] for item in validation),
        "average_rank_page_type_match_rate": _mean([item["rank_page_type_match_rate"] for item in validation]),
        "average_rank_score_match_rate": _mean([item["rank_score_match_rate"] for item in validation]),
        "average_selected_identity_text_match_rate": _mean([
            item["selected_identity_text_match_rate"] for item in validation
        ]),
    }
    selection = overall["groups"]["selection_loss10"]
    correct = overall["groups"]["correct_regression10"]
    worth_it = bool(
        overall["replay_validation"]["all_valid"]
        and selection["delta"]["evidence_coverage"] >= 0.02
        and selection["delta"]["selected_gold_page_hit"] >= 0.05
        and correct["delta"]["evidence_coverage"] >= 0
        and overall["delta"]["required_number_hit"] >= 0
        and overall["delta"]["required_period_hit"] >= 0
    )
    shadow_signal = bool(
        overall["replay_validation"]["all_valid"]
        and selection["delta"]["selected_gold_page_hit"] >= 0.05
    )
    overall["decision"] = {
        "metadata_worth_production_integration": worth_it,
        "metadata_signal_worth_shadow_followup": shadow_signal,
        "criterion": "valid replay; selection-loss coverage improves >=2pp and selected gold-page hit >=5pp; correct-regression and required number/period do not regress",
        "next_focus": "metadata" if worth_it else "packing",
        "reason": (
            "Metadata produces material evidence-level gains without regression."
            if worth_it else
            "Gold-page selection moves, but evidence coverage and required number/period do not improve materially; packing remains the stronger direct bottleneck."
        ),
    }
    overall["external_calls"] = {
        "retrieval": 0,
        "milvus_exact_id_storage_lookup": 1,
        "llm": 0,
        "jina": 0,
        "judge": 0,
        "langsmith": 0,
    }
    return overall


def _load_db_data(chunk_ids: set[str], page_ids: set[str]) -> tuple[dict, dict, list[dict]]:
    db = SessionLocal()
    try:
        chunks = db.query(ParentChunk).filter(ParentChunk.chunk_id.in_(chunk_ids)).all()
        pages = db.query(DocumentPage).filter(DocumentPage.page_id.in_(page_ids)).all()
        tables = db.query(DocumentTable).filter(DocumentTable.page_id.in_(page_ids)).all()
        chunks_by_id: dict[str, ParentChunk | dict] = {item.chunk_id: item for item in chunks}
        missing = sorted(chunk_ids - set(chunks_by_id))
        if missing:
            manager = MilvusManager()
            for start in range(0, len(missing), 200):
                for item in manager.get_chunks_by_ids(missing[start:start + 200]):
                    chunk_id = str(item.get("chunk_id") or "")
                    if chunk_id:
                        chunks_by_id[chunk_id] = item
        unresolved = sorted(chunk_ids - set(chunks_by_id))
        if unresolved:
            raise RuntimeError(f"Frozen chunk IDs unavailable in local stores: {unresolved[:5]} ({len(unresolved)} total)")
        return (chunks_by_id, {item.page_id: _page_dict(item) for item in pages}, [_table_dict(item) for item in tables])
    finally:
        db.close()


def evaluate_record(record: dict, block_record: dict, row: dict, db_data: tuple[dict, dict, list[dict]]) -> dict:
    route = record["routes"]["evidence_ranking_v1"]
    frozen_trace = list((route.get("trace") or {}).get("rank_trace") or [])
    selected_units = list(route.get("selected_units") or [])
    ids = _chunk_ids(block_record)
    chunks_by_id, all_pages, all_tables = db_data
    page_ids = {str(item.get("page_id") or "") for item in frozen_trace}
    pages = {page_id: all_pages[page_id] for page_id in page_ids if page_id in all_pages}
    chunks, chunk_validation = reconstruct_frozen_chunks(
        record["question"], frozen_trace, selected_units, ids, chunks_by_id, pages,
    )
    page_keys = {(_filename(chunk.get("filename")), int(chunk.get("page_number") or 0)) for chunk in chunks}
    relevant_pages = [page for page in pages.values() if (_filename(page.get("filename")), int(page.get("page_number") or 0)) in page_keys]
    relevant_tables = [table for table in all_tables if table.get("page_id") in {page.get("page_id") for page in relevant_pages}]
    current_candidates = [unit.to_dict() for unit in build_evidence_units(
        record["question"], chunks, pages=relevant_pages, tables=relevant_tables,
    )]
    current_ranked = rank_evidence_units_v1(record["question"], current_candidates)
    trace_validation = _rank_trace_match(current_ranked, frozen_trace)
    replay_validation = {**chunk_validation, **trace_validation}

    current_context, current_selected, _ = select_ranked_evidence_v1(
        record["question"], current_candidates, max_context_chars=MAX_CONTEXT_CHARS,
    )
    selected_validation = _selected_unit_match(current_selected, selected_units)
    replay_validation.update(selected_validation)
    replay_validation["valid"] = bool(replay_validation["valid"] and selected_validation["selected_units_valid"])
    nearby = _nearby_texts(current_candidates)
    counter_candidates = []
    metadata_traces = []
    for unit in current_candidates:
        corrected, metadata_trace = extract_counterfactual_metadata(
            record["question"], unit, pages.get(str(unit.get("page_id") or "")),
            nearby_text=nearby.get(str(unit.get("page_id") or ""), ""),
        )
        counter_candidates.append(corrected)
        metadata_traces.append(metadata_trace)
    counter_context, counter_selected, _ = select_ranked_evidence_v1(
        record["question"], counter_candidates, max_context_chars=MAX_CONTEXT_CHARS,
    )
    counter_ranked = rank_evidence_units_v1(record["question"], counter_candidates)
    current_rank_by_key = {
        (item.get("source_type"), item.get("page_id"), (item.get("metadata") or {}).get("chunk_id"), (item.get("metadata") or {}).get("table_id"), (item.get("metadata") or {}).get("row_index")): item
        for item in current_ranked
    }
    counter_rank_by_key = {
        (item.get("source_type"), item.get("page_id"), (item.get("metadata") or {}).get("chunk_id"), (item.get("metadata") or {}).get("table_id"), (item.get("metadata") or {}).get("row_index")): item
        for item in counter_ranked
    }
    selected_counter_keys = {
        (item.get("source_type"), item.get("page_id"), (item.get("metadata") or {}).get("chunk_id"), (item.get("metadata") or {}).get("table_id"), (item.get("metadata") or {}).get("row_index"))
        for item in counter_selected
    }
    saved_candidates = []
    for unit, metadata_trace in zip(current_candidates, metadata_traces):
        key = (unit.get("source_type"), unit.get("page_id"), (unit.get("metadata") or {}).get("chunk_id"), (unit.get("metadata") or {}).get("table_id"), (unit.get("metadata") or {}).get("row_index"))
        old = current_rank_by_key[key]
        new = counter_rank_by_key[key]
        saved_candidates.append({
            **unit,
            "current_ranking": {"rank": old["ranking_v1_rank"], "score": old["ranking_v1_score"], "features": old["ranking_v1_features"]},
            "counterfactual_metadata": metadata_trace,
            "counterfactual_ranking": {"rank": new["ranking_v1_rank"], "score": new["ranking_v1_score"], "features": new["ranking_v1_features"], "selected": key in selected_counter_keys},
        })
    candidate_context = "\n\n".join(unit.get("source_text") or "" for unit in current_candidates)
    gold = _gold(row)
    return {
        "financebench_id": record["financebench_id"],
        "group": record["group"],
        "question": record["question"],
        "replay_validation": replay_validation,
        "candidate_unit_count": len(saved_candidates),
        "candidate_units": saved_candidates,
        "routes": {
            "current_ranking": {
                "metrics": _context_metrics(row, current_context, current_selected),
                "gold_evidence_retention": _gold_retention(gold, candidate_context, current_context),
                "selected_unit_ranks": [item["ranking_v1_rank"] for item in current_selected],
            },
            "counterfactual_metadata_ranking": {
                "metrics": _context_metrics(row, counter_context, counter_selected),
                "gold_evidence_retention": _gold_retention(gold, candidate_context, counter_context),
                "selected_unit_ranks": [item["ranking_v1_rank"] for item in counter_selected],
            },
        },
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evidence Metadata Counterfactual Replay v1",
        "",
        "> Frozen Top120 chunk IDs and unchanged candidate set, Ranking v1 weights, 28k context budget, and packing. Retrieval=0; Milvus is accessed only by exact frozen chunk ID. LLM=0, Jina=0, Judge=0, LangSmith=0.",
        "",
        "## Replay validity",
        "",
        f"- Valid questions: {summary['replay_validation']['valid_questions']}/30",
        f"- Frozen Top120 chunks saved/replayed: {summary['frozen_top120_chunks']} (120 per question)",
        f"- Candidate Evidence Units saved: {summary['saved_candidate_evidence_units']}",
        f"- Average frozen rank page/type match: {_pct(summary['replay_validation']['average_rank_page_type_match_rate'])}",
        f"- Average frozen rank score match: {_pct(summary['replay_validation']['average_rank_score_match_rate'])}",
        f"- Average selected unit identity/text match: {_pct(summary['replay_validation']['average_selected_identity_text_match_rate'])}",
        "",
        "A result is eligible for a production recommendation only when every question reproduces at least 99% of frozen rank/page/type and score entries.",
        "",
        "## A/B summary",
        "",
        "| Group | Route | Selected gold page | Gold retention | Evidence coverage | Number hit | Period hit | Avg chars |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("overall", *GROUPS):
        item = summary if group == "overall" else summary["groups"][group]
        for route in ("current_ranking", "counterfactual_metadata_ranking"):
            value = item[route]
            lines.append(
                f"| {group} | {route} | {_pct(value['selected_gold_page_hit'])} | "
                f"{_pct(value['gold_evidence_retention'])} | {_pct(value['evidence_coverage'])} | "
                f"{_pct(value['required_number_hit'])} | {_pct(value['required_period_hit'])} | "
                f"{value['average_context_chars']} |"
            )
    decision = summary["decision"]
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Metadata worth production integration: `{decision['metadata_worth_production_integration']}`",
        f"- Metadata signal worth another shadow follow-up: `{decision['metadata_signal_worth_shadow_followup']}`",
        f"- Next focus: **{decision['next_focus']}**",
        f"- Criterion: {decision['criterion']}",
        f"- Reason: {decision['reason']}",
        "",
        "## Per question",
        "",
    ])
    for index, record in enumerate(payload["records"], 1):
        current = record["routes"]["current_ranking"]
        counter = record["routes"]["counterfactual_metadata_ranking"]
        lines.extend([
            f"### {index}. {record['financebench_id']} — {record['group']}",
            "",
            f"- Question: {record['question']}",
            f"- Replay valid: `{record['replay_validation']['valid']}`; candidates: `{record['candidate_unit_count']}`",
            f"- Rank page/type match / score match: `{record['replay_validation']['rank_page_type_match_rate']}` / `{record['replay_validation']['rank_score_match_rate']}`",
            f"- Selected gold page current/counterfactual: `{current['metrics']['selected_gold_page_hit']}` / `{counter['metrics']['selected_gold_page_hit']}`",
            f"- Evidence coverage current/counterfactual: `{current['metrics']['answer_evidence_coverage']['ratio']}` / `{counter['metrics']['answer_evidence_coverage']['ratio']}`",
            f"- Gold retention current/counterfactual: `{current['gold_evidence_retention']['ratio']}` / `{counter['gold_evidence_retention']['ratio']}`",
            f"- Required number current/counterfactual: `{current['metrics']['required_number_hit']}` / `{counter['metrics']['required_number_hit']}`",
            f"- Required period current/counterfactual: `{current['metrics']['required_period_hit']}` / `{counter['metrics']['required_period_hit']}`",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-json", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--block-json", type=Path, default=DEFAULT_BLOCKS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    ranking = json.loads(args.ranking_json.read_text(encoding="utf-8"))
    blocks = json.loads(args.block_json.read_text(encoding="utf-8"))
    rows = _load_dataset(args.dataset)
    blocks_by_id = {item["financebench_id"]: item for item in blocks.get("records") or []}
    all_chunk_ids = {
        chunk_id for record in blocks_by_id.values() for chunk_id in _chunk_ids(record)
    }
    all_page_ids = {
        str(item.get("page_id") or "")
        for record in ranking.get("records") or []
        for item in ((record["routes"]["evidence_ranking_v1"].get("trace") or {}).get("rank_trace") or [])
        if item.get("page_id")
    }
    db_data = _load_db_data(all_chunk_ids, all_page_ids)
    records = []
    for index, record in enumerate(ranking.get("records") or [], 1):
        financebench_id = record["financebench_id"]
        if financebench_id not in blocks_by_id or financebench_id not in rows:
            raise RuntimeError(f"Frozen source missing for {financebench_id}")
        result = evaluate_record(record, blocks_by_id[financebench_id], rows[financebench_id], db_data)
        records.append(result)
        print(
            f"[{index:02d}/30] {financebench_id} valid={result['replay_validation']['valid']} "
            f"coverage={result['routes']['current_ranking']['metrics']['answer_evidence_coverage']['ratio']}->"
            f"{result['routes']['counterfactual_metadata_ranking']['metrics']['answer_evidence_coverage']['ratio']}",
            flush=True,
        )
    payload = {
        "evaluation": "evidence_metadata_counterfactual_replay_v1",
        "scope": "frozen Top120 IDs; same candidates, ranking weights, 28k budget, and packing",
        "summary": summarize(records),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {args.output_json}\nMarkdown: {args.output_markdown}", flush=True)


if __name__ == "__main__":
    main()
