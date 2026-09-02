"""Build and audit cross-page table evidence groups without changing runtime RAG.

This module is intentionally offline-only.  It reads ``document_tables`` and the
frozen diagnostic30 report, then writes analysis artifacts.  It does not mutate
PostgreSQL or any retrieval collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from database import SessionLocal  # noqa: E402
from models import DocumentTable  # noqa: E402


DEFAULT_DIAGNOSTIC = ROOT / "reports" / "table_aware_retrieval_v1_diagnostic30.json"
DEFAULT_JSON = ROOT / "reports" / "table_evidence_groups_v1.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "table_evidence_groups_v1.md"

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
CONTINUATION_RE = re.compile(
    r"\b(?:continued|continuation|cont[.']?d|continued\s+from|continued\s+on|"
    r"following\s+page|preceding\s+page|next\s+page|previous\s+page)\b",
    re.IGNORECASE,
)
FORWARD_RE = re.compile(r"\b(?:continued\s+on|continues?\s+on|next\s+page|following\s+page)\b", re.I)
BACKWARD_RE = re.compile(r"\b(?:continued\s+from|previous\s+page|preceding\s+page|cont[.']?d|continued)\b", re.I)
STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or",
    "the", "to", "table", "unaudited", "continued", "continuation", "page",
    "consolidated", "financial", "statement", "statements", "notes", "note",
}


@dataclass(frozen=True)
class TableRecord:
    table_id: str
    document_id: str
    page_id: str
    filename: str
    page_number: int
    start_page: int
    end_page: int
    title: str
    caption: str
    columns: tuple[str, ...]
    before_context: str
    after_context: str
    quality_score: float


@dataclass(frozen=True)
class GroupLink:
    left_table_id: str
    right_table_id: str
    left_page: int
    right_page: int
    page_id_contiguous: bool
    title_similarity: float
    header_similarity: float
    nearby_similarity: float
    continuation: bool
    score: float
    reasons: tuple[str, ...]


def _text(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "")).strip()


def _flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            result.extend((_text(key), *_flatten_strings(item)))
        return [item for item in result if item]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten_strings(item))
        return result
    text = _text(value)
    return [text] if text else []


def _tokens(value: Any) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN_RE.findall(" ".join(_flatten_strings(value)))
        if len(token) > 1 and not token.isdigit() and token.casefold() not in STOPWORDS
    }


def _jaccard(left: Any, right: Any) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _title_similarity(left: TableRecord, right: TableRecord) -> float:
    a = _text(left.title or left.caption).casefold()
    b = _text(right.title or right.caption).casefold()
    if not a or not b:
        return 0.0
    token_similarity = _jaccard(a, b)
    # Sequence similarity is useful for OCR spelling variation, but common SEC
    # boilerplate must not make semantically different table titles look equal.
    sequence_similarity = SequenceMatcher(None, a, b).ratio()
    overlap = len(_tokens(a) & _tokens(b))
    return max(token_similarity, sequence_similarity * min(1.0, overlap / 2.0))


def _continuation_text(table: TableRecord) -> str:
    return " ".join((table.title, table.caption, table.before_context, table.after_context))


def score_link(left: TableRecord, right: TableRecord) -> GroupLink | None:
    """Score an adjacent-page table pair; high score is not itself runtime evidence."""
    if not left.document_id or left.document_id != right.document_id:
        return None
    if not left.page_id or not right.page_id:
        return None
    if right.page_number != left.page_number + 1:
        return None
    title = _title_similarity(left, right)
    header = _jaccard(left.columns, right.columns)
    nearby = _jaccard(left.after_context, right.before_context)
    continuation = bool(CONTINUATION_RE.search(_continuation_text(left) + " " + _continuation_text(right)))
    score = 0.35 * title + 0.35 * header + 0.15 * nearby + 0.15 * float(continuation)
    reasons = []
    if title >= 0.55:
        reasons.append("title")
    if header >= 0.50:
        reasons.append("header")
    if nearby >= 0.25:
        reasons.append("nearby_text")
    if continuation:
        reasons.append("continuation")
    # A shared year-only header is deliberately ignored by token normalization.
    # Without an explicit continuation cue, demand near-identical titles or a
    # corroborating boundary-text match; adjacent but different statements must
    # not be joined merely because their columns have the same shape.
    strong_pair = title >= 0.90 and header >= 0.50
    continuation_pair = continuation and (title >= 0.35 or header >= 0.50)
    context_pair = nearby >= 0.40 and title >= 0.55 and header >= 0.35
    accepted = score >= 0.48 and (strong_pair or continuation_pair or context_pair)
    if not accepted:
        return None
    return GroupLink(
        left_table_id=left.table_id,
        right_table_id=right.table_id,
        left_page=left.page_number,
        right_page=right.page_number,
        page_id_contiguous=True,
        title_similarity=round(title, 4),
        header_similarity=round(header, 4),
        nearby_similarity=round(nearby, 4),
        continuation=continuation,
        score=round(score, 4),
        reasons=tuple(reasons),
    )


class _UnionFind:
    def __init__(self, ids: Iterable[str]):
        self.parent = {item: item for item in ids}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def build_groups(tables: list[TableRecord]) -> list[dict]:
    """Build conservative one-to-one adjacent-page chains within each document."""
    table_by_id = {table.table_id: table for table in tables}
    union = _UnionFind(table_by_id)
    accepted_links: list[GroupLink] = []
    by_document_page: dict[tuple[str, int], list[TableRecord]] = defaultdict(list)
    for table in tables:
        by_document_page[(table.document_id, table.page_number)].append(table)

    documents = defaultdict(set)
    for document_id, page in by_document_page:
        documents[document_id].add(page)
    for document_id, pages in documents.items():
        for page in sorted(pages):
            left = by_document_page[(document_id, page)]
            right = by_document_page.get((document_id, page + 1), [])
            candidates = [link for a in left for b in right if (link := score_link(a, b))]
            used_left: set[str] = set()
            used_right: set[str] = set()
            for link in sorted(candidates, key=lambda item: item.score, reverse=True):
                if link.left_table_id in used_left or link.right_table_id in used_right:
                    continue
                used_left.add(link.left_table_id)
                used_right.add(link.right_table_id)
                accepted_links.append(link)
                union.union(link.left_table_id, link.right_table_id)

    members: dict[str, list[TableRecord]] = defaultdict(list)
    for table in tables:
        members[union.find(table.table_id)].append(table)
    links_by_root: dict[str, list[GroupLink]] = defaultdict(list)
    for link in accepted_links:
        links_by_root[union.find(link.left_table_id)].append(link)

    groups = []
    for group_tables in members.values():
        group_tables.sort(key=lambda item: (item.page_number, item.table_id))
        document_id = group_tables[0].document_id
        member_pages = sorted({item.page_number for item in group_tables})
        declared_pages = sorted({
            page
            for item in group_tables
            for page in range(min(item.start_page, item.end_page), max(item.start_page, item.end_page) + 1)
            if page >= 0
        } | set(member_pages))
        inferred_boundary_pages: set[int] = set()
        for item in group_tables:
            title_before = " ".join((item.title, item.caption, item.before_context))
            title_after = " ".join((item.title, item.caption, item.after_context))
            if BACKWARD_RE.search(title_before) and item.page_number > 0:
                inferred_boundary_pages.add(item.page_number - 1)
            if FORWARD_RE.search(title_after):
                inferred_boundary_pages.add(item.page_number + 1)
        digest_source = document_id + "\0" + "\0".join(item.table_id for item in group_tables)
        group_id = "tg_" + hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:20]
        groups.append({
            "table_group_id": group_id,
            "document_id": document_id,
            "filename": group_tables[0].filename,
            "table_ids": [item.table_id for item in group_tables],
            "page_ids": [item.page_id for item in group_tables],
            "member_pages": member_pages,
            "declared_pages": declared_pages,
            "continuation_inferred_boundary_pages": sorted(inferred_boundary_pages - set(declared_pages)),
            "start_page": min(declared_pages),
            "end_page": max(declared_pages),
            "page_count": len(declared_pages),
            "cross_page": len(declared_pages) > 1,
            "titles": [item.title for item in group_tables if item.title],
            "headers": [list(item.columns) for item in group_tables if item.columns],
            "quality_scores": [item.quality_score for item in group_tables],
            "links": [asdict(link) for link in links_by_root.get(union.find(group_tables[0].table_id), [])],
        })
    return sorted(groups, key=lambda item: (item["filename"].casefold(), item["start_page"], item["table_group_id"]))


def _filename(value: Any) -> str:
    return _text(value).replace("\\", "/").rsplit("/", 1)[-1].casefold()


def evaluate_gold_coverage(groups: list[dict], records: list[dict]) -> dict:
    member: set[tuple[str, int]] = set()
    declared: set[tuple[str, int]] = set()
    inferred: set[tuple[str, int]] = set()
    all_table_pages: set[tuple[str, int]] = set()
    for group in groups:
        name = _filename(group["filename"])
        member.update((name, page) for page in group["member_pages"])
        declared.update((name, page) for page in group["declared_pages"])
        inferred.update((name, page) for page in group["continuation_inferred_boundary_pages"])
        all_table_pages.update((name, page) for page in group["member_pages"])

    details = []
    for record in records:
        gold = {(_filename(name), int(page)) for name, page in record.get("gold_pages", [])}
        direct_hit = bool(gold & member)
        declared_hit = bool(gold & declared)
        continuation_hit = bool(gold & (declared | inferred))
        adjacent_hit = any((name, page + offset) in all_table_pages for name, page in gold for offset in (-1, 1))
        details.append({
            "financebench_id": record.get("financebench_id"),
            "gold_pages": sorted([list(item) for item in gold]),
            "direct_member_hit": direct_hit,
            "declared_range_hit": declared_hit,
            "continuation_evidenced_hit": continuation_hit,
            "adjacent_table_upper_bound_hit": direct_hit or adjacent_hit,
            "recovered_by_declared_range": declared_hit and not direct_hit,
            "recovered_by_continuation_evidence": continuation_hit and not declared_hit,
        })

    def count(key: str) -> int:
        return sum(bool(item[key]) for item in details)

    total = len(details)
    return {
        "questions": total,
        "direct_table_member_coverage": {"count": count("direct_member_hit"), "rate": round(count("direct_member_hit") / max(1, total), 4)},
        "declared_table_range_coverage": {"count": count("declared_range_hit"), "rate": round(count("declared_range_hit") / max(1, total), 4)},
        "continuation_evidenced_group_coverage": {"count": count("continuation_evidenced_hit"), "rate": round(count("continuation_evidenced_hit") / max(1, total), 4)},
        "adjacent_table_upper_bound": {"count": count("adjacent_table_upper_bound_hit"), "rate": round(count("adjacent_table_upper_bound_hit") / max(1, total), 4)},
        "recovered_ids": {
            "declared_range": [item["financebench_id"] for item in details if item["recovered_by_declared_range"]],
            "continuation_evidence": [item["financebench_id"] for item in details if item["recovered_by_continuation_evidence"]],
        },
        "details": details,
    }


def summarize(tables: list[TableRecord], groups: list[dict], coverage: dict) -> dict:
    cross = [group for group in groups if group["cross_page"]]
    multi_table = [group for group in groups if len(group["table_ids"]) > 1]
    reason_counts = Counter(reason for group in groups for link in group["links"] for reason in link["reasons"])
    return {
        "source_table_count": len(tables),
        "document_count": len({table.document_id for table in tables}),
        "table_group_count": len(groups),
        "multi_table_group_count": len(multi_table),
        "cross_page_group_count": len(cross),
        "cross_page_group_rate": round(len(cross) / max(1, len(groups)), 4),
        "average_pages_per_group": round(sum(group["page_count"] for group in groups) / max(1, len(groups)), 4),
        "average_tables_per_group": round(sum(len(group["table_ids"]) for group in groups) / max(1, len(groups)), 4),
        "largest_group_pages": max((group["page_count"] for group in groups), default=0),
        "accepted_link_count": sum(len(group["links"]) for group in groups),
        "link_reason_counts": dict(reason_counts),
        "diagnostic30_coverage": {key: value for key, value in coverage.items() if key != "details"},
    }


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    coverage = payload["coverage"]
    lines = [
        "# Evidence Architecture v5 — Table Evidence Group 离线审计", "",
        "> 本报告仅从 PostgreSQL 读取 `document_tables`，没有写数据库、没有修改生产 pipeline，也没有调用 Dense/BM25/RRF/Jina/LLM/Judge。", "",
        "## 结论", "",
        f"- PostgreSQL 实际读取 `{summary['source_table_count']}` 张表，来自 `{summary['document_count']}` 个 document。",
        f"- 构建 `{summary['table_group_count']}` 个 group，其中多 table group `{summary['multi_table_group_count']}` 个，跨页 group `{summary['cross_page_group_count']}` 个（`{summary['cross_page_group_rate']:.2%}`）。",
        f"- 平均每组 `{summary['average_pages_per_group']}` 页、`{summary['average_tables_per_group']}` 张表；最大 group `{summary['largest_group_pages']}` 页。",
        f"- 验收指标没有提升：可信 TableStore gold-page coverage 仍为 `{coverage['direct_table_member_coverage']['count']}/{coverage['questions']}`，group 后仍为 `{coverage['continuation_evidenced_group_coverage']['count']}/{coverage['questions']}`。",
        "- 覆盖结论必须按证据强度分层。`adjacent_table_upper_bound` 只是无条件相邻页的理论上限，不算 Table Evidence Group 已实现的覆盖。", "",
        "## Diagnostic30 gold-page coverage", "",
        "| 口径 | 覆盖 | 含义 |", "|---|---:|---|",
    ]
    labels = (
        ("direct_table_member_coverage", "表实际绑定到 gold page"),
        ("declared_table_range_coverage", "表的 start/end page 声明覆盖 gold page"),
        ("continuation_evidenced_group_coverage", "声明范围或明确 continuation 线索覆盖"),
        ("adjacent_table_upper_bound", "任一相邻页有表（仅理论上限）"),
    )
    for key, label in labels:
        item = coverage[key]
        lines.append(f"| `{key}` | {item['count']}/{coverage['questions']} ({item['rate']:.2%}) | {label} |")
    lines.extend([
        "", "### 恢复题目", "",
        f"- `declared_range`: {coverage['recovered_ids']['declared_range'] or '无'}",
        f"- `continuation_evidence`: {coverage['recovered_ids']['continuation_evidence'] or '无'}",
        "", "## Group 构建规则", "",
        "- `document_id` 必须完全一致，`page_id` 必须存在，并且仅比较相邻的 0-based `page_number`；link trace 显式记录 `page_id_contiguous`。",
        "- 使用 table title、headers、边界 nearby text 的相似度与 continuation 关键词加权打分。",
        "- 阈值为 0.48，并要求 title/header 或 continuation+nearby text 的独立佐证。",
        "- 无 continuation 时要求近乎一致的 title+header，或由边界 nearby text 共同佐证；相同年份列本身不参与相似度。",
        "- 同一相邻页对采用一对一最高分匹配，避免把同页多个无关表合并成一个大组。",
        "- group ID 由 document_id 与有序 table IDs 确定性哈希生成；本实验不写回数据库。",
        "", "## Link 统计", "",
        f"- Accepted links: `{summary['accepted_link_count']}`",
        f"- Link reasons: `{summary['link_reason_counts']}`",
        "", "## 跨页 groups（前 30 个）", "",
        "| Group | Document | Pages | Tables | Mean quality | Link score |", "|---|---|---|---:|---:|---:|",
    ])
    cross = [group for group in payload["groups"] if group["cross_page"]]
    for group in cross[:30]:
        scores = group["quality_scores"]
        link_scores = [link["score"] for link in group["links"]]
        lines.append(
            f"| `{group['table_group_id']}` | `{group['filename']}` | "
            f"{group['start_page']}–{group['end_page']} | {len(group['table_ids'])} | "
            f"{sum(scores) / max(1, len(scores)):.3f} | {max(link_scores, default=0):.3f} |"
        )
    lines.extend([
        "", "## 判断与下一步", "",
        "- 如果可信 group coverage 高于 12/30，说明跨页范围/continuation 能恢复一部分错误页关联，可进入 shadow table-group retrieval。",
        "- 如果仍为 12/30，而相邻页上限明显更高，说明瓶颈是 parser 的 page/table 关联或 gold 页附近未抽取出结构，不能靠检索模型或无条件 page±1 修复。",
        "- 下一步仍应先做离线人工抽样核验跨页 group precision；通过后才设计 shadow collection，不能直接接入生产。", "",
    ])
    return "\n".join(lines)


def load_tables() -> list[TableRecord]:
    db = SessionLocal()
    try:
        rows = db.query(DocumentTable).order_by(DocumentTable.document_id, DocumentTable.page_number).all()
        return [
            TableRecord(
                table_id=_text(row.table_id),
                document_id=_text(row.document_id),
                page_id=_text(row.page_id),
                filename=_text(row.filename),
                page_number=int(row.page_number or 0),
                start_page=int(row.start_page if row.start_page is not None else row.page_number or 0),
                end_page=int(row.end_page if row.end_page is not None else row.page_number or 0),
                title=_text(row.title),
                caption=_text(row.caption),
                columns=tuple(_flatten_strings(row.columns)),
                before_context=_text(row.before_context),
                after_context=_text(row.after_context),
                quality_score=float(row.quality_score or 0.0),
            )
            for row in rows
        ]
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    tables = load_tables()
    if not tables:
        raise RuntimeError("document_tables is empty")
    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    records = diagnostic.get("records") or []
    if not records:
        raise RuntimeError(f"no records found in {args.diagnostic}")
    groups = build_groups(tables)
    coverage = evaluate_gold_coverage(groups, records)
    payload = {
        "profile": "evidence_architecture_v5_table_group_offline",
        "scope": "PostgreSQL read-only; frozen diagnostic30; no retrieval/API/LLM/Judge calls",
        "source_diagnostic": str(args.diagnostic),
        "grouping_config": {
            "page_numbering": "internal loader 0-based",
            "adjacent_page_only": True,
            "link_threshold": 0.48,
            "weights": {"title": 0.35, "header": 0.35, "nearby_text": 0.15, "continuation": 0.15},
            "one_to_one_per_adjacent_page_pair": True,
        },
        "summary": summarize(tables, groups, coverage),
        "coverage": coverage,
        "groups": groups,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {args.output_json}\nMarkdown: {args.output_markdown}", flush=True)


if __name__ == "__main__":
    main()
