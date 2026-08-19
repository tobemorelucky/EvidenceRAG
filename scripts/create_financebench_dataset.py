"""Create a FinanceBench split in LangSmith, following the LegalQA dataset scripts."""

import argparse
import csv
import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
load_dotenv(PROJECT_ROOT / ".env", override=True)


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def development_ids(rows: list[dict], size: int = 20) -> set[str]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get("question_type") or "unknown", []).append(row)
    selected: list[dict] = []
    while len(selected) < size:
        added = False
        for _, group in sorted(groups.items()):
            group.sort(key=lambda item: item.get("financebench_id") or "")
            if group:
                selected.append(group.pop(0))
                added = True
                if len(selected) == size:
                    break
        if not added:
            break
    return {row.get("financebench_id") or "" for row in selected}


def parse_evidence(raw: str) -> list[dict]:
    try:
        records = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(records, list):
        records = [records]
    return [
        {
            "doc_name": item.get("doc_name", ""),
            "page_number": item.get("evidence_page_num"),
        }
        for item in records
        if isinstance(item, dict)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a FinanceBench LangSmith dataset")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--dataset-name", default="evidencerag_financebench_dev20_v1")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(DATA_PATH)
    dev_ids = development_ids(rows)
    if args.split == "dev":
        rows = [row for row in rows if row.get("financebench_id") in dev_ids]
    elif args.split == "holdout":
        rows = [row for row in rows if row.get("financebench_id") not in dev_ids]
    if args.limit > 0:
        rows = rows[: args.limit]

    client = Client()
    if list(client.list_datasets(dataset_name=args.dataset_name)):
        print(f"数据集已存在：{args.dataset_name}")
        print("为避免重复写入样本，请在 LangSmith 删除旧数据集或使用新的 --dataset-name。")
        return
    dataset = client.create_dataset(
        dataset_name=args.dataset_name,
        description=f"EvidenceRAG FinanceBench {args.split} split; citation-aware financial RAG evaluation.",
    )
    inputs, outputs, metadata = [], [], []
    for row in rows:
        inputs.append({"question": row["question"], "profile": "finance", "execution_mode": "static"})
        outputs.append({"answer": row.get("answer", "")})
        metadata.append(
            {
                "financebench_id": row.get("financebench_id", ""),
                "company": row.get("company", ""),
                "question_type": row.get("question_type", ""),
                "doc_name": row.get("doc_name", ""),
                "doc_type": row.get("doc_type", ""),
                "doc_period": row.get("doc_period", ""),
                "gold_evidence": parse_evidence(row.get("evidence", "")),
            }
        )
    client.create_examples(dataset_id=dataset.id, inputs=inputs, outputs=outputs, metadata=metadata)
    print(f"完成：已创建 LangSmith 数据集 {args.dataset_name}，共 {len(rows)} 条样本。")


if __name__ == "__main__":
    main()
