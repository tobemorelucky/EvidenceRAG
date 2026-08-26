import csv
from pathlib import Path

from scripts.backfill_finance_tables import financebench_filenames, select_pdf_paths


def test_backfill_defaults_to_documents_named_by_financebench(tmp_path: Path):
    dataset = tmp_path / "dataset.csv"
    with dataset.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doc_name"])
        writer.writeheader()
        writer.writerow({"doc_name": "REPORT_A"})
        writer.writerow({"doc_name": "REPORT_B.pdf"})
    (tmp_path / "REPORT_A.pdf").write_bytes(b"pdf")
    (tmp_path / "REPORT_B.pdf").write_bytes(b"pdf")
    (tmp_path / "UNRELATED.pdf").write_bytes(b"pdf")

    assert financebench_filenames(dataset) == {"REPORT_A.pdf", "REPORT_B.pdf"}
    selected = select_pdf_paths(tmp_path, dataset, [], all_pdfs=False)

    assert [path.name for path in selected] == ["REPORT_A.pdf", "REPORT_B.pdf"]


def test_backfill_explicit_filename_is_case_insensitive(tmp_path: Path):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("doc_name\n", encoding="utf-8")
    (tmp_path / "Report.pdf").write_bytes(b"pdf")

    selected = select_pdf_paths(tmp_path, dataset, ["report"], all_pdfs=False)

    assert [path.name for path in selected] == ["Report.pdf"]
