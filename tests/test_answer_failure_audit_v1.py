import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_answer_failures_v1.py"


def load_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("audit_answer_failures_v1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_audit_selects_expected_15_and_valid_categories():
    module = load_module()
    records = module.read_jsonl(module.DEFAULT_ANSWERS)
    references = module.read_references(module.DEFAULT_DATASET)

    audit = module.build_audit(records, references)

    assert audit["summary"]["audited"] == 15
    assert sum(audit["summary"]["category_counts"].values()) == 15
    assert {item["question_id"] for item in audit["items"]} == set(module.ANNOTATIONS)
    assert all(item["category"] in module.ALLOWED_CATEGORIES for item in audit["items"])
    assert audit["summary"]["strict_context_hit_within_audit"] == 13


def test_cli_writes_markdown_and_json_without_network(tmp_path):
    json_output = tmp_path / "audit.json"
    markdown_output = tmp_path / "audit.md"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    assert payload["summary"]["audited"] == 15
    assert "# Answer Failure Audit v1" in markdown
    assert "financebench_id_01911" in markdown
