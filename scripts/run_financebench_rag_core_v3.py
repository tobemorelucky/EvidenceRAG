"""Run targeted or formal local RAG Core v3 evaluations."""

import csv
import json
import sys
from pathlib import Path

from run_financebench_langsmith_experiment import main


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json"
DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"


def _ensure_option(name: str, value: str | None = None) -> None:
    if name in sys.argv:
        return
    sys.argv.append(name)
    if value is not None:
        sys.argv.append(value)


def _add_ids(ids: list[str]) -> None:
    _ensure_option("--split", "all")
    for financebench_id in ids:
        sys.argv.extend(["--question-id", financebench_id])


if __name__ == "__main__":
    with_skills = "--with-skills" in sys.argv
    if with_skills:
        sys.argv.remove("--with-skills")

    diagnostic_group = ""
    if "--diagnostic-group" in sys.argv:
        index = sys.argv.index("--diagnostic-group")
        if index + 1 >= len(sys.argv):
            raise SystemExit("--diagnostic-group requires a group name")
        diagnostic_group = sys.argv[index + 1]
        del sys.argv[index:index + 2]
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        group_names = {
            "selection": ["selection_loss10"],
            "candidate": ["candidate_miss10"],
            "correct": ["correct_regression10"],
            "selection-correct": ["selection_loss10", "correct_regression10"],
            "candidate-correct": ["candidate_miss10", "correct_regression10"],
        }
        if diagnostic_group not in group_names:
            raise SystemExit(f"unknown diagnostic group: {diagnostic_group}")
        ids = []
        for name in group_names[diagnostic_group]:
            for item in fixture[name]:
                financebench_id = item if isinstance(item, str) else item["financebench_id"]
                if financebench_id not in ids:
                    ids.append(financebench_id)
        _add_ids(ids)
        _ensure_option("--include-evidence-context")

    if "--explicit8" in sys.argv:
        sys.argv.remove("--explicit8")
        sys.path.insert(0, str(ROOT / "backend"))
        from skills.explicit_formula.skill import build_formula_contract

        with DATASET.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        ids = [
            row["financebench_id"] for row in rows
            if build_formula_contract(row.get("question") or "")[0] is not None
        ]
        _add_ids(ids)
        print(f"[formula-regression] auto-selected {len(ids)} questions", flush=True)

    profile = "rag_core_v3_skills" if with_skills else "rag_core_v3"
    suffix = "skills" if with_skills else "core"
    prefix = f"evidencerag-rag-core-v3-{suffix}"
    if diagnostic_group:
        prefix += f"-{diagnostic_group}"
    _ensure_option("--evaluation-backend", "local")
    _ensure_option("--rag-profile", profile)
    _ensure_option("--experiment-prefix", prefix)
    _ensure_option("--enable-rerank")
    main()
