"""Run local RAG Core v2 evaluation with optional frozen Skills."""

import json
import sys
from pathlib import Path

from run_financebench_langsmith_experiment import main


def _ensure_option(name: str, value: str | None = None) -> None:
    if name in sys.argv:
        return
    sys.argv.append(name)
    if value is not None:
        sys.argv.append(value)


if __name__ == "__main__":
    with_skills = "--with-skills" in sys.argv
    if with_skills:
        sys.argv.remove("--with-skills")
    diagnostic = "--diagnostic-fixed20" in sys.argv
    if diagnostic:
        sys.argv.remove("--diagnostic-fixed20")
        fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rag_core_v2_diagnostic_ids.json"
        categories = json.loads(fixture.read_text(encoding="utf-8"))["categories"]
        _ensure_option("--split", "all")
        for values in categories.values():
            for financebench_id in values:
                sys.argv.extend(["--question-id", financebench_id])
        _ensure_option("--include-evidence-context")
    explicit8 = "--explicit8" in sys.argv
    if explicit8:
        sys.argv.remove("--explicit8")
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "backend"))
        import csv
        from skills.explicit_formula.skill import build_formula_contract

        with (root / "data" / "financebench_top40_100_langsmith_with_evidence.csv").open(
            encoding="utf-8-sig", newline="",
        ) as handle:
            formula_rows = list(csv.DictReader(handle))
        formula_ids = [
            row.get("financebench_id") or ""
            for row in formula_rows
            if build_formula_contract(row.get("question") or "")[0] is not None
        ]
        _ensure_option("--split", "all")
        for financebench_id in formula_ids:
            sys.argv.extend(["--question-id", financebench_id])
        print(f"[formula-regression] auto-selected {len(formula_ids)} questions", flush=True)
    profile = "rag_core_v2_skills" if with_skills else "rag_core_v2"
    prefix = "evidencerag-rag-core-v2-skills" if with_skills else "evidencerag-rag-core-v2"
    _ensure_option("--evaluation-backend", "local")
    _ensure_option("--rag-profile", profile)
    _ensure_option("--experiment-prefix", prefix)
    _ensure_option("--enable-rerank")
    main()
