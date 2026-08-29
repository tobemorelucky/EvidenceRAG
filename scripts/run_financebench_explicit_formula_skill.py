"""Run the isolated clean-baseline + explicit-formula-skill regression locally."""

import csv
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
    # With no explicit split/IDs, derive the fixed regression set from question
    # semantics. IDs are evaluation metadata and never enter skill code.
    if "--question-id" not in sys.argv and "--split" not in sys.argv:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "backend"))
        from skills.explicit_formula.skill import build_formula_contract

        with (root / "data" / "financebench_top40_100_langsmith_with_evidence.csv").open(
            encoding="utf-8-sig", newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        formula_ids = [
            row.get("financebench_id") or ""
            for row in rows
            if build_formula_contract(row.get("question") or "")[0] is not None
        ]
        sys.argv.extend(["--split", "all"])
        for financebench_id in formula_ids:
            sys.argv.extend(["--question-id", financebench_id])
        print(f"[formula-regression] auto-selected {len(formula_ids)} explicit-formula questions", flush=True)
    _ensure_option("--evaluation-backend", "local")
    _ensure_option("--rag-profile", "clean_baseline_formula_skill")
    _ensure_option("--experiment-prefix", "evidencerag-skill-explicit-formula-v1")
    _ensure_option("--enable-rerank")
    main()
