"""Run the isolated clean baseline + two-skill FinanceBench regression locally."""

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
    # With no explicit selection, include every question mentioning one of the
    # configured metric aliases. Safety exclusions remain visible as fallbacks.
    if "--question-id" not in sys.argv and "--split" not in sys.argv:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "backend"))
        from skills.canonical_finance_metric.skill import detect_metric_alias

        with (root / "data" / "financebench_top40_100_langsmith_with_evidence.csv").open(
            encoding="utf-8-sig", newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        target_ids = [
            row.get("financebench_id") or ""
            for row in rows
            if detect_metric_alias(row.get("question") or "")[0]
        ]
        sys.argv.extend(["--split", "all"])
        for financebench_id in target_ids:
            sys.argv.extend(["--question-id", financebench_id])
        print(f"[canonical-regression] auto-selected {len(target_ids)} metric-alias questions", flush=True)
    _ensure_option("--evaluation-backend", "local")
    _ensure_option("--rag-profile", "finance_skills_v1")
    _ensure_option("--experiment-prefix", "evidencerag-skill-canonical-finance-metric-v1-target")
    _ensure_option("--enable-rerank")
    main()
