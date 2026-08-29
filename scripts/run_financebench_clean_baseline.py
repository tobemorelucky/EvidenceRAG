"""Run the frozen local clean-baseline-v1 FinanceBench experiment."""

import sys

from run_financebench_langsmith_experiment import main


def _ensure_option(name: str, value: str | None = None) -> None:
    if name in sys.argv:
        return
    sys.argv.append(name)
    if value is not None:
        sys.argv.append(value)


if __name__ == "__main__":
    _ensure_option("--evaluation-backend", "local")
    _ensure_option("--rag-profile", "clean_baseline")
    _ensure_option("--experiment-prefix", "evidencerag-clean-baseline-v1")
    _ensure_option("--enable-rerank")
    main()
