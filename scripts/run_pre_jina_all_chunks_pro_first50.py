"""Run FinanceBench questions 1-50 with all pre-Jina chunks and Pro answer/Judge."""

from pre_jina_all_chunks_pro_common import ROOT, run_split


if __name__ == "__main__":
    run_split("first50", ROOT / "reports" / "pre_jina_all_chunks_pro_first50_v1")
