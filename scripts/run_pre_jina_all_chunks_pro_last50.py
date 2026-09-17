"""Run FinanceBench questions 51-100 with all pre-Jina chunks and Pro answer/Judge."""

from pre_jina_all_chunks_pro_common import ROOT, run_split


if __name__ == "__main__":
    run_split("last50", ROOT / "reports" / "pre_jina_all_chunks_pro_last50_v1")

