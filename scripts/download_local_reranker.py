"""Download the default local reranker into the repository's ignored models directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a local reranker model")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "bge-reranker-v2-m3")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.model, local_dir=args.output)
    print(f"Downloaded {args.model} to {args.output}")


if __name__ == "__main__":
    main()
