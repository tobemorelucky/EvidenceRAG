"""Lazy local cross-encoder reranking for the external-service fallback path."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class LocalReranker:
    def __init__(self) -> None:
        self.enabled = os.getenv("LOCAL_RERANK_ENABLED", "false").lower() == "true"
        self.model_path = Path(os.getenv("LOCAL_RERANK_MODEL_PATH", "models/bge-reranker-v2-m3"))
        self.device = os.getenv("LOCAL_RERANK_DEVICE", "auto")
        self.batch_size = max(1, int(os.getenv("LOCAL_RERANK_BATCH_SIZE", "8")))
        self.max_length = max(64, int(os.getenv("LOCAL_RERANK_MAX_LENGTH", "384")))
        self.candidate_limit = max(1, int(os.getenv("LOCAL_RERANK_CANDIDATE_K", "5")))
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            raise FileNotFoundError(f"local reranker model not found: {self.model_path}")

        import torch
        from sentence_transformers import CrossEncoder

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"torch_dtype": torch.float16} if device.startswith("cuda") else {}
        self._model = CrossEncoder(
            str(self.model_path),
            device=device,
            max_length=self.max_length,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        return self._model

    def rerank(self, query: str, docs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        model = self._load()
        candidates = docs[: self.candidate_limit]
        scores = model.predict(
            [(query, doc.get("text", "") or "") for doc in candidates],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        ranked = []
        for doc, score in zip(candidates, scores):
            ranked.append({**doc, "rerank_score": float(score)})
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked[:top_k]
