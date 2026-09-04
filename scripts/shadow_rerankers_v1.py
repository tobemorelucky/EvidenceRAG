"""Isolated cross-encoder adapters. Inputs are query/text only, never gold."""
from __future__ import annotations

import math
import time
from pathlib import Path
from urllib.parse import urlparse


def validate_order(items, count):
    if len(items) != count or {x["index"] for x in items} != set(range(count)):
        raise ValueError("Reranker must return every input exactly once")
    if any(type(x["index"]) is not int or not math.isfinite(float(x["score"])) for x in items):
        raise ValueError("Invalid index/score")
    return sorted(items, key=lambda x: (-float(x["score"]), x["index"]))


class IdentityReranker:
    config = {"backend": "identity", "model": "original_rrf_order"}

    def rank(self, query, texts):
        return [{"index": i, "score": float(len(texts) - i)} for i in range(len(texts))], {"requests": 0}


class JinaReranker:
    def __init__(self, key, model="jina-reranker-v3", endpoint="https://api.jina.ai/v1/rerank", interval=8):
        parsed = urlparse(endpoint)
        if parsed.scheme != "https" or parsed.hostname != "api.jina.ai" or parsed.path != "/v1/rerank" or parsed.query or parsed.username:
            raise ValueError("Only the configured official Jina rerank endpoint is allowed")
        if not key:
            raise ValueError("Jina credential missing")
        self.key, self.endpoint, self.interval = key, endpoint, interval
        self.last_request = None
        self.config = {"backend": "jina", "model": model, "endpoint": endpoint,
                       "interval_seconds": interval, "timeout_seconds": [10, 45], "attempts": 1}

    def rank(self, query, texts):
        import requests
        wait = max(0, self.interval - (time.monotonic() - self.last_request)) if self.last_request else 0
        if wait:
            print(f"[jina] pacing {wait:.1f}s", flush=True)
            time.sleep(wait)
        self.last_request = time.monotonic()
        try:
            response = requests.post(self.endpoint,
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                json={"model": self.config["model"], "query": query, "documents": texts,
                      "top_n": len(texts), "return_documents": False}, timeout=(10, 45), allow_redirects=False)
        except requests.RequestException as exc:
            # Never log response bodies, credentials, or exception request dumps.
            raise RuntimeError(f"Jina transport failure: {type(exc).__name__}") from None
        if response.status_code != 200:
            detail = ""
            try:
                body = response.json()
                error = body.get("detail") or body.get("error") or body.get("message")
                if isinstance(error, dict):
                    error = error.get("message") or error.get("code")
                if isinstance(error, str):
                    detail = error.replace(self.key, "[REDACTED]")[:500]
            except (ValueError, TypeError, AttributeError):
                pass
            raise RuntimeError(f"Jina HTTP {response.status_code}; no local fallback; {detail}")
        try:
            data = response.json()
            ranked = validate_order([{"index": r["index"], "score": float(r["relevance_score"])} for r in data["results"]], len(texts))
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("Invalid/incomplete Jina response; no fallback") from None
        return ranked, {"requests": 1, "pacing_seconds": wait, "usage": data.get("usage")}


class BGEReranker:
    def __init__(self, model_path, device="auto", batch_size=4, max_length=1024):
        self.path = Path(model_path).resolve()
        if not (self.path / "model.safetensors").is_file():
            raise FileNotFoundError("Local BGE weights missing; automatic download disabled")
        if batch_size < 1 or max_length < 32:
            raise ValueError("Invalid BGE batch size/max length")
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.model = self.tokenizer = None
        self.config = {"backend": "bge", "model": "bge-reranker-v2-m3", "path": str(self.path),
                       "device": device, "batch_size": batch_size, "max_length": max_length,
                       "weights_bytes": (self.path / "model.safetensors").stat().st_size,
                       "weights_mtime_ns": (self.path / "model.safetensors").stat().st_mtime_ns}

    def rank(self, query, texts):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        load_ms = 0
        if self.model is None:
            started = time.perf_counter()
            self.device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.path), local_files_only=True,
                trust_remote_code=False, torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32)
            self.model.to(self.device).eval()
            load_ms = (time.perf_counter() - started) * 1000
        scores, truncated = [], 0
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                lengths = self.tokenizer([query] * len(batch), batch, truncation=False, padding=False)["input_ids"]
                truncated += sum(len(ids) > self.max_length for ids in lengths)
                encoded = self.tokenizer([query] * len(batch), batch, padding=True,
                    truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                values = self.model(**encoded).logits.float().reshape(-1).cpu().tolist()
                scores.extend(values)
        ranked = validate_order([{"index": i, "score": s} for i, s in enumerate(scores)], len(texts))
        return ranked, {"requests": 0, "device": self.device, "load_ms": load_ms,
                        "truncated_pairs": truncated, "max_length": self.max_length}
