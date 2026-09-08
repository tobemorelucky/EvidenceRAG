"""Immutable online configuration for the production finance profile."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "production" / "finance_online_v2.json"


def _positive_int(payload: dict, key: str) -> int:
    value = int(payload[key])
    if value < 1:
        raise ValueError(f"{key} must be positive")
    return value


def _non_negative_int(payload: dict, key: str) -> int:
    value = int(payload[key])
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


@lru_cache(maxsize=1)
def load_finance_online_profile() -> dict[str, Any]:
    """Load and validate the checked-in profile once per process."""
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("profile") != "finance":
        raise ValueError("finance online profile must target profile=finance")
    rewrite = dict(payload.get("query_rewrite") or {})
    retrieval = dict(payload.get("retrieval") or {})
    rerank = dict(payload.get("rerank") or {})
    context = dict(payload.get("context") or {})
    prompt = dict(payload.get("prompt") or {})
    answer = dict(payload.get("answer") or {})
    if rewrite.get("enabled") is not True or rewrite.get("original_retained") is not True:
        raise ValueError("finance online query rewrite must be enabled and retain the original query")
    if str(rerank.get("provider") or "").casefold() != "jina":
        raise ValueError("finance online reranker provider must be jina")
    normalized = {
        "name": str(payload.get("name") or "finance_online_v2"),
        "profile": "finance",
        "query_rewrite": {
            "enabled": True,
            "max_rewrites": _positive_int(rewrite, "max_rewrites"),
            "original_retained": True,
        },
        "retrieval": {
            "dense_top_k": _positive_int(retrieval, "dense_top_k"),
            "bm25_top_k": _positive_int(retrieval, "bm25_top_k"),
            "rrf_top_k": _positive_int(retrieval, "rrf_top_k"),
            "rrf_k": _positive_int(retrieval, "rrf_k"),
        },
        "rerank": {
            "provider": "jina",
            "input_k": _positive_int(rerank, "input_k"),
            "output_k": _positive_int(rerank, "output_k"),
            "input_max_chars": max(
                0,
                int(os.getenv("FINANCE_JINA_MAX_CHARS", str(_non_negative_int(rerank, "input_max_chars")))),
            ),
        },
        "context": {"max_chars": _positive_int(context, "max_chars")},
        "prompt": {
            "name": str(prompt.get("name") or ""),
            "answer_directives": bool(prompt.get("answer_directives")),
            "calculation_contract": bool(prompt.get("calculation_contract")),
            "finance_policy": bool(prompt.get("finance_policy")),
        },
        "answer": {
            "model": str(answer.get("model") or ""),
            "temperature": float(answer.get("temperature")),
            "thinking": str(answer.get("thinking") or ""),
            "max_tokens": _positive_int(answer, "max_tokens"),
        },
    }
    if normalized["rerank"]["input_k"] > normalized["retrieval"]["rrf_top_k"]:
        raise ValueError("Jina input depth cannot exceed RRF output depth")
    if normalized["rerank"]["output_k"] > normalized["rerank"]["input_k"]:
        raise ValueError("Jina output depth cannot exceed input depth")
    if normalized["prompt"] != {
        "name": "clean_baseline_v1",
        "answer_directives": False,
        "calculation_contract": False,
        "finance_policy": False,
    }:
        raise ValueError("finance online prompt must be the isolated clean baseline")
    if normalized["answer"]["thinking"] != "disabled":
        raise ValueError("finance online answer thinking must be disabled")
    if not normalized["answer"]["model"]:
        raise ValueError("finance online answer model is required")
    return normalized


def finance_online_trace_fields(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_finance_online_profile()
    return {
        "profile_config": config["name"],
        "query_rewrite_enabled": bool(config["query_rewrite"]["enabled"]),
        "dense_top_k": config["retrieval"]["dense_top_k"],
        "bm25_top_k": config["retrieval"]["bm25_top_k"],
        "rrf_top_k": config["retrieval"]["rrf_top_k"],
        "jina_input_k": config["rerank"]["input_k"],
        "jina_output_k": config["rerank"]["output_k"],
        "jina_input_max_chars": config["rerank"]["input_max_chars"],
        "context_budget": config["context"]["max_chars"],
        "answer_prompt_name": config["prompt"]["name"],
        "answer_temperature": config["answer"]["temperature"],
        "answer_thinking": config["answer"]["thinking"],
        "answer_max_tokens": config["answer"]["max_tokens"],
        "retrieval_config": dict(config["retrieval"]),
        "jina_config": dict(config["rerank"]),
        "prompt_config": dict(config["prompt"]),
        "answer_config": dict(config["answer"]),
    }
