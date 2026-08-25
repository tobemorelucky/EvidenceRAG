"""Load cached, evidence-safe financial task policies from local configuration."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict


POLICY_DIR = Path(__file__).resolve().parents[1] / "configs" / "finance_policies"
SUPPORTED_TASK_TYPES = frozenset({"calculation", "comparison", "lookup", "selection", "judgment"})
_CACHE: Dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()


def finance_policy_enabled() -> bool:
    return os.getenv("FINANCE_POLICY_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def clear_finance_policy_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def load_finance_policy(task_type: str) -> dict:
    """Return one local policy and load metadata without invoking external services."""
    started = time.perf_counter()
    normalized = str(task_type or "lookup").strip().lower()
    if normalized not in SUPPORTED_TASK_TYPES:
        normalized = "lookup"
    if not finance_policy_enabled():
        return {
            "enabled": False,
            "task_type": normalized,
            "policy_file": "",
            "text": "",
            "chars": 0,
            "estimated_tokens": 0,
            "cache_hit": False,
            "load_ms": round((time.perf_counter() - started) * 1000, 4),
        }

    policy_file = POLICY_DIR / f"{normalized}.json"
    cache_hit = normalized in _CACHE
    if not cache_hit:
        with _CACHE_LOCK:
            cache_hit = normalized in _CACHE
            if not cache_hit:
                payload = json.loads(policy_file.read_text(encoding="utf-8"))
                if payload.get("task_type") != normalized:
                    raise ValueError(f"Policy task_type mismatch in {policy_file.name}")
                instructions = [str(item).strip() for item in payload.get("instructions") or [] if str(item).strip()]
                if not instructions:
                    raise ValueError(f"Policy has no instructions: {policy_file.name}")
                body = "\n".join(f"{index}. {item}" for index, item in enumerate(instructions, 1))
                text = (
                    "This policy guides the evidence-processing procedure only; it is not a factual source. "
                    "Every fact, operand, and conclusion must be supported by the Evidence section.\n"
                    f"{body}"
                )
                _CACHE[normalized] = {
                    "task_type": normalized,
                    "policy_file": policy_file.name,
                    "text": text,
                    "chars": len(text),
                    "estimated_tokens": (len(text) + 3) // 4,
                }
    policy = dict(_CACHE[normalized])
    policy.update(
        {
            "enabled": True,
            "cache_hit": cache_hit,
            "load_ms": round((time.perf_counter() - started) * 1000, 4),
        }
    )
    return policy
