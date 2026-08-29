"""EvidenceRAG chat service.

The public function names are retained for API compatibility. Retrieval routing,
answer generation, and conversation persistence are implemented by dedicated
modules.
"""

import asyncio
import json
import os
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from answer_generator import generate_answer, stream_answer, summarize_messages
from calculation_service import validate_numeric_display, validate_or_repair_structured_answer
from conversation_service import storage
from rag_orchestrator import RetrievalServiceError, prepare_rag_response
from query_parser import assess_answer_facets
from tools import set_rag_step_queue


INSUFFICIENT_EVIDENCE_MESSAGE = "未检索到足够证据，无法基于当前知识库可靠回答。"


def _finalize_generated_answer(answer: str, prepared: dict) -> str:
    task_spec = prepared.get("query_spec") or {}
    calculation = prepared.get("calculation")
    answer, consistency_trace = validate_or_repair_structured_answer(answer, task_spec, calculation)
    answer, numeric_trace = validate_numeric_display(answer, task_spec, calculation)
    facet_trace = assess_answer_facets(answer, task_spec)
    trace = prepared["rag_trace"]
    trace["answer_consistency"] = consistency_trace
    trace["numeric_display_validation"] = numeric_trace
    trace["answer_facet_validation"] = facet_trace
    if facet_trace.get("missing_facets") and trace.get("evidence_flow_stage") == "evidence_ready_for_utilization":
        trace["evidence_flow_stage"] = "evidence_utilization_failure"
    elif facet_trace.get("complete") and trace.get("evidence_flow_stage") == "evidence_ready_for_utilization":
        trace["evidence_flow_stage"] = "evidence_utilization_complete"
    return answer


def _compact_history(messages: list) -> list:
    if len(messages) <= 50:
        return messages
    summary = summarize_messages(messages[:40])
    return [SystemMessage(content=f"之前的对话摘要：\n{summary}")] + messages[40:]


def _response_metadata(prepared: dict, usage: dict | None = None) -> dict:
    return {
        "execution_mode": prepared["execution_mode"],
        "route_reason": prepared["route_reason"],
        "citations": prepared["citations"],
        "evidence_status": prepared["evidence_status"],
        "calculation": prepared.get("calculation"),
        "trace_id": prepared["trace_id"],
        "usage": usage or {},
    }


def chat_with_agent(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
    profile: str | None = None,
    execution_mode: str | None = None,
):
    """Run one evidence-grounded chat turn."""
    started = time.perf_counter()
    messages = _compact_history(storage.load(user_id, session_id))
    history = list(messages)
    messages.append(HumanMessage(content=user_text))

    prepared = prepare_rag_response(user_text, profile=profile, mode=execution_mode)
    if prepared["evidence_status"] == "insufficient":
        response_content, usage = INSUFFICIENT_EVIDENCE_MESSAGE, {}
    else:
        response_content, usage = generate_answer(
            user_text,
            prepared["evidence"],
            history,
            prepared.get("task_policy", ""),
        )
        response_content = _finalize_generated_answer(response_content, prepared)

    messages.append(AIMessage(content=response_content))
    rag_trace = prepared["rag_trace"]
    latency = dict(rag_trace.get("latency_breakdown") or {})
    latency["total_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    rag_trace["latency_breakdown"] = latency
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)

    return {
        "response": response_content,
        "rag_trace": rag_trace,
        **_response_metadata(prepared, usage),
    }


async def chat_with_agent_stream(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
    profile: str | None = None,
    execution_mode: str | None = None,
):
    """Stream operational retrieval status and a grounded answer using SSE."""
    messages = _compact_history(storage.load(user_id, session_id))
    history = list(messages)
    messages.append(HumanMessage(content=user_text))
    status_queue: asyncio.Queue = asyncio.Queue()

    class _StatusProxy:
        def put_nowait(self, step):
            status_queue.put_nowait(step)

    set_rag_step_queue(_StatusProxy())
    prepare_task = asyncio.create_task(
        asyncio.to_thread(prepare_rag_response, user_text, profile, execution_mode)
    )

    try:
        while not prepare_task.done():
            try:
                step = await asyncio.wait_for(status_queue.get(), timeout=0.15)
            except asyncio.TimeoutError:
                continue
            event = {
                "type": "status",
                "stage": "retrieval",
                "label": step.get("label", "正在检索证据"),
                "detail": step.get("detail", ""),
                "step": step,
            }
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        prepared = await prepare_task
        while not status_queue.empty():
            step = status_queue.get_nowait()
            event = {
                "type": "status",
                "stage": "retrieval",
                "label": step.get("label", "正在检索证据"),
                "detail": step.get("detail", ""),
                "step": step,
            }
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except RetrievalServiceError as exc:
        yield f"data: {json.dumps({'type': 'error', 'content': str(exc)}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        return
    finally:
        set_rag_step_queue(None)
        if not prepare_task.done():
            prepare_task.cancel()

    for citation in prepared["citations"]:
        yield f"data: {json.dumps({'type': 'citation', 'citation': citation}, ensure_ascii=False)}\n\n"

    full_response = ""
    usage = {}
    if prepared["evidence_status"] == "insufficient":
        full_response = INSUFFICIENT_EVIDENCE_MESSAGE
        yield f"data: {json.dumps({'type': 'content', 'content': full_response}, ensure_ascii=False)}\n\n"
    elif (prepared.get("calculation") or {}).get("authoritative") or (
        prepared.get("calculation")
        and os.getenv("NUMERIC_DISPLAY_VALIDATOR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    ):
        generated = ""
        async for content, chunk_usage in stream_answer(
            user_text,
            prepared["evidence"],
            history,
            prepared.get("task_policy", ""),
        ):
            generated += content
            usage = chunk_usage or usage
        full_response = _finalize_generated_answer(generated, prepared)
        yield f"data: {json.dumps({'type': 'content', 'content': full_response}, ensure_ascii=False)}\n\n"
    else:
        async for content, chunk_usage in stream_answer(
            user_text,
            prepared["evidence"],
            history,
            prepared.get("task_policy", ""),
        ):
            full_response += content
            usage = chunk_usage or usage
            yield f"data: {json.dumps({'type': 'content', 'content': content}, ensure_ascii=False)}\n\n"
        _finalize_generated_answer(full_response, prepared)

    rag_trace = prepared["rag_trace"]
    trace_event = {"type": "trace", "rag_trace": rag_trace, **_response_metadata(prepared, usage)}
    yield f"data: {json.dumps(trace_event, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

    messages.append(AIMessage(content=full_response))
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)
