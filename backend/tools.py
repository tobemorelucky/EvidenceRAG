from typing import Optional
from langchain_core.tools import tool

_LAST_RAG_CONTEXT = None
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
_RAG_STEP_QUEUE = None  # asyncio.Queue, set by agent before streaming
_RAG_STEP_LOOP = None   # asyncio loop, captured when setting queue
_EMPTY_KNOWLEDGE_BASE_MESSAGE = "知识库当前为空，尚未上传文档，无法基于文档检索回答。"


def _set_last_rag_context(context: dict):
    global _LAST_RAG_CONTEXT
    _LAST_RAG_CONTEXT = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    global _LAST_RAG_CONTEXT
    context = _LAST_RAG_CONTEXT
    if clear:
        _LAST_RAG_CONTEXT = None
    return context


def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数。"""
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    _RAG_STEP_QUEUE = queue
    if queue:
        import asyncio
        try:
            _RAG_STEP_LOOP = asyncio.get_running_loop()
        except RuntimeError:
            _RAG_STEP_LOOP = asyncio.get_event_loop()
    else:
        _RAG_STEP_LOOP = None


def emit_rag_step(icon: str, label: str, detail: str = ""):
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    if _RAG_STEP_QUEUE is not None and _RAG_STEP_LOOP is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not _RAG_STEP_LOOP.is_closed():
                _RAG_STEP_LOOP.call_soon_threadsafe(_RAG_STEP_QUEUE.put_nowait, step)
        except Exception:
            pass


def _is_knowledge_base_empty() -> bool:
    """Best-effort empty check for the knowledge base.

    Return True only when we can clearly confirm the collection is missing
    or contains no rows. If the check itself fails, fall back to normal RAG
    flow instead of misreporting an infrastructure issue as an empty KB.
    """
    try:
        from milvus_client import MilvusManager

        milvus_manager = MilvusManager()
        if not milvus_manager.has_collection():
            return True

        rows = milvus_manager.query(output_fields=["filename"], limit=1)
        return not rows
    except Exception:
        return False


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    # ... guards omitted ...
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    if _KNOWLEDGE_TOOL_CALLS_THIS_TURN >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN += 1

    if _is_knowledge_base_empty():
        emit_rag_step("ℹ️", "知识库为空", "尚未上传文档，已跳过检索")
        return _EMPTY_KNOWLEDGE_BASE_MESSAGE

    from rag_pipeline import run_rag_graph

    # 在同步工具中获取当前的 Loop 可能不可靠，但我们之前是通过 call_soon_threadsafe 调度的。
    # 这里 _RAG_STEP_QUEUE 是在主线程/Loop 设置的全局变量。
    # 如果工具运行在线程池中，它是可以访问到全局变量 _RAG_STEP_QUEUE 的。
    # emit_rag_step 内部做了 try-except 和 get_event_loop()。

    # 问题可能出在 asyncio.get_event_loop() 在子线程中调用会报错或者拿不到主线程的loop。
    # 我们应该在 set_rag_step_queue 时也保存 loop 引用，或者在 emit_rag_step 中更健壮地获取 loop。

    rag_result = run_rag_graph(query)

    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        _set_last_rag_context({"rag_trace": rag_trace})
    retrieval_mode = (rag_trace or {}).get("retrieval_mode") or "baseline"

    if not docs:
        return "未检索到足够证据，无法基于当前知识库可靠回答。"

    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        doc_name = result.get("doc_name", "") or source.rsplit(".", 1)[0]
        page = result.get("page_number", "N/A")
        evidence_type = result.get("type", "chunk") or "chunk"
        text = result.get("text", "")
        formatted.append(f"[{i}] {doc_name} | {source} | Page {page} | Type {evidence_type}:\n{text}")

    if retrieval_mode == "finance_experimental":
        return (
            "Retrieved Evidence Pack:\n"
            + "\n\n---\n\n".join(formatted)
            + "\n\nAnswering Rules:\n"
            + "- Use only the evidence above.\n"
            + "- Prefer evidence that best matches the user's company, year, and document type.\n"
            + "- If a calculation is needed, show the formula and the numbers you used.\n"
            + "- If evidence is incomplete, say which field is missing.\n"
            + "- Do not invent numbers that are not present in the evidence above."
        )

    return (
        "Retrieved Context:\n"
        + "\n\n---\n\n".join(formatted)
        + "\n\nAnswering Rules:\n"
        + "- Use the retrieved context above to answer the question.\n"
        + "- If the context is relevant but partial, provide the best grounded answer and briefly note any uncertainty.\n"
        + "- Cite the supporting document and page when available.\n"
        + "- Do not invent facts that are not supported by the retrieved context."
    )
