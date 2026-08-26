from pydantic import BaseModel
from typing import Optional, List


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: Optional[str] = "user"
    admin_code: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class CurrentUserResponse(BaseModel):
    username: str
    role: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_session"
    profile: Optional[str] = None
    execution_mode: Optional[str] = None


class DebugRetrievalRequest(BaseModel):
    question: str
    top_k: int = 10


class RetrievedChunk(BaseModel):
    rank: Optional[int] = None
    filename: str
    doc_name: Optional[str] = None
    page_number: Optional[str | int] = None
    type: Optional[str] = None
    text: Optional[str] = None
    score: Optional[float] = None
    rrf_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    source: Optional[str] = None
    chunk_id: Optional[str] = None


class RagTrace(BaseModel):
    tool_used: bool = True
    tool_name: str = "retrieval_orchestrator"
    query: Optional[str] = None
    original_question: Optional[str] = None
    rewritten_question: Optional[str] = None
    expanded_query: Optional[str] = None
    step_back_question: Optional[str] = None
    step_back_answer: Optional[str] = None
    expansion_type: Optional[str] = None
    hypothetical_doc: Optional[str] = None
    retrieval_stage: Optional[str] = None
    grade_score: Optional[str] = None
    grade_route: Optional[str] = None
    rewrite_needed: Optional[bool] = None
    rewrite_used: Optional[bool] = None
    rewrite_strategy: Optional[str] = None
    rewrite_query: Optional[str] = None
    rerank_enabled: Optional[bool] = None
    rerank_applied: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    rerank_error: Optional[str] = None
    rerank_provider: Optional[str] = None
    remote_attempt_count: Optional[int] = None
    remote_success: Optional[bool] = None
    remote_errors: Optional[List[str]] = None
    rerank_cache_hit: Optional[bool] = None
    rerank_fallback_used: Optional[bool] = None
    rerank_fallback_reason: Optional[str] = None
    rerank_trace: Optional[dict] = None
    retrieval_mode: Optional[str] = None
    query_planner_enabled: Optional[bool] = None
    planner_intent: Optional[str] = None
    planner_must_keep_terms: Optional[List[str]] = None
    planner_dense_queries: Optional[List[str]] = None
    planner_semantic_queries: Optional[List[str]] = None
    planner_evidence_field_queries: Optional[List[str]] = None
    planner_table_heading_queries: Optional[List[str]] = None
    planner_keyword_queries: Optional[List[str]] = None
    planner_table_queries: Optional[List[str]] = None
    planner_validation_dropped_queries: Optional[List[dict]] = None
    planner_parse_error: Optional[str] = None
    per_query_retrieval_counts: Optional[List[dict]] = None
    rrf_fused_candidate_count: Optional[int] = None
    page_level_fusion_enabled: Optional[bool] = None
    fused_page_count: Optional[int] = None
    fused_top_pages: Optional[List[dict]] = None
    fused_pages_after_anchor_guard: Optional[List[dict]] = None
    page_anchor_filtered_count: Optional[int] = None
    page_contributing_routes: Optional[dict] = None
    page_fusion_used_for_final_context: Optional[bool] = None
    final_evidence_pack_source: Optional[str] = None
    candidate_k: Optional[int] = None
    final_top_k: Optional[int] = None
    two_stage_retrieval: Optional[bool] = None
    doc_stage_top_n: Optional[int] = None
    page_stage_top_n: Optional[int] = None
    leaf_retrieve_level: Optional[int] = None
    auto_merge_enabled: Optional[bool] = None
    auto_merge_applied: Optional[bool] = None
    auto_merge_threshold: Optional[int] = None
    auto_merge_replaced_chunks: Optional[int] = None
    auto_merge_steps: Optional[int] = None
    page_merge_applied: Optional[bool] = None
    merged_chunk_count: Optional[int] = None
    final_context_chunk_count: Optional[int] = None
    cover_page_filtered_count: Optional[int] = None
    fallback_used: Optional[bool] = None
    fallback_reason: Optional[str] = None
    table_aware_retrieval_mode: Optional[str] = None
    table_aware_auto_triggered: Optional[bool] = None
    table_aware_trigger_reason: Optional[List[str]] = None
    query_anchors: Optional[List[str]] = None
    anchor_guard_applied: Optional[bool] = None
    anchor_filtered_count: Optional[int] = None
    table_context_source: Optional[str] = None
    table_evidence_hit_count: Optional[int] = None
    table_context_table_count: Optional[int] = None
    table_context_char_count: Optional[int] = None
    evidence_unit_count: Optional[int] = None
    evidence_units_with_tables: Optional[int] = None
    table_attached_count: Optional[int] = None
    table_attach_reasons: Optional[List[str]] = None
    evidence_group_count: Optional[int] = None
    selected_evidence_group_count: Optional[int] = None
    evidence_frame_enabled: Optional[bool] = None
    evidence_frame_count: Optional[int] = None
    table_frame_count: Optional[int] = None
    evidence_frame_tables_considered: Optional[int] = None
    evidence_frame_tables_accepted: Optional[int] = None
    frames_with_period: Optional[int] = None
    frames_with_unit_scale: Optional[int] = None
    frames_used_for_execution: Optional[int] = None
    evidence_frame_skipped: Optional[dict] = None
    evidence_frame_load_ms: Optional[float] = None
    evidence_frame_load_errors: Optional[List[str]] = None
    evidence_frame_page_window: Optional[int] = None
    evidence_frame_adjacent_page_tables: Optional[int] = None
    evidence_groups_debug: Optional[List[dict]] = None
    selected_evidence_groups: Optional[List[dict]] = None
    group_scores: Optional[List[dict]] = None
    expanded_snippet_count: Optional[int] = None
    relevant_table_row_count: Optional[int] = None
    dropped_group_reasons: Optional[List[dict]] = None
    table_candidate_filenames: Optional[List[str]] = None
    table_candidate_pages: Optional[List[dict]] = None
    table_ids: Optional[List[str]] = None
    table_context_skipped_reasons: Optional[List[str]] = None
    query_parse: Optional[dict] = None
    latency_breakdown: Optional[dict] = None
    final_evidence_pack_debug_count: Optional[int] = None
    final_evidence_pack_used_count: Optional[int] = None
    dropped_evidence_count: Optional[int] = None
    dropped_reasons: Optional[dict] = None
    prompt_context_char_count_estimate: Optional[int] = None
    selected_docs: Optional[List[dict]] = None
    selected_pages: Optional[List[dict]] = None
    page_scores: Optional[List[dict]] = None
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
    initial_retrieved_chunks: Optional[List[RetrievedChunk]] = None
    expanded_retrieved_chunks: Optional[List[RetrievedChunk]] = None
    page_stage_candidates: Optional[List[RetrievedChunk]] = None
    final_retrieved_chunks: Optional[List[RetrievedChunk]] = None
    final_evidence_pack: Optional[List[RetrievedChunk]] = None
    final_evidence_pack_debug: Optional[List[RetrievedChunk]] = None
    final_evidence_pack_used: Optional[List[RetrievedChunk]] = None
    doc_stage_selected_docs: Optional[List[dict]] = None
    profile: Optional[str] = None
    execution_mode: Optional[str] = None
    route_reason: Optional[str] = None
    evidence_status: Optional[str] = None
    trace_id: Optional[str] = None
    prompt_version: Optional[str] = None
    agent_tool_calls: Optional[List[dict]] = None
    agent_tool_call_count: Optional[int] = None


class Citation(BaseModel):
    id: str
    filename: str
    page_number: Optional[str | int] = None
    text: Optional[str] = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    response: str
    rag_trace: Optional[RagTrace] = None
    execution_mode: Optional[str] = None
    route_reason: Optional[str] = None
    citations: Optional[List[Citation]] = None
    evidence_status: Optional[str] = None
    calculation: Optional[dict] = None
    trace_id: Optional[str] = None
    usage: Optional[dict] = None


class MessageInfo(BaseModel):
    type: str
    content: str
    timestamp: str
    rag_trace: Optional[RagTrace] = None


class SessionMessagesResponse(BaseModel):
    messages: List[MessageInfo]


class SessionInfo(BaseModel):
    session_id: str
    updated_at: str
    message_count: int


class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]


class SessionDeleteResponse(BaseModel):
    session_id: str
    message: str


class DocumentInfo(BaseModel):
    filename: str
    file_type: str
    chunk_count: int
    page_count: int = 0
    index_status: str = "ready"
    uploaded_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_processed: int
    message: str


class DocumentUploadStartResponse(BaseModel):
    job_id: str
    filename: str
    message: str


class UploadStepInfo(BaseModel):
    key: str
    label: str
    percent: int
    status: str
    message: str = ""


class DocumentUploadJobResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    current_step: str
    message: str
    total_chunks: int = 0
    processed_chunks: int = 0
    error: Optional[str] = None
    created_at: str
    updated_at: str
    steps: List[UploadStepInfo]


class DocumentDeleteStartResponse(BaseModel):
    job_id: str
    filename: str
    message: str


class DocumentBatchDeleteRequest(BaseModel):
    filenames: List[str]


class DocumentBatchDeleteStartResponse(BaseModel):
    jobs: List[DocumentDeleteStartResponse]
    message: str


class DocumentDeleteJobResponse(DocumentUploadJobResponse):
    pass


class DocumentDeleteResponse(BaseModel):
    filename: str
    chunks_deleted: int
    message: str
