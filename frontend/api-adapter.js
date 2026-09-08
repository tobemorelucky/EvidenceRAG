(function (global) {
    class EvidenceRagApiAdapter {
        constructor({ authFetch, fetchImpl } = {}) {
            this.authFetch = authFetch;
            this.fetchImpl = fetchImpl || global.fetch.bind(global);
            this.useConversationApi = false;
        }

        async loadConfig() {
            try {
                const response = await this.fetchImpl('/config/frontend', { cache: 'no-store' });
                if (response.ok) {
                    const config = await response.json();
                    this.useConversationApi = config.use_conversation_api === true;
                }
            } catch (_) {
                this.useConversationApi = false;
            }
            return this.useConversationApi;
        }

        async createConversation(metadata = {}) {
            if (!this.useConversationApi) {
                return { conversation_id: 'session_' + Date.now() };
            }
            const response = await this.authFetch('/conversation/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ metadata })
            });
            return this.readJson(response, '创建会话失败');
        }

        streamChat({ message, conversationId, profile, executionMode, signal }) {
            const url = this.useConversationApi ? '/conversation/chat/stream' : '/chat/stream';
            const identity = this.useConversationApi
                ? { conversation_id: conversationId }
                : { session_id: conversationId };
            return this.authFetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message,
                    ...identity,
                    profile,
                    execution_mode: executionMode
                }),
                signal
            });
        }

        async listConversations() {
            const url = this.useConversationApi ? '/conversation' : '/sessions';
            const data = await this.readJson(await this.authFetch(url), '加载历史记录失败');
            if (!this.useConversationApi) return data.sessions || [];
            return (data.conversations || []).map(item => ({
                session_id: item.conversation_id,
                title: item.title || item.metadata?.title || '未命名会话',
                updated_at: item.updated_at,
                message_count: item.message_count,
                metadata: item.metadata || {}
            }));
        }

        async getMessages(conversationId) {
            const url = this.useConversationApi
                ? `/conversation/${encodeURIComponent(conversationId)}/messages`
                : `/sessions/${encodeURIComponent(conversationId)}`;
            const data = await this.readJson(await this.authFetch(url), '加载会话消息失败');
            if (!this.useConversationApi) return data.messages || [];
            return (data.messages || []).map(item => ({
                type: item.role === 'user' ? 'human' : 'ai',
                content: item.content,
                timestamp: item.created_at,
                rag_trace: item.trace || null,
                evidence_refs: item.evidence_refs || []
            }));
        }

        async getTrace(conversationId) {
            if (!this.useConversationApi) return [];
            const url = `/conversation/${encodeURIComponent(conversationId)}/trace`;
            const data = await this.readJson(await this.authFetch(url), '加载会话 Trace 失败');
            return data.traces || [];
        }

        async deleteConversation(conversationId) {
            const url = this.useConversationApi
                ? `/conversation/${encodeURIComponent(conversationId)}`
                : `/sessions/${encodeURIComponent(conversationId)}`;
            return this.readJson(
                await this.authFetch(url, { method: 'DELETE' }),
                '删除失败'
            );
        }

        normalizeTrace(trace, citations = []) {
            if (!trace) return null;
            const reusedEvidence = trace.evidence_reuse?.source === 'previous'
                || (trace.policy_decision?.reuse_previous_evidence && trace.retrieval_executed === false);
            if (!trace.retrieval_counts) {
                return {
                    ...trace,
                    trace_id: trace.trace_id || trace.observability?.trace_id,
                    evidence_status: reusedEvidence ? 'reused' : trace.evidence_status,
                    latency_breakdown: trace.latency_breakdown || {
                        total_latency_ms: trace.latency_ms?.total
                    }
                };
            }
            return {
                ...trace,
                trace_id: trace.trace_id,
                rrf_fused_candidate_count: trace.retrieval_counts?.rrf,
                candidate_k: trace.rerank_parameters?.candidate_k,
                final_top_k: trace.rerank_parameters?.final_top_k,
                rerank_applied: trace.rerank_parameters?.applied,
                final_evidence_pack_used: trace.evidence_items || [],
                evidence_status: reusedEvidence
                    ? 'reused'
                    : (citations.length || trace.evidence_items?.length ? 'sufficient' : 'limited'),
                latency_breakdown: {
                    total_latency_ms: trace.latency_ms?.total
                }
            };
        }

        citationsFromTrace(trace) {
            return (trace?.evidence_items || []).map((item, index) => ({
                id: item.id || item.chunk_id || `trace-evidence-${index + 1}`,
                filename: item.filename,
                page_number: item.page_number,
                text: item.text || '',
                score: item.score
            })).filter(item => item.filename);
        }

        async readJson(response, fallbackMessage) {
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(data.detail || fallbackMessage);
            }
            return data;
        }
    }

    global.EvidenceRagApiAdapter = EvidenceRagApiAdapter;
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = EvidenceRagApiAdapter;
    }
})(typeof window !== 'undefined' ? window : globalThis);
