const test = require('node:test');
const assert = require('node:assert/strict');

const EvidenceRagApiAdapter = require('../frontend/api-adapter.js');

function response(data, ok = true) {
    return {
        ok,
        async json() { return data; }
    };
}

test('conversation mode uses every v1.1 endpoint and normalizes responses', async () => {
    const calls = [];
    const authFetch = async (url, options = {}) => {
        calls.push({ url, options });
        if (url === '/conversation/create') return response({ conversation_id: 'db-1' });
        if (url === '/conversation') return response({
            conversations: [{
                conversation_id: 'db-1', title: 'Adobe 研发费用', updated_at: 'now', message_count: 2, metadata: {}
            }]
        });
        if (url.endsWith('/messages')) return response({
            messages: [{ role: 'user', content: 'question', created_at: 'now', trace: null }]
        });
        if (url.endsWith('/trace')) return response({ traces: [{ trace_id: 't1' }] });
        if (options.method === 'DELETE') return response({ conversation_id: 'db-1' });
        return response({});
    };
    const adapter = new EvidenceRagApiAdapter({
        authFetch,
        fetchImpl: async () => response({ use_conversation_api: true })
    });

    assert.equal(await adapter.loadConfig(), true);
    assert.equal((await adapter.createConversation()).conversation_id, 'db-1');
    const stream = adapter.streamChat({
        message: 'question', conversationId: 'db-1', profile: 'finance', executionMode: 'auto'
    });
    await stream;
    const conversation = (await adapter.listConversations())[0];
    assert.equal(conversation.session_id, 'db-1');
    assert.equal(conversation.title, 'Adobe 研发费用');
    assert.equal((await adapter.getMessages('db-1'))[0].type, 'human');
    assert.equal((await adapter.getTrace('db-1'))[0].trace_id, 't1');
    await adapter.deleteConversation('db-1');

    assert.ok(calls.some(call => call.url === '/conversation/chat/stream'));
    const streamCall = calls.find(call => call.url === '/conversation/chat/stream');
    assert.equal(JSON.parse(streamCall.options.body).conversation_id, 'db-1');
    assert.ok(calls.some(call => call.url === '/conversation/db-1/messages'));
    assert.ok(calls.some(call => call.url === '/conversation/db-1/trace'));
    assert.ok(calls.some(call => call.url === '/conversation/db-1' && call.options.method === 'DELETE'));
});

test('disabled feature flag preserves legacy chat and session contracts', async () => {
    const calls = [];
    const adapter = new EvidenceRagApiAdapter({
        authFetch: async (url, options = {}) => {
            calls.push({ url, options });
            if (url === '/sessions') return response({ sessions: [{ session_id: 'legacy-1' }] });
            if (url === '/sessions/legacy-1') return response({
                messages: [{ type: 'human', content: 'question' }]
            });
            return response({ message: 'ok' });
        },
        fetchImpl: async () => response({ use_conversation_api: false })
    });

    assert.equal(await adapter.loadConfig(), false);
    await adapter.streamChat({ message: 'question', conversationId: 'legacy-1' });
    assert.equal((await adapter.listConversations())[0].session_id, 'legacy-1');
    assert.equal((await adapter.getMessages('legacy-1'))[0].type, 'human');
    assert.deepEqual(await adapter.getTrace('legacy-1'), []);
    await adapter.deleteConversation('legacy-1');

    const streamCall = calls.find(call => call.url === '/chat/stream');
    assert.equal(JSON.parse(streamCall.options.body).session_id, 'legacy-1');
    assert.ok(calls.some(call => call.url === '/sessions/legacy-1' && call.options.method === 'DELETE'));
});

test('runtime config failure safely falls back to legacy mode', async () => {
    const adapter = new EvidenceRagApiAdapter({
        authFetch: async () => response({}),
        fetchImpl: async () => { throw new Error('offline'); }
    });
    assert.equal(await adapter.loadConfig(), false);
});

test('observability trace maps to the existing evidence inspector model', () => {
    const adapter = new EvidenceRagApiAdapter({
        authFetch: async () => response({}),
        fetchImpl: async () => response({ use_conversation_api: false })
    });
    const trace = adapter.normalizeTrace({
        trace_id: 'trace-1',
        retrieval_counts: { rrf: 120 },
        rerank_parameters: { candidate_k: 80, final_top_k: 12, applied: true },
        evidence_items: [{ id: 'e1', filename: 'report.pdf', page_number: 4, text: 'fact' }],
        latency_ms: { total: 123 }
    });
    assert.equal(trace.rrf_fused_candidate_count, 120);
    assert.equal(trace.final_top_k, 12);
    assert.equal(adapter.citationsFromTrace(trace)[0].filename, 'report.pdf');
});
