async (page) => {
    let conversations = [];
    const messageMap = {};
    const traceMap = {};
    let nextId = 1;

    await page.addInitScript(() => localStorage.setItem('accessToken', 'test-token'));
    await page.route('**/config/frontend', route => route.fulfill({
        json: { use_conversation_api: true }
    }));
    await page.route('**/auth/me', route => route.fulfill({
        json: { username: 'alice', role: 'user' }
    }));
    await page.route('**/conversation**', async route => {
        const request = route.request();
        const path = request.url().replace(/^https?:\/\/[^/]+/, '').split('?')[0];
        const method = request.method();

        if (path === '/conversation/create' && method === 'POST') {
            const id = `db-${nextId++}`;
            conversations.push({
                conversation_id: id,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
                message_count: 0,
                metadata: {}
            });
            messageMap[id] = [];
            traceMap[id] = [];
            return route.fulfill({ json: {
                conversation_id: id,
                created_at: new Date().toISOString(),
                memory_trace: { postgres_written: true }
            }});
        }

        if (path === '/conversation/chat/stream' && method === 'POST') {
            const body = request.postDataJSON();
            const id = body.conversation_id;
            const turn = messageMap[id].filter(item => item.role === 'assistant').length + 1;
            const answer = `第${turn}轮回答`;
            const citation = {
                id: `e${turn}`,
                filename: 'report.pdf',
                page_number: turn,
                text: `evidence ${turn}`
            };
            messageMap[id].push(
                {
                    id: messageMap[id].length + 1,
                    role: 'user',
                    content: body.message,
                    evidence_refs: [],
                    trace: null,
                    created_at: new Date().toISOString()
                },
                {
                    id: messageMap[id].length + 2,
                    role: 'assistant',
                    content: answer,
                    evidence_refs: [citation.id],
                    trace: { observability: { trace_id: `t${turn}` } },
                    created_at: new Date().toISOString()
                }
            );
            conversations.find(item => item.conversation_id === id).message_count = messageMap[id].length;
            traceMap[id].push({
                trace_id: `t${turn}`,
                retrieval_counts: { dense: 120, bm25: 30, rrf: 100 },
                rerank_parameters: { applied: true, candidate_k: 80, final_top_k: 12 },
                evidence_items: [citation],
                latency_ms: { total: 100 }
            });
            const events = [
                { type: 'content', content: answer },
                { type: 'citation', citation },
                { type: 'trace', rag_trace: { observability: { trace_id: `t${turn}` } }, citations: [citation] },
                { type: 'done', conversation_id: id }
            ].map(item => `data: ${JSON.stringify(item)}\n\n`).join('');
            return route.fulfill({ status: 200, contentType: 'text/event-stream', body: events });
        }

        if (path === '/conversation' && method === 'GET') {
            return route.fulfill({ json: { conversations } });
        }

        const match = path.match(/^\/conversation\/([^/]+)(?:\/(messages|trace))?$/);
        if (match) {
            const id = decodeURIComponent(match[1]);
            if (method === 'DELETE') {
                conversations = conversations.filter(item => item.conversation_id !== id);
                delete messageMap[id];
                delete traceMap[id];
                return route.fulfill({ json: {
                    conversation_id: id,
                    message: '成功删除会话',
                    deletion_trace: { postgres_deleted: true }
                }});
            }
            if (match[2] === 'messages') {
                return route.fulfill({ json: { conversation_id: id, messages: messageMap[id] || [] } });
            }
            if (match[2] === 'trace') {
                return route.fulfill({ json: { conversation_id: id, traces: traceMap[id] || [] } });
            }
        }
        return route.fulfill({ status: 404, json: { detail: 'not mocked' } });
    });

    await page.goto('http://127.0.0.1:8765');
    await page.waitForTimeout(1500);
}
