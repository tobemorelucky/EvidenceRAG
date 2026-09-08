from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
SCRIPT = (ROOT / "frontend" / "script.js").read_text(encoding="utf-8")
STYLE = (ROOT / "frontend" / "style.css").read_text(encoding="utf-8")


def test_finance_product_identity_and_examples_are_visible():
    assert "EvidenceRAG" in INDEX
    assert "金融财报智能问答系统" in INDEX
    assert "基于证据检索增强生成的可追溯 RAG 系统" in INDEX
    assert INDEX.count("@click=\"useExample(") >= 4


def test_user_controls_and_document_fields_are_localized():
    assert ">Profile<" not in INDEX
    assert ">Mode<" not in INDEX
    assert ">Chunks<" not in INDEX
    assert "知识领域" in INDEX
    assert "回答方式" in INDEX
    assert "文本片段" in INDEX


def test_advanced_retrieval_information_is_collapsed_by_default():
    marker = '<details v-if="activeTrace" class="advanced-details">'
    assert marker in INDEX
    assert marker.replace(">", " open>") not in INDEX
    for label in ("查询理解", "查询改写", "检索过程", "重排序", "证据来源"):
        assert label in INDEX
    assert "advancedTrace()" in SCRIPT


def test_answer_markdown_and_citation_cards_have_product_styles():
    assert "marked.parse(text)" in SCRIPT
    assert ".message-body h1" in STYLE
    assert ".message-body table" in STYLE
    assert ".message-body strong" in STYLE
    assert ".inline-citations button" in STYLE
    assert "第 {{ citation.page_number }} 页" in INDEX


def test_chat_dom_is_preserved_across_navigation_and_delete_uses_dialog():
    assert 'v-show="activeNav !== \'settings\'" class="conversation-pane"' in INDEX
    assert 'class="confirm-dialog"' in INDEX
    assert "删除这条会话？" in INDEX
    assert "requestDeleteSession(session)" in INDEX
    assert 'confirm(`确定要删除会话' not in SCRIPT
    assert "当前会话正在生成回答" in SCRIPT


def test_primary_navigation_is_interactive_and_history_load_is_non_blocking():
    assert '<template>\n            <section v-show="activeNav' not in INDEX
    assert 'type="button" class="new-chat-button" @click="handleNewChat"' in INDEX
    assert '@click="handleWorkspace"' in INDEX
    assert '@click="handleHistory"' in INDEX
    assert 'v-if="sessionsLoading"' in INDEX
    assert 'v-else-if="historyError"' in INDEX
    assert "async loadHistorySessions()" in SCRIPT
    assert "this.sessionsLoading = false" in SCRIPT
    assert "grid-column: 1" in STYLE
    assert "grid-column: 2" in STYLE
    assert "grid-column: 3" in STYLE


def test_conversation_switching_is_session_scoped_and_non_blocking():
    assert "conversationMessageCache: {}" in SCRIPT
    assert "activeStreams: {}" in SCRIPT
    assert "this.cacheActiveConversation();" in SCRIPT
    assert "const conversationMessages = this.messages;" in SCRIPT
    assert "this.activeStreams[sessionId]" in SCRIPT
    assert "请等待完成或先停止生成后再切换会话" not in SCRIPT


def test_document_table_cells_keep_native_row_alignment():
    assert 'class="file-cell-content"' in INDEX
    assert ".document-table td { height: 52px;" in STYLE
    assert "vertical-align: middle" in STYLE
    assert ".filename-cell { display: flex" not in STYLE
