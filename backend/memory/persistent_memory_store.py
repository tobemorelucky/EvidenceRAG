"""PostgreSQL-backed authoritative memory for Conversation-aware RAG v5."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

try:
    from database import SessionLocal, engine
except ModuleNotFoundError:
    from backend.database import SessionLocal, engine

from .conversation_state import ConversationState


MemoryBase = declarative_base()


def init_memory_db() -> None:
    # Register optional v6 observability tables before creating the shadow schema.
    try:
        import backend.rag_trace_service  # noqa: F401
    except ModuleNotFoundError:
        import rag_trace_service  # noqa: F401
    MemoryBase.metadata.create_all(bind=engine)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class MemoryConversation(MemoryBase):
    __tablename__ = "memory_conversations"
    __table_args__ = (UniqueConstraint("user_id", "conversation_id", name="uq_memory_user_conversation"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    conversation_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)


class MemoryMessage(MemoryBase):
    __tablename__ = "memory_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_ref_id: Mapped[int] = mapped_column(
        ForeignKey("memory_conversations.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_refs: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    trace_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)


class MemorySummary(MemoryBase):
    __tablename__ = "memory_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_ref_id: Mapped[int] = mapped_column(
        ForeignKey("memory_conversations.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    through_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)


class PersistentMemoryStore:
    def __init__(self, session_factory=SessionLocal):
        self.session_factory = session_factory

    @staticmethod
    def _conversation(db, user_id: str, conversation_id: str) -> MemoryConversation | None:
        return db.query(MemoryConversation).filter(
            MemoryConversation.user_id == user_id,
            MemoryConversation.conversation_id == conversation_id,
        ).first()

    def create_conversation(
        self, user_id: str, conversation_id: str | None = None, metadata: dict | None = None,
    ) -> dict:
        if not user_id.strip():
            raise ValueError("user_id must not be empty")
        conversation_id = conversation_id or str(uuid.uuid4())
        db = self.session_factory()
        try:
            if self._conversation(db, user_id, conversation_id):
                raise ValueError("conversation already exists")
            row = MemoryConversation(
                user_id=user_id, conversation_id=conversation_id, metadata_json=metadata or {},
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return self._serialize_conversation(row)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_conversation(self, user_id: str, conversation_id: str) -> dict | None:
        db = self.session_factory()
        try:
            row = self._conversation(db, user_id, conversation_id)
            return self._serialize_conversation(row) if row else None
        finally:
            db.close()

    def list_conversations(self, user_id: str) -> list[dict]:
        db = self.session_factory()
        try:
            rows = db.query(MemoryConversation).filter(
                MemoryConversation.user_id == user_id,
            ).order_by(MemoryConversation.updated_at.desc(), MemoryConversation.id.desc()).all()
            if not rows:
                return []
            counts = dict(
                db.query(MemoryMessage.conversation_ref_id, func.count(MemoryMessage.id))
                .filter(MemoryMessage.conversation_ref_id.in_([row.id for row in rows]))
                .group_by(MemoryMessage.conversation_ref_id)
                .all()
            )
            first_user_message = (
                db.query(
                    MemoryMessage.conversation_ref_id.label("conversation_ref_id"),
                    func.min(MemoryMessage.id).label("message_id"),
                )
                .filter(
                    MemoryMessage.conversation_ref_id.in_([row.id for row in rows]),
                    MemoryMessage.role == "user",
                )
                .group_by(MemoryMessage.conversation_ref_id)
                .subquery()
            )
            titles = dict(
                db.query(first_user_message.c.conversation_ref_id, MemoryMessage.content)
                .join(MemoryMessage, MemoryMessage.id == first_user_message.c.message_id)
                .all()
            )
            return [
                {
                    **self._serialize_conversation(row),
                    "message_count": int(counts[row.id]),
                    "title": self._conversation_title(titles.get(row.id, "")),
                }
                for row in rows
                if counts.get(row.id, 0) > 0
            ]
        finally:
            db.close()

    @staticmethod
    def _conversation_title(content: str, limit: int = 28) -> str:
        title = " ".join((content or "").split())
        if not title:
            return "未命名会话"
        return title if len(title) <= limit else f"{title[:limit]}…"

    def append_message(
        self, user_id: str, conversation_id: str, role: str, content: str,
        *, evidence_refs: list[str] | None = None, trace: dict | None = None,
    ) -> dict:
        if role not in {"user", "assistant"} or not content.strip():
            raise ValueError("valid role and non-empty content are required")
        db = self.session_factory()
        try:
            conversation = self._conversation(db, user_id, conversation_id)
            if not conversation:
                raise KeyError("conversation not found")
            row = MemoryMessage(
                conversation_ref_id=conversation.id, role=role, content=content,
                evidence_refs=evidence_refs or [], trace_json=trace,
            )
            conversation.updated_at = _utcnow()
            db.add(row)
            db.commit()
            db.refresh(row)
            return self._serialize_message(row)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def list_messages(self, user_id: str, conversation_id: str, *, limit: int | None = None) -> list[dict]:
        db = self.session_factory()
        try:
            conversation = self._conversation(db, user_id, conversation_id)
            if not conversation:
                return []
            query = db.query(MemoryMessage).filter(
                MemoryMessage.conversation_ref_id == conversation.id,
            ).order_by(MemoryMessage.id.desc() if limit else MemoryMessage.id.asc())
            rows = query.limit(limit).all() if limit else query.all()
            if limit:
                rows.reverse()
            return [self._serialize_message(row) for row in rows]
        finally:
            db.close()

    def save_summary(
        self, user_id: str, conversation_id: str, summary: str,
        *, through_message_id: int | None = None, metadata: dict | None = None,
    ) -> dict:
        if not summary.strip():
            raise ValueError("summary must not be empty")
        db = self.session_factory()
        try:
            conversation = self._conversation(db, user_id, conversation_id)
            if not conversation:
                raise KeyError("conversation not found")
            row = MemorySummary(
                conversation_ref_id=conversation.id, summary=summary,
                through_message_id=through_message_id, metadata_json=metadata or {},
            )
            conversation.updated_at = _utcnow()
            db.add(row)
            db.commit()
            db.refresh(row)
            return self._serialize_summary(row)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def latest_summary(self, user_id: str, conversation_id: str) -> dict | None:
        db = self.session_factory()
        try:
            conversation = self._conversation(db, user_id, conversation_id)
            if not conversation:
                return None
            row = db.query(MemorySummary).filter(
                MemorySummary.conversation_ref_id == conversation.id,
            ).order_by(MemorySummary.id.desc()).first()
            return self._serialize_summary(row) if row else None
        finally:
            db.close()

    def load_state(self, user_id: str, conversation_id: str) -> ConversationState | None:
        conversation = self.get_conversation(user_id, conversation_id)
        if not conversation:
            return None
        state = ConversationState(conversation_id=conversation_id, metadata=conversation["metadata"])
        for row in self.list_messages(user_id, conversation_id):
            state.append(row["role"], row["content"], row["evidence_refs"])
        return state

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        db = self.session_factory()
        try:
            conversation = self._conversation(db, user_id, conversation_id)
            if not conversation:
                return False
            try:
                from backend.rag_trace_service import RagTraceRecord
            except ModuleNotFoundError:
                from rag_trace_service import RagTraceRecord
            db.query(RagTraceRecord).filter(
                RagTraceRecord.conversation_ref_id == conversation.id,
            ).delete(synchronize_session=False)
            db.query(MemorySummary).filter(
                MemorySummary.conversation_ref_id == conversation.id,
            ).delete(synchronize_session=False)
            db.query(MemoryMessage).filter(
                MemoryMessage.conversation_ref_id == conversation.id,
            ).delete(synchronize_session=False)
            db.delete(conversation)
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _serialize_conversation(row: MemoryConversation) -> dict:
        return {
            "id": row.id, "user_id": row.user_id, "conversation_id": row.conversation_id,
            "metadata": dict(row.metadata_json or {}),
            "created_at": row.created_at.isoformat(), "updated_at": row.updated_at.isoformat(),
        }

    @staticmethod
    def _serialize_message(row: MemoryMessage) -> dict:
        return {
            "id": row.id, "role": row.role, "content": row.content,
            "evidence_refs": list(row.evidence_refs or []), "trace": row.trace_json,
            "created_at": row.created_at.isoformat(),
        }

    @staticmethod
    def _serialize_summary(row: MemorySummary) -> dict:
        return {
            "id": row.id, "summary": row.summary, "through_message_id": row.through_message_id,
            "metadata": dict(row.metadata_json or {}), "created_at": row.created_at.isoformat(),
        }
