from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

try:
    from database import Base
except ModuleNotFoundError:
    from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (UniqueConstraint("user_id", "session_id", name="uq_user_session"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_ref_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    message_type: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    rag_trace: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    session = relationship("ChatSession", back_populates="messages")


class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    chunk_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    file_type: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    parent_chunk_id: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    root_chunk_id: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    chunk_level: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_idx: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class DocumentPage(Base):
    __tablename__ = "document_pages"
    __table_args__ = (UniqueConstraint("filename", "page_number", name="uq_document_page"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    doc_name: Mapped[str] = mapped_column(String(255), default="", nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    file_type: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)
    company: Mapped[str] = mapped_column(String(255), default="", nullable=False, index=True)
    report_year: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)
    financial_document_type: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    location: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), default="", nullable=False, index=True)
    page_text: Mapped[str] = mapped_column(Text, nullable=False)
    table_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    chunk_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    embedding_cache_key: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    page_dense_embedding: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    page_tokens: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    page_numbers: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    page_years: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    page_metric_tokens: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class DocumentTable(Base):
    __tablename__ = "document_tables"

    table_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    doc_name: Mapped[str] = mapped_column(String(255), default="", nullable=False, index=True)
    file_type: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)
    table_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    title: Mapped[str] = mapped_column(Text, default="", nullable=False)
    caption: Mapped[str] = mapped_column(Text, default="", nullable=False)
    before_context: Mapped[str] = mapped_column(Text, default="", nullable=False)
    after_context: Mapped[str] = mapped_column(Text, default="", nullable=False)
    columns: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    rows: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    html: Mapped[str] = mapped_column(Text, default="", nullable=False)
    csv_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
