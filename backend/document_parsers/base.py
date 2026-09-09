"""Common interface for optional document parser backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .unified_document import ParsedDocument


class DocumentParser(ABC):
    """Parse one source file into the ingestion-neutral document model."""

    backend_name: str
    supported_extensions: frozenset[str]

    def supports(self, file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() in self.supported_extensions

    @abstractmethod
    def parse(self, file_path: str | Path, *, filename: str | None = None) -> ParsedDocument:
        """Parse ``file_path`` without applying chunking or persistence."""

