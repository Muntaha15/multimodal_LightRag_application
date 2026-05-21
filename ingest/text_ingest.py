"""
ingest/text_ingest.py — Plain-text file ingestion via RAGAnything.

The correct RAGAnything API for file ingestion is process_document_complete().
It handles all file types including .txt, routing them through the appropriate
pipeline (fast text path for .txt, full multimodal parsing for PDF/DOCX/images).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("lightrag")


async def insert_text_file(
    rag: Any,
    path: str | Path,
) -> None:
    """Ingest a plain-text file directly via LightRAG's native insertion.

    Bypasses RAGAnything's PDF-conversion and MinerU pipeline since plain text
    does not have layout features, speeding up ingestion from minutes to seconds.

    Args:
        rag:  A RAGAnything instance wrapping the LightRAG instance.
        path: Path to the .txt file to ingest.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    logger.info(
        "Ingesting plain text directly via LightRAG (bypassing MinerU/PDF conversion): %s (%d bytes)",
        path,
        path.stat().st_size,
    )
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Use LightRAG's native insert method
    await rag.lightrag.ainsert(
        input=content,
        file_paths=path.name
    )

    # Register document status in RAGAnything so the registry remains in sync
    doc_id = rag._get_file_reference(str(path))
    await rag._upsert_doc_status(
        doc_id=doc_id,
        file_name=path.name,
        status="success",
        error_msg=""
    )
    
    logger.info("Text file ingestion complete: %s", path.name)

