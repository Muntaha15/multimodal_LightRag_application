"""
ingest/text_ingest.py — Plain-text file ingestion via LightRAG bypass.

When BYPASS_TEXT_MINERU is enabled, plain text (.txt, .md) files are inserted
directly into LightRAG's native pipeline — skipping MinerU layout parsing
entirely.  This reduces ingestion time from minutes to seconds for text-only
documents.

Document ID Alignment
---------------------
LightRAG's ``ainsert()`` internally sanitizes the content string and then
computes an MD5 hash for the document ID (``doc-<hash>``).  We must replicate
that exact sequence here so the doc-status record we create in RAGAnything
uses the **same** key.  Without this, the status store ends up with two
entries for the same document (one under the hash, one under the filename),
breaking deletion, status checks, and management queries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

logger = logging.getLogger("lightrag")


async def insert_text_file(
    rag: Any,
    path: str | Path,
) -> None:
    """Ingest a plain-text file directly via LightRAG's native insertion.

    Bypasses RAGAnything's PDF-conversion and MinerU pipeline since plain text
    does not have layout features, speeding up ingestion from minutes to seconds.

    The document ID is pre-computed from the sanitized content hash so it
    matches the key that LightRAG would generate internally, keeping the
    RAGAnything status registry perfectly in sync.

    Args:
        rag:  A RAGAnything instance wrapping the LightRAG instance.
        path: Path to the .txt or .md file to ingest.

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

    # ── Pre-compute the same document ID that LightRAG will generate ──────
    # LightRAG's apipeline_enqueue_documents sanitizes the text before hashing:
    #     sanitize_text_for_encoding(doc) → compute_mdhash_id(cleaned, "doc-")
    # We replicate that here to produce a matching key.
    cleaned_content = sanitize_text_for_encoding(content)
    doc_id = compute_mdhash_id(cleaned_content, prefix="doc-")

    # Insert via LightRAG with the explicit doc ID and file path.
    await rag.lightrag.ainsert(
        input=content,
        ids=doc_id,
        file_paths=path.name,
    )

    # Register / update document status in RAGAnything's doc_status store
    # using the same content-hash doc_id so both layers agree on the key.
    await rag._upsert_doc_status(
        doc_id,
        str(path),          # file_path — used by _get_file_reference() internally
        status="processed",
        error_msg="",
    )

    logger.info("Text file ingestion complete: %s (doc_id=%s)", path.name, doc_id)
