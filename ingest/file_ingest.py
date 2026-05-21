"""
ingest/file_ingest.py — Multimodal file ingestion via RAGAnything.

The correct RAGAnything API for file ingestion is process_document_complete()
for individual files, or process_folder_complete() for an entire directory.

process_document_complete() handles all supported types internally:
    .txt              → fast text-only path (no OCR)
    .pdf              → full parse + multimodal processing
    .docx/.ppt/.xlsx  → Office/HTML parser
    .png/.jpg/.jpeg   → image parser + vision model captioning

Supported extensions follow RAGAnything's config.supported_file_extensions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("raganything")

# Extensions we explicitly accept — RAGAnything will handle routing internally.
# Kept here as a pre-flight guard so we log a clear warning before handing off.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".txt", ".md"}
)



async def insert_files(
    rag: Any,
    paths: list[str | Path],
) -> None:
    """Ingest a list of files via RAGAnything.process_document_complete().

    For each path:
    - Skips with WARNING if the file does not exist.
    - Skips with WARNING if the extension is not in SUPPORTED_EXTENSIONS.
    - Calls ``insert_text_file()`` for .txt/.md files.
    - Calls ``await rag.process_document_complete(str(path))`` for others.

    Args:
        rag:   A RAGAnything instance (must expose process_document_complete).
        paths: Iterable of file paths (str or Path objects).
    """
    from ingest.text_ingest import insert_text_file

    for raw_path in paths:
        path = Path(raw_path)

        if not path.exists():
            logger.warning("Skipping missing file: %s", path)
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(
                "Skipping unsupported file type '%s': %s", path.suffix, path
            )
            continue

        logger.info("Ingesting file via RAGAnything wrapper: %s", path)
        try:
            bypass = os.getenv("BYPASS_TEXT_MINERU", "True").lower() == "true"
            if bypass and path.suffix.lower() in {".txt", ".md"}:
                await insert_text_file(rag, path)
            else:
                await rag.process_document_complete(
                    file_path=str(path),
                    output_dir="./rag_storage/output",
                    parse_method="auto"
                )
            logger.info("Done: %s", path.name)
        except Exception as exc:
            logger.error(
                "Failed to ingest %s: %s", path, exc, exc_info=True
            )


async def insert_folder(
    rag: Any,
    folder: str | Path,
) -> None:
    """Ingest all supported files in a folder via RAGAnything.

    Traverses the directory, filters the supported formats, and routes each
    through the correct wrapper ingestion method (direct for text, parsed for others).

    Args:
        rag:    A RAGAnything instance.
        folder: Path to the folder to ingest.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    logger.info("Ingesting folder via RAGAnything wrapper: %s", folder)
    
    # Check recursive config if available, default to True
    recursive = True
    if hasattr(rag, "config") and hasattr(rag.config, "recursive_folder_processing"):
        recursive = rag.config.recursive_folder_processing

    pattern = "**/*" if recursive else "*"
    all_files = [
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not all_files:
        logger.info("No supported files found in folder: %s", folder)
        return

    logger.info("Found %d supported files in %s. Ingesting...", len(all_files), folder)
    await insert_files(rag, all_files)
    logger.info("Folder ingestion complete: %s", folder)
