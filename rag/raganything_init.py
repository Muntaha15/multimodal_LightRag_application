"""
rag/raganything_init.py — RAGAnything wrapper factory (Option B outer layer).

Option B — inject an existing LightRAG instance
------------------------------------------------
Instead of letting RAGAnything construct its own internal LightRAG (Option A),
we pass in a pre-configured instance.  This keeps all LightRAG settings in a
single place (rag/lightrag_init.py) and avoids parameter duplication.

Wiring:
    lightrag    = await initialize_lightrag()          # inner layer
    rag         = initialize_raganything(lightrag)     # outer layer  ← here

The RAGAnything instance exposes the same .ainsert() / .aquery() surface as
LightRAG, plus .ainsert_file() for multimodal documents (PDF, DOCX, images).
vision_model_func is the ONLY parameter added at this layer.
"""

from __future__ import annotations

import logging

from raganything import RAGAnything, RAGAnythingConfig

from rag.llm import vision_llm_func

logger = logging.getLogger("raganything")


def initialize_raganything(lightrag_instance) -> RAGAnything:
    """Wrap an existing LightRAG instance with RAGAnything (Option B).

    Args:
        lightrag_instance: A fully initialised LightRAG object returned by
                           ``await initialize_lightrag()``.

    Returns:
        A RAGAnything instance ready for multimodal ingestion (.ainsert_file)
        and querying (.aquery).  All graph/vector/LLM config is inherited
        from the injected LightRAG instance.
    """
    logger.info(
        "Initialising RAGAnything (Option B) — injecting existing LightRAG instance."
    )
    
    config = RAGAnythingConfig(
        working_dir=lightrag_instance.working_dir,
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    return RAGAnything(
        config=config,
        lightrag=lightrag_instance,
        vision_model_func=vision_llm_func,
    )
