"""
rag/reranker.py — Reranker implementations for LightRAG.

Two backends are provided (both from your existing code):

    local_rerank_func  — CrossEncoder from sentence_transformers (GPU via CUDA)
    flag_rerank_func   — FlagReranker from FlagEmbedding (fp16, faster)

Both are fully implemented but lazily loaded and gated behind ENABLE_RERANK.

One-line toggle
---------------
To activate reranking, change:
    ENABLE_RERANK: bool = False
to:
    ENABLE_RERANK: bool = True

To switch backend, change ACTIVE_RERANKER to "cross_encoder" or "flag".
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

logger = logging.getLogger("lightrag")

# ── Configuration ──────────────────────────────────────────────────────────

ENABLE_RERANK: bool = False        # ← flip to True to activate
ACTIVE_RERANKER: str = "flag"      # "flag" | "cross_encoder"
RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"

# ── Lazy singletons (only instantiated when ENABLE_RERANK is True) ─────────

_cross_encoder = None
_flag_reranker = None


def _load_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading CrossEncoder reranker: %s", RERANK_MODEL)
        _cross_encoder = CrossEncoder(RERANK_MODEL, device="cuda")
    return _cross_encoder


def _load_flag_reranker():
    global _flag_reranker
    if _flag_reranker is None:
        from FlagEmbedding import FlagReranker
        logger.info("Loading FlagReranker: %s", RERANK_MODEL)
        _flag_reranker = FlagReranker(RERANK_MODEL, use_fp16=True)
    return _flag_reranker


# ── Reranker implementations ───────────────────────────────────────────────

async def local_rerank_func(
    query: str,
    documents: list[dict],
    top_k: Optional[int] = None,
    **kwargs,
) -> list[dict]:
    """Rerank documents using a CrossEncoder from sentence_transformers.

    Runs the blocking .predict() call in a thread-pool executor so the
    event loop stays non-blocking.

    Args:
        query:     The user query string.
        documents: List of document dicts (must contain "content" or "text").
        top_k:     If set, return only the top-k ranked documents.

    Returns:
        Documents sorted by descending rerank_score, truncated to top_k.
    """
    if not documents:
        return documents

    encoder = _load_cross_encoder()
    pairs = [(query, doc.get("content", doc.get("text", ""))) for doc in documents]

    loop = asyncio.get_event_loop()
    scores = await loop.run_in_executor(None, encoder.predict, pairs)

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    ranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k] if top_k else ranked


async def flag_rerank_func(
    query: str,
    documents: list[dict],
    top_k: Optional[int] = None,
    **kwargs,
) -> list[dict]:
    """Rerank documents using FlagReranker from FlagEmbedding (fp16).

    compute_score() is synchronous; called directly (fast enough for batch
    sizes typical in RAG pipelines). Switch to run_in_executor if latency
    becomes an issue with very large document sets.

    Args:
        query:     The user query string.
        documents: List of document dicts (must contain "content").
        top_k:     If set, return only the top-k ranked documents.

    Returns:
        Documents sorted by descending rerank_score, truncated to top_k.
    """
    if not documents:
        return documents

    reranker = _load_flag_reranker()
    pairs = [(query, doc.get("content", "")) for doc in documents]
    scores = reranker.compute_score(pairs)

    # compute_score returns a float when len(pairs)==1
    if isinstance(scores, float):
        scores = [scores]

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    ranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k] if top_k else ranked


# ── Public factory ─────────────────────────────────────────────────────────

def get_rerank_func() -> Optional[Callable]:
    """Return the active reranker, or None if reranking is disabled.

    Controlled by module-level flags ENABLE_RERANK and ACTIVE_RERANKER.
    No heavy models are loaded unless ENABLE_RERANK is True.

    Returns:
        An async reranker callable, or None.
    """
    if not ENABLE_RERANK:
        return None

    if ACTIVE_RERANKER == "cross_encoder":
        return local_rerank_func
    elif ACTIVE_RERANKER == "flag":
        return flag_rerank_func
    else:
        logger.warning(
            "Unknown ACTIVE_RERANKER '%s', falling back to flag.", ACTIVE_RERANKER
        )
        return flag_rerank_func
