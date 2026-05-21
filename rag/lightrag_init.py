"""
rag/lightrag_init.py — LightRAG instance factory (Option B inner layer).

This module owns ALL LightRAG configuration.  No LightRAG constructor params
should appear in raganything_init.py — RAGAnything simply wraps the instance
produced here (see rag/raganything_init.py for the outer-layer details).

Option B summary
----------------
    Inner  →  LightRAG  (graph DB, vector stores, LLM, embeddings)
    Outer  →  RAGAnything(lightrag=<this instance>, vision_model_func=…)
"""

from __future__ import annotations

import os
import logging

from lightrag import LightRAG

from rag.llm import custom_llm_func
from rag.embeddings import get_embedding_func
from rag.reranker import get_rerank_func, ENABLE_RERANK

logger = logging.getLogger("lightrag")


async def initialize_lightrag() -> LightRAG:
    """Create, configure, and initialise a LightRAG instance.

    Reads all settings from environment variables (see .env template).
    Calls ``await lightrag.initialize_storages()`` before returning so the
    caller receives a fully ready instance — no extra init step required.

    Key design decisions
    --------------------
    * custom_llm_func injects ENTITY_EXTRACTION_ADDENDUM only during graph
      construction (keyword_extraction=True), not during query-time synthesis.
    * EmbeddingFunc uses ollama_embed.func (unwrapped) via get_embedding_func()
      to avoid double-wrapping (see rag/embeddings.py for details).
    * llm_model_max_async=2 / max_parallel_insert=2 — conservative concurrency
      that keeps Ollama stable on a single-GPU workstation.
    * "language": "English" in addon_params ensures extraction prompts are
      generated in English regardless of document language.
    * rerank_model_func is a post-init attribute on the LightRAG instance —
      NOT a constructor param. Set it after LightRAG() if reranking is enabled.
    * enable_rerank is a QueryParam field (per-query flag) — see query_runner.py.

    Returns:
        A fully initialised LightRAG instance.
    """
    working_dir = os.getenv("WORKING_DIR", "./storage/dickens_v1")
    os.makedirs(working_dir, exist_ok=True)

    rerank_func = get_rerank_func()
    if ENABLE_RERANK and rerank_func:
        logger.info("Reranking ENABLED — backend: %s", rerank_func.__name__)
    else:
        logger.info("Reranking DISABLED (set ENABLE_RERANK=True in rag/reranker.py to activate).")

    lightrag = LightRAG(
        working_dir=working_dir,
        graph_storage=os.getenv("GRAPH_STORAGE", "NetworkXStorage"),

        # ── LLM ───────────────────────────────────────────────────────────
        llm_model_func=custom_llm_func,
        llm_model_name=os.getenv("LLM_MODEL", "qwen2.5-coder:14b"),
        llm_model_max_async=2,
        max_parallel_insert=2,

        # ── Graph construction params ──────────────────────────────────────
        addon_params={
            "insert_batch_size": 10,
            "entity_extraction_worker_timeout": 1800,
            "worker_timeout": 1800,
            "llm_timeout": 1800,
            "language": "English",
            "entity_types": ["person", "location", "organization"],
        },

        summary_max_tokens=4096,

        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
            "options": {"num_ctx": 16384},
            "timeout": int(os.getenv("TIMEOUT", "900")),
        },

        # ── Embeddings ────────────────────────────────────────────────────
        # ollama_embed.func (accessed inside get_embedding_func) is the raw
        # async callable — bypasses the @wrap_embedding_func_with_attrs
        # decorator to prevent double-wrapping in our own EmbeddingFunc.
        embedding_func=get_embedding_func(),
    )

    # ── Reranker ──────────────────────────────────────────────────────────
    # rerank_model_func is a post-init instance attribute, not a constructor
    # param. enable_rerank is a per-query flag on QueryParam (see query_runner).
    if ENABLE_RERANK and rerank_func:
        lightrag.rerank_model_func = rerank_func
        logger.info("Reranking ENABLED — backend: %s", rerank_func.__name__)
    else:
        logger.info(
            "Reranking DISABLED. Set ENABLE_RERANK=True in rag/reranker.py "
            "and pass enable_rerank=True in QueryParam to activate."
        )

    await lightrag.initialize_storages()

    # initialize_pipeline_status() sets up the module-level pipeline state
    # dict (e.g. history_messages, processing flags) that LightRAG reads
    # during entity extraction.  initialize_storages() only sets up DB
    # backends — without this call the state dict is empty and you get
    # KeyError: 'history_messages' on the first extraction attempt.
    from lightrag.kg.shared_storage import initialize_pipeline_status
    await initialize_pipeline_status()

    logger.info("LightRAG initialised — working_dir=%s", working_dir)
    return lightrag
