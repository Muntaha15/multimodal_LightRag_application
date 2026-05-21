"""
rag/embeddings.py — EmbeddingFunc factory wired to Ollama.

Why ollama_embed.func (not ollama_embed directly)?
    ollama_embed is decorated with @wrap_embedding_func_with_attrs, which
    already wraps the raw async callable inside an EmbeddingFunc object.
    If we passed ollama_embed straight into another EmbeddingFunc we would
    get double-wrapping — the outer EmbeddingFunc would call the inner one
    as if it were a plain coroutine function, leading to type errors.

    Accessing .func gives us the original unwrapped async callable so we
    can construct our own EmbeddingFunc with the correct embedding_dim and
    max_token_size without any layering issues.
"""

from __future__ import annotations

import os
from functools import partial

from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc


def get_embedding_func() -> EmbeddingFunc:
    """Build an EmbeddingFunc instance wired to the configured Ollama model.

    Environment variables consumed:
        EMBEDDING_MODEL        — model name (default: nomic-embed-text:latest)
        EMBEDDING_DIM          — vector dimension (default: 768)
        MAX_EMBED_TOKENS       — max tokens per chunk (default: 8192)
        EMBEDDING_BINDING_HOST — Ollama base URL (default: http://localhost:11434)

    Returns:
        A fully configured EmbeddingFunc ready to pass to LightRAG.
    """
    return EmbeddingFunc(
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
        max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
        func=partial(
            ollama_embed.func,   # unwrapped — avoids double EmbeddingFunc wrapping
            embed_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
            host=os.getenv("EMBEDDING_BINDING_HOST", "http://127.0.0.1:11434"),
        ),
    )
