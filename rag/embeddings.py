"""
rag/embeddings.py — EmbeddingFunc factory wired to vLLM (OpenAI-compatible API).

Why openai_embed.func (not openai_embed directly)?
    openai_embed is decorated with @wrap_embedding_func_with_attrs, which
    already wraps the raw async callable inside an EmbeddingFunc object.
    If we passed openai_embed straight into another EmbeddingFunc we would
    get double-wrapping — the outer EmbeddingFunc would call the inner one
    as if it were a plain coroutine function, leading to type errors.

    Accessing .func gives us the original unwrapped async callable so we
    can construct our own EmbeddingFunc with the correct embedding_dim and
    max_token_size without any layering issues.
"""

from __future__ import annotations

import os
from functools import partial

from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc


def get_embedding_func() -> EmbeddingFunc:
    """Build an EmbeddingFunc instance wired to the configured vLLM embedding endpoint.

    Environment variables consumed:
        EMBEDDING_MODEL        — model name (default: nomic-ai/nomic-embed-text-v1.5)
        EMBEDDING_DIM          — vector dimension (default: 768)
        MAX_EMBED_TOKENS       — max tokens per chunk (default: 8192)
        EMBEDDING_BINDING_HOST — vLLM base URL (default: http://127.0.0.1:8001/v1)
        EMBEDDING_BINDING_API_KEY — API key (default: not_needed)

    Returns:
        A fully configured EmbeddingFunc ready to pass to LightRAG.
    """
    return EmbeddingFunc(
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
        max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
        func=partial(
            openai_embed.func,   # unwrapped — avoids double EmbeddingFunc wrapping
            model=os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
            base_url=os.getenv("EMBEDDING_BINDING_HOST", "http://127.0.0.1:8001/v1"),
            api_key=os.getenv("EMBEDDING_BINDING_API_KEY", "not_needed"),
        ),
    )
