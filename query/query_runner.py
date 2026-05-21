"""
query/query_runner.py — Multi-mode query runner for LightRAG / RAGAnything.

Runs the same query through all four QueryParam modes:
    naive → local → global → hybrid

stream=True is used for all modes; both plain-string and async-generator
responses are handled correctly (matching your existing print_stream pattern).
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from lightrag import QueryParam
from rag.reranker import ENABLE_RERANK

logger = logging.getLogger("lightrag")

_MODES = ("naive", "local", "global", "hybrid")


async def _print_stream(stream: Any) -> None:
    """Consume and print an async generator stream chunk by chunk."""
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()  # newline after stream ends


async def run_all_modes(rag: Any, query: str) -> None:
    """Run a query through all four LightRAG retrieval modes and print results.

    Modes executed in order:
        naive   — direct vector similarity (no graph traversal)
        local   — entity-centric local graph context
        global  — community-level global graph context
        hybrid  — local + global combined

    stream=True is set for all modes. If the response is an async generator
    it is streamed to stdout; if it is a plain string it is printed directly.

    Args:
        rag:   A LightRAG or RAGAnything instance exposing .aquery().
        query: The question to answer across all modes.
    """
    for mode in _MODES:
        print(f"\n{'='*45}")
        print(f" Query mode: {mode}")
        print(f"{'='*45}")

        try:
            # vlm_enhanced=False is critical here. Without it, RAGAnything
            # auto-detects that vision_model_func is set and silently
            # redirects to aquery_vlm_enhanced(), which:
            #   1. Builds its own QueryParam(only_need_prompt=True)
            #   2. Ignores our stream=True and enable_rerank kwargs
            #   3. Returns a plain string instead of an async generator
            # By explicitly disabling VLM enhancement, we ensure our kwargs
            # flow into QueryParam correctly and streaming works as expected.
            resp = await rag.aquery(
                query,
                mode=mode,
                stream=True,
                enable_rerank=ENABLE_RERANK,
                vlm_enhanced=False,
            )
            if inspect.isasyncgen(resp):
                await _print_stream(resp)
            else:
                print(resp)
        except Exception as exc:
            logger.error("Query failed in mode '%s': %s", mode, exc, exc_info=True)
            print(f"[ERROR in {mode} mode] {exc}")
