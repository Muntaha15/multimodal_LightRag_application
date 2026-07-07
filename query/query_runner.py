"""
query/query_runner.py — Multi-mode query runner for LightRAG / RAGAnything.

Provides two query helpers:

    run_all_modes   — CLI demo: run all 4 modes, print streamed output.
    aquery_with_vlm_stream — Streamable VLM-enhanced query that returns
                              BOTH structured retrieval data (entities,
                              relationships) AND an async-generator of
                              answer tokens.

VLM Streaming Design
--------------------
RAGAnything's native ``aquery_vlm_enhanced()`` blocks streaming because it:

    1. Builds its own ``QueryParam(only_need_prompt=True)``
    2. Ignores the caller's ``stream=True`` and ``enable_rerank``
    3. Returns a plain string (not an async generator)

To solve this, ``aquery_with_vlm_stream()`` replicates the VLM flow at the
wrapper layer, calling:
    a. ``lightrag.aquery_llm()`` with ``only_need_prompt=True``  → retrieval data
    b. ``rag._process_image_paths_for_vlm()``                   → base64 images
    c. ``rag.vision_model_func()``  via ``ollama_model_complete`` with ``stream=True``
       → yields tokens as an async generator

When no valid images are found in the retrieved context, the wrapper falls
back to the standard text-only streaming path via ``lightrag.aquery_llm()``
with ``stream=True``, preserving all existing behaviour.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, AsyncGenerator

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


# ---------------------------------------------------------------------------
# Streamable VLM Query Handler
# ---------------------------------------------------------------------------


async def aquery_with_vlm_stream(
    rag: Any,
    lightrag: Any,
    query: str,
    mode: str = "hybrid",
    enable_rerank: bool = False,
) -> dict[str, Any]:
    """Run a VLM-enhanced query that streams the answer AND returns subgraph data.

    This function bridges the gap between LightRAG's ``aquery_llm()`` (which
    returns structured retrieval data + streaming answers) and RAG-Anything's
    ``aquery_vlm_enhanced()`` (which processes images but blocks streaming).

    The return dict matches the ``aquery_llm()`` format so the Streamlit Tab 1
    UI can consume it without changes:

        {
            "status": "success",
            "data":   {"entities": [...], "relationships": [...]},
            "llm_response": {
                "content":           None,
                "response_iterator": <async generator>,
                "is_streaming":      True,
            },
        }

    Args:
        rag:           RAGAnything instance (with vision_model_func set).
        lightrag:      Inner LightRAG instance.
        query:         The user question.
        mode:          LightRAG search mode (hybrid/local/global/naive).
        enable_rerank: Whether to enable reranking in the retrieval step.

    Returns:
        A dict compatible with ``aquery_llm()``'s response format.
    """
    has_vlm = (
        hasattr(rag, "vision_model_func")
        and rag.vision_model_func is not None
    )

    # If VLM is not available, fall back to the standard streaming path.
    if not has_vlm:
        logger.info("VLM not available — falling back to standard streaming query.")
        param = QueryParam(mode=mode, stream=True, enable_rerank=enable_rerank)
        return await lightrag.aquery_llm(query, param=param)

    # ── Step 1: Retrieve context + subgraph data (no LLM generation yet) ──
    prompt_param = QueryParam(
        mode=mode,
        only_need_prompt=True,
        enable_rerank=enable_rerank,
    )
    prompt_result = await lightrag.aquery_llm(query, param=prompt_param)

    if not prompt_result or prompt_result.get("status") != "success":
        logger.warning("Prompt retrieval failed — falling back to standard query.")
        param = QueryParam(mode=mode, stream=True, enable_rerank=enable_rerank)
        return await lightrag.aquery_llm(query, param=param)

    # Extract the raw context prompt and structured entity/relationship data
    raw_prompt = prompt_result.get("llm_response", {}).get("content", "")
    retrieval_data = prompt_result.get("data", {})

    if not raw_prompt:
        logger.warning("Empty retrieval prompt — falling back to standard query.")
        param = QueryParam(mode=mode, stream=True, enable_rerank=enable_rerank)
        return await lightrag.aquery_llm(query, param=param)

    # ── Step 2: Check for images in the retrieved context ──────────────────
    try:
        # Ensure RAG-Anything's LightRAG + processors are initialised
        await rag._ensure_lightrag_initialized()

        enhanced_prompt, images_found = await rag._process_image_paths_for_vlm(
            raw_prompt
        )
    except Exception as exc:
        logger.warning("Image processing failed (%s) — falling back.", exc)
        images_found = 0

    if not images_found:
        # No images → standard text streaming (preserves subgraph data)
        logger.info("No images in retrieved context — using text-only streaming.")
        stream_param = QueryParam(mode=mode, stream=True, enable_rerank=enable_rerank)
        stream_result = await lightrag.aquery_llm(query, param=stream_param)
        # Merge the retrieval data from the prompt step (which has complete
        # entity/relationship info) since stream_result may also have it.
        if retrieval_data:
            stream_result.setdefault("data", {}).update(retrieval_data)
        return stream_result

    # ── Step 3: Build VLM messages and stream the response ─────────────────
    logger.info("Found %d image(s) — using VLM-enhanced streaming query.", images_found)
    messages = rag._build_vlm_messages_with_images(enhanced_prompt, query, None)

    async def _vlm_stream() -> AsyncGenerator[str, None]:
        """Stream tokens from the vision model via Ollama's low-level chat API.

        We use ``_ollama_model_if_cache`` directly because:
          • ``ollama_model_complete`` expects ``hashing_kv`` from the LightRAG
            global config and doesn't accept model/host as plain kwargs.
          • ``_ollama_model_if_cache`` supports ``stream=True``, ``host``,
            ``model``, and the standard message format that Ollama uses.
        """
        from lightrag.llm.ollama import _ollama_model_if_cache

        # Extract images and text from the VLM message content parts
        user_message = messages[1]
        content = user_message["content"]
        system_prompt = messages[0]["content"] if messages else None

        images_b64: list[str] = []
        text_parts: list[str] = []

        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if "base64," in url:
                        images_b64.append(url.split("base64,")[-1])
        elif isinstance(content, str):
            text_parts.append(content)

        prompt_text = "\n".join(text_parts) if text_parts else query

        # Build Ollama-native message format with images
        ollama_messages: list[dict[str, Any]] = []
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})

        user_msg: dict[str, Any] = {"role": "user", "content": prompt_text}
        if images_b64:
            user_msg["images"] = images_b64
        ollama_messages.append(user_msg)

        vision_model = os.getenv("VISION_MODEL", "qwen2.5vl:7b")
        host = os.getenv("LLM_BINDING_HOST", "http://127.0.0.1:11434")
        timeout = int(os.getenv("TIMEOUT", "900"))

        # _ollama_model_if_cache builds its own messages from prompt +
        # system_prompt, but we need the full multimodal message list.
        # So we call it with the prompt and pass messages via history.
        # Actually, looking at the code more carefully, it constructs
        # messages = [system, *history, user]. We want to send our own
        # messages directly, so we use history_messages for the user msg
        # and set prompt to empty.
        import ollama as ollama_lib

        ollama_client = ollama_lib.AsyncClient(host=host, timeout=timeout)
        try:
            response = await ollama_client.chat(
                model=vision_model,
                messages=ollama_messages,
                stream=True,
                options={"num_ctx": 4096},
            )
            async for chunk in response:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
        except Exception as e:
            logger.error("VLM streaming failed: %s", e, exc_info=True)
            yield f"\n[VLM Error: {e}]"
        finally:
            try:
                await ollama_client._client.aclose()
            except Exception:
                pass

    return {
        "status": "success",
        "message": "VLM-enhanced streaming response",
        "data": retrieval_data,
        "metadata": {},
        "llm_response": {
            "content": None,
            "response_iterator": _vlm_stream(),
            "is_streaming": True,
        },
    }

