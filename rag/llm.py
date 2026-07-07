"""
rag/llm.py — LLM function wrappers for LightRAG and RAGAnything.

custom_llm_func
    Wraps ollama_model_complete for the main text LLM.
    Injects ENTITY_EXTRACTION_ADDENDUM into EVERY system prompt so that
    entity/relation extraction stays scoped to named, concrete entities.

vision_llm_func
    Targets the vision model (VISION_MODEL env var).
    Accepts image_data as list[dict] with keys:
        type       — "image"
        data       — base64-encoded image string
        media_type — e.g. "image/png"
    Extracts base64 strings and passes them to ollama_model_complete
    as the images= list with num_ctx=4096.
"""

from __future__ import annotations

import os
import logging
from typing import Any

from lightrag.llm.ollama import ollama_model_complete

logger = logging.getLogger("lightrag")

# ---------------------------------------------------------------------------
# Entity-extraction guardrails injected into every system prompt
# ---------------------------------------------------------------------------

ENTITY_EXTRACTION_ADDENDUM: str = """

---Additional Extraction Constraints---

1. SCOPE: Extract named entities, concrete entities, and significant high-level concepts, themes, methods, or processes.
   Acceptable types:
   - A named person, creature, place, or organization.
   - A named artifact, document, or law.
   - Significant thematic concepts, theories, methods, or processes (e.g., "Industrialization", "Poverty", "Binary Search", "Cryptography").
   NOT acceptable: trivial everyday objects, common foods, minor transient emotions, or descriptive fragments.
   Examples of what to REJECT: "Mashed Potatoes", "Feather-Beds", "Wicker Baskets", "Happy Mood", "Quickly Running".

2. CANONICAL NAMES: When the same entity appears under multiple names or
   titles, use only the most complete and formal name.
   Example: "Scrooge", "Master Scrooge", and "Ebenezer Scrooge"
            → use "Ebenezer Scrooge" only.
   Example: "Marley", "Old Jacob Marley", "Marley's Ghost"
            → use "Jacob Marley" only.

3. DOCUMENT SCOPE: Only extract entities that are part of the main narrative
   or content. DO NOT extract copyright notices, legal metadata, publisher
   names, ISBNs, addresses, URLs, or license terms.

4. HALLUCINATION CHECK: Every entity you extract must be explicitly named in
   the provided text chunk. Do not infer entities from general knowledge or
   introduce names not present in the text.
"""


# ---------------------------------------------------------------------------
# Main LLM wrapper
# ---------------------------------------------------------------------------

async def custom_llm_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] = [],
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> str:
    """Wrap ollama_model_complete, appending entity-extraction constraints.

    ENTITY_EXTRACTION_ADDENDUM is appended ONLY when keyword_extraction=True,
    which LightRAG sets during graph construction (entity extraction, relation
    extraction).  During query-time answer synthesis keyword_extraction is
    False, so the addendum is skipped — otherwise instructions like "Only
    extract NAMED entities" would confuse the model when it's supposed to be
    generating a natural-language answer.

    Args:
        prompt:            The user / task prompt.
        system_prompt:     Optional system-level instruction from LightRAG.
        history_messages:  Prior conversation turns.
        keyword_extraction: Flag set by LightRAG during graph construction.
        **kwargs:          Forwarded verbatim to ollama_model_complete
                           (model, host, options, timeout, …).

    Returns:
        The model's text response as a plain string.
    """
    if system_prompt and keyword_extraction:
        system_prompt = system_prompt + ENTITY_EXTRACTION_ADDENDUM

    # keyword_extraction MUST flow through **kwargs, NOT as a named arg.
    # Upstream ollama_model_complete() does:
    #     keyword_extraction = kwargs.pop("keyword_extraction", None)
    # If we also pass it as a named parameter, Python binds it to the
    # function signature and kwargs.pop returns None — silently disabling
    # the JSON format toggle during entity extraction.
    kwargs["keyword_extraction"] = keyword_extraction

    return await ollama_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Vision LLM wrapper (used by RAGAnything)
# ---------------------------------------------------------------------------

async def vision_llm_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] = [],
    image_data: Any | None = None,
    messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    """Vision-capable LLM wrapper for RAGAnything multimodal processing.

    Accepts image_data as a base64 string or messages array for VLM enhanced queries.
    Extracts base64 strings and forwards them to ollama_model_complete
    via the images= kwarg, targeting VISION_MODEL with a 4096-token context.

    Args:
        prompt:           Textual prompt or question about the images.
        system_prompt:    Optional system instruction.
        history_messages: Prior conversation turns.
        image_data:       Single image as base64 string (or list of dicts fallback).
        messages:         List of message dicts (from VLM enhanced queries).
        **kwargs:         Extra kwargs — model/host/options overrides are
                          stripped to prevent conflicts.

    Returns:
        The vision model's text response.
    """
    images: list[str] = []
    extracted_prompt = prompt

    if messages:
        text_parts = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if "base64," in url:
                            images.append(url.split("base64,")[-1])
            elif isinstance(content, str):
                text_parts.append(content)
        
        if not prompt and text_parts:
            extracted_prompt = "\n".join(text_parts)
            
    elif image_data:
        if isinstance(image_data, str):
            images.append(image_data)
        elif isinstance(image_data, list):
            for item in image_data:
                if isinstance(item, dict) and "data" in item:
                    images.append(item["data"])

    # Strip keys we're explicitly setting to avoid kwarg conflicts
    _safe_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in ("model", "host", "options", "images")
    }

    return await ollama_model_complete(
        extracted_prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        model=os.getenv("VISION_MODEL", "qwen2.5vl:7b"),
        host=os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
        options={"num_ctx": 4096},
        images=images if images else None,
        **_safe_kwargs,
    )
