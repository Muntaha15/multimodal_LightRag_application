"""
rag/llm.py — LLM function wrappers for LightRAG and RAGAnything.

custom_llm_func
    Wraps openai_complete_if_cache for the main text LLM via vLLM.
    Injects ENTITY_EXTRACTION_ADDENDUM into EVERY system prompt so that
    entity/relation extraction stays scoped to named, concrete entities.

vision_llm_func
    Targets the vision model (VISION_MODEL env var) via vLLM.
    Accepts image_data as list[dict] with keys:
        type       — "image"
        data       — base64-encoded image string
        media_type — e.g. "image/png"
    Converts base64 strings into OpenAI-compatible image_url content parts
    and passes them to openai_complete_if_cache.
"""

from __future__ import annotations

import os
import logging
from typing import Any

from lightrag.llm.openai import openai_complete_if_cache

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
    """Wrap openai_complete_if_cache, appending entity-extraction constraints.

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
        **kwargs:          Forwarded verbatim to openai_complete_if_cache
                           (model, base_url, api_key, timeout, …).

    Returns:
        The model's text response as a plain string.
    """
    if system_prompt and keyword_extraction:
        system_prompt = system_prompt + ENTITY_EXTRACTION_ADDENDUM

    # keyword_extraction MUST flow through **kwargs, NOT as a named arg.
    # Upstream openai_complete_if_cache() does:
    #     keyword_extraction = kwargs.pop("keyword_extraction", None)
    # If we also pass it as a named parameter, Python binds it to the
    # function signature and kwargs.pop returns None — silently disabling
    # the JSON format toggle during entity extraction.
    kwargs["keyword_extraction"] = keyword_extraction

    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-14B-Instruct"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=os.getenv("LLM_BINDING_HOST", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("LLM_BINDING_API_KEY", "not_needed"),
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
    Converts base64 strings into OpenAI-compatible image_url content parts
    and passes them to openai_complete_if_cache targeting the VISION_MODEL.

    Args:
        prompt:           Textual prompt or question about the images.
        system_prompt:    Optional system instruction.
        history_messages: Prior conversation turns.
        image_data:       Single image as base64 string (or list of dicts fallback).
        messages:         List of message dicts (from VLM enhanced queries).
        **kwargs:         Extra kwargs — model/base_url/api_key overrides are
                          stripped to prevent conflicts.

    Returns:
        The vision model's text response.
    """
    images_b64: list[str] = []
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
                            images_b64.append(url.split("base64,")[-1])
            elif isinstance(content, str):
                text_parts.append(content)
        
        if not prompt and text_parts:
            extracted_prompt = "\n".join(text_parts)
            
    elif image_data:
        if isinstance(image_data, str):
            images_b64.append(image_data)
        elif isinstance(image_data, list):
            for item in image_data:
                if isinstance(item, dict) and "data" in item:
                    images_b64.append(item["data"])

    # Strip keys we're explicitly setting to avoid kwarg conflicts
    _safe_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in ("model", "base_url", "api_key", "images")
    }

    # Build OpenAI-compatible multimodal content parts if images are present
    if images_b64:
        content_parts: list[dict[str, Any]] = [
            {"type": "text", "text": extracted_prompt}
        ]
        for b64_img in images_b64:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64_img}",
                },
            })

        # For OpenAI-compatible VLM, we send the multimodal content
        # via history_messages with the proper content format.
        vlm_messages = []
        if system_prompt:
            vlm_messages.append({"role": "system", "content": system_prompt})
        vlm_messages.append({"role": "user", "content": content_parts})

        return await openai_complete_if_cache(
            os.getenv("VISION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
            "",  # prompt is embedded in messages
            system_prompt=None,  # already in messages
            history_messages=vlm_messages,
            base_url=os.getenv("LLM_BINDING_HOST", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("LLM_BINDING_API_KEY", "not_needed"),
            **_safe_kwargs,
        )
    else:
        # No images — standard text-only call to vision model
        return await openai_complete_if_cache(
            os.getenv("VISION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
            extracted_prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=os.getenv("LLM_BINDING_HOST", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("LLM_BINDING_API_KEY", "not_needed"),
            **_safe_kwargs,
        )
