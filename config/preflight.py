"""
config/preflight.py — Pre-flight checks for the RAG pipeline.

Verifies external service dependencies (Ollama) are reachable BEFORE
the pipeline begins initialising heavy objects.  Failing fast here
produces a clear, actionable error instead of a deep traceback from
inside the embedding or LLM stack.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

logger = logging.getLogger("lightrag")


class OllamaNotReachableError(RuntimeError):
    """Raised when the Ollama service cannot be contacted."""


async def check_ollama_connectivity(
    host: str | None = None,
    timeout: float = 10.0,
) -> dict:
    """Verify that Ollama is running and list locally available models.

    Hits the ``/api/tags`` endpoint (lightweight, no GPU work) to confirm
    the service is up and returns the model catalogue.

    Args:
        host:    Ollama base URL.  Falls back to EMBEDDING_BINDING_HOST,
                 then LLM_BINDING_HOST, then ``http://127.0.0.1:11434``.
        timeout: Seconds to wait before declaring unreachable.

    Returns:
        A dict with keys ``host`` (str) and ``models`` (list[str]).

    Raises:
        OllamaNotReachableError: If the service is not reachable.
    """
    import httpx  # lightweight; already a transitive dep of ollama

    if host is None:
        host = os.getenv(
            "EMBEDDING_BINDING_HOST",
            os.getenv("LLM_BINDING_HOST", "http://127.0.0.1:11434"),
        )

    # Normalise: strip trailing slash so /api/tags joins cleanly
    host = host.rstrip("/")
    url = f"{host}/api/tags"

    logger.info("Pre-flight: checking Ollama connectivity at %s …", host)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.ConnectError as exc:
        raise OllamaNotReachableError(
            f"\n{'='*60}\n"
            f"  ❌  Ollama is NOT reachable at {host}\n"
            f"{'='*60}\n"
            f"  Possible fixes:\n"
            f"    1. Start Ollama:   ollama serve\n"
            f"    2. Verify the host/port in .env:\n"
            f"         LLM_BINDING_HOST={host}\n"
            f"         EMBEDDING_BINDING_HOST={host}\n"
            f"    3. If Ollama runs on a different machine, update the\n"
            f"       host to its reachable IP/hostname.\n"
            f"    4. Download Ollama: https://ollama.com/download\n"
            f"{'='*60}\n"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise OllamaNotReachableError(
            f"Ollama at {host} returned HTTP {exc.response.status_code}. "
            f"Is a reverse proxy interfering?"
        ) from exc
    except Exception as exc:
        raise OllamaNotReachableError(
            f"Unexpected error reaching Ollama at {host}: {exc}"
        ) from exc

    # Parse model names from the /api/tags response
    data = resp.json()
    model_names = [m.get("name", "?") for m in data.get("models", [])]
    logger.info(
        "Pre-flight: Ollama is UP — %d model(s) available: %s",
        len(model_names),
        ", ".join(model_names) if model_names else "(none pulled yet)",
    )
    return {"host": host, "models": model_names}


async def verify_required_models(
    available_models: list[str],
) -> None:
    """Warn (not crash) if required models are not pulled in Ollama.

    Reads model names from .env (LLM_MODEL, EMBEDDING_MODEL, VISION_MODEL)
    and checks them against the list returned by /api/tags.

    Args:
        available_models: Model names returned by check_ollama_connectivity().
    """
    required = {
        "LLM_MODEL": os.getenv("LLM_MODEL", "qwen2.5-coder:14b"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
        "VISION_MODEL": os.getenv("VISION_MODEL", "qwen2.5vl:7b"),
    }

    # Ollama model names are case-insensitive; /api/tags returns full
    # names like "nomic-embed-text:latest".
    available_lower = {m.lower() for m in available_models}

    for env_key, model_name in required.items():
        if model_name.lower() not in available_lower:
            logger.warning(
                "Pre-flight: model '%s' (%s) is NOT pulled in Ollama. "
                "Run: ollama pull %s",
                model_name,
                env_key,
                model_name,
            )
        else:
            logger.info("Pre-flight: ✔ %s = %s (available)", env_key, model_name)
