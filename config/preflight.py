"""
config/preflight.py — Pre-flight checks for the RAG pipeline.

Verifies external service dependencies (vLLM) are reachable BEFORE
the pipeline begins initialising heavy objects.  Failing fast here
produces a clear, actionable error instead of a deep traceback from
inside the embedding or LLM stack.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

logger = logging.getLogger("lightrag")


class VllmNotReachableError(RuntimeError):
    """Raised when the vLLM service cannot be contacted."""


async def check_vllm_connectivity(
    host: str | None = None,
    timeout: float = 10.0,
) -> dict:
    """Verify that vLLM is running and list locally available models.

    Hits the ``/models`` endpoint (OpenAI-compatible, lightweight) to confirm
    the service is up and returns the model catalogue.

    Args:
        host:    vLLM base URL (with /v1 suffix).  Falls back to
                 LLM_BINDING_HOST env var, then ``http://127.0.0.1:8000/v1``.
        timeout: Seconds to wait before declaring unreachable.

    Returns:
        A dict with keys ``host`` (str) and ``models`` (list[str]).

    Raises:
        VllmNotReachableError: If the service is not reachable.
    """
    import httpx  # lightweight; already a transitive dep of openai

    if host is None:
        host = os.getenv(
            "LLM_BINDING_HOST",
            "http://127.0.0.1:8000/v1",
        )

    # Normalise: strip trailing slash so /models joins cleanly
    host = host.rstrip("/")
    url = f"{host}/models"

    logger.info("Pre-flight: checking vLLM connectivity at %s …", host)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.ConnectError as exc:
        raise VllmNotReachableError(
            f"\n{'='*60}\n"
            f"  ❌  vLLM is NOT reachable at {host}\n"
            f"{'='*60}\n"
            f"  Possible fixes:\n"
            f"    1. Start vLLM:  vllm serve <model-name>\n"
            f"    2. Verify the host/port in .env:\n"
            f"         LLM_BINDING_HOST={host}\n"
            f"    3. If vLLM runs on a different machine, update the\n"
            f"       host to its reachable IP/hostname.\n"
            f"    4. Install vLLM: pip install vllm\n"
            f"{'='*60}\n"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise VllmNotReachableError(
            f"vLLM at {host} returned HTTP {exc.response.status_code}. "
            f"Is a reverse proxy interfering?"
        ) from exc
    except Exception as exc:
        raise VllmNotReachableError(
            f"Unexpected error reaching vLLM at {host}: {exc}"
        ) from exc

    # Parse model names from the /models response (OpenAI format)
    data = resp.json()
    model_names = [m.get("id", "?") for m in data.get("data", [])]
    logger.info(
        "Pre-flight: vLLM is UP — %d model(s) available: %s",
        len(model_names),
        ", ".join(model_names) if model_names else "(none loaded yet)",
    )
    return {"host": host, "models": model_names}


async def check_all_vllm_endpoints(
    timeout: float = 10.0,
) -> dict:
    """Check connectivity to all configured vLLM endpoints.

    Checks LLM_BINDING_HOST and EMBEDDING_BINDING_HOST separately since
    vLLM runs one model per process (different ports).

    Returns:
        A dict with keys ``host`` (str), ``models`` (list[str]),
        and ``endpoints_checked`` (list[str]).
    """
    import httpx

    all_models: list[str] = []
    primary_host = os.getenv("LLM_BINDING_HOST", "http://127.0.0.1:8000/v1")
    endpoints_checked: list[str] = []

    hosts_to_check = set()
    hosts_to_check.add(os.getenv("LLM_BINDING_HOST", "http://127.0.0.1:8000/v1"))
    hosts_to_check.add(os.getenv("EMBEDDING_BINDING_HOST", "http://127.0.0.1:8001/v1"))

    for host in hosts_to_check:
        host = host.rstrip("/")
        url = f"{host}/models"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            data = resp.json()
            models = [m.get("id", "?") for m in data.get("data", [])]
            all_models.extend(models)
            endpoints_checked.append(host)
            logger.info("Pre-flight: vLLM at %s — models: %s", host, ", ".join(models))
        except Exception as exc:
            logger.warning("Pre-flight: vLLM at %s is NOT reachable: %s", host, exc)

    return {
        "host": primary_host,
        "models": all_models,
        "endpoints_checked": endpoints_checked,
    }


async def verify_required_models(
    available_models: list[str],
) -> None:
    """Warn (not crash) if required models are not served by vLLM.

    Reads model names from .env (LLM_MODEL, EMBEDDING_MODEL, VISION_MODEL)
    and checks them against the list returned by /models.

    Args:
        available_models: Model names returned by check_vllm_connectivity().
    """
    required = {
        "LLM_MODEL": os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-14B-Instruct"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
        "VISION_MODEL": os.getenv("VISION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
    }

    # vLLM model IDs are case-sensitive but let's normalise for comparison
    available_lower = {m.lower() for m in available_models}

    for env_key, model_name in required.items():
        if model_name.lower() not in available_lower:
            logger.warning(
                "Pre-flight: model '%s' (%s) is NOT loaded in any vLLM instance. "
                "Start vLLM with: vllm serve %s",
                model_name,
                env_key,
                model_name,
            )
        else:
            logger.info("Pre-flight: ✔ %s = %s (available)", env_key, model_name)
