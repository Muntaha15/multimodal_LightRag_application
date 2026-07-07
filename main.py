"""
main.py — Entry point for the local multimodal Graph RAG pipeline.

Pipeline overview
-----------------
1. Configure logging (before any imports that might log).
2. Load .env.
3. Optionally purge stale index files (--fresh flag).
4. Initialise LightRAG (inner layer) → wrap with RAGAnything (outer layer).
5. Embedding smoke-test (prints detected dimension).
6. Ingest documents in configured directory via file_ingest (multimodal).
7. Run all 4 query modes on the demo question.
8. finally: flush LLM cache + finalize_storages.

Usage
-----
    python main.py               # normal run
    python main.py --fresh       # delete stale index files before starting
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ── Logging MUST be configured before any LightRAG import ─────────────────
from config.logging_config import configure_logging
configure_logging()

# ── Load env AFTER logging so the log path can be overridden via .env ─────
load_dotenv(dotenv_path=".env", override=True)

from config.preflight import check_ollama_connectivity, verify_required_models
from rag.lightrag_init import initialize_lightrag
from rag.raganything_init import initialize_raganything
from ingest.file_ingest import insert_files, insert_folder
from query.query_runner import run_all_modes

logger = logging.getLogger("lightrag")

# ── Stale index files that the --fresh flag removes ───────────────────────
_STALE_FILES = [
    "graph_chunk_entity_relation.graphml",
    "kv_store_doc_status.json",
    "kv_store_full_docs.json",
    "kv_store_text_chunks.json",
    "vdb_chunks.json",
    "vdb_entities.json",
    "vdb_relationships.json",
]

DEMO_QUERY = "What are the top themes in this story?"


def _purge_stale_files(working_dir: str) -> None:
    """Delete stale index files from the LightRAG working directory."""
    for filename in _STALE_FILES:
        file_path = Path(working_dir) / filename
        if file_path.exists():
            file_path.unlink()
            logger.info("Deleted stale index file: %s", file_path)


async def _smoke_test_embedding(rag) -> bool:
    """Run a single embedding call and print the detected vector dimension.

    Returns True on success, False on failure. Never raises — the pipeline
    can continue (e.g. for query-only runs on an existing index).
    """
    test_texts = ["This is a smoke-test string for the embedding function."]
    try:
        embedding = await rag.embedding_func(test_texts)
    except Exception as exc:
        logger.error("Embedding smoke test FAILED: %s", exc)
        print("\n" + "=" * 45)
        print(" ❌ Embedding smoke test FAILED")
        print("=" * 45)
        print(f"  Error: {exc}")
        print("  → Is Ollama running?  ollama serve")
        print("  → Is the model pulled? ollama pull " + os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"))
        print("=" * 45 + "\n")
        return False

    dim = embedding.shape[1]
    print("\n" + "=" * 45)
    print(" ✔ Embedding smoke test")
    print("=" * 45)
    print(f"  Input : {test_texts[0]!r}")
    print(f"  Shape : {embedding.shape}  (dim={dim})")
    print("=" * 45 + "\n")
    return True


async def main(fresh: bool = False) -> None:
    """Main async pipeline."""
    working_dir = os.getenv("WORKING_DIR", "./storage/dickens_v1")
    lightrag = None   # inner LightRAG instance
    rag = None        # outer RAGAnything wrapper

    try:
        # ── Optional purge ─────────────────────────────────────────────────
        if fresh:
            logger.info("--fresh flag set: purging stale index files.")
            _purge_stale_files(working_dir)

        # ── Pre-flight: verify Ollama is reachable ─────────────────────────
        preflight = await check_ollama_connectivity()
        await verify_required_models(preflight["models"])

        # ── Init LightRAG + RAGAnything ────────────────────────────────────
        lightrag = await initialize_lightrag()
        rag = initialize_raganything(lightrag)

        # ── Embedding smoke test ───────────────────────────────────────────
        embed_ok = await _smoke_test_embedding(lightrag)
        if not embed_ok:
            logger.warning(
                "Embedding smoke test failed — continuing anyway "
                "(queries on an existing index may still work)."
            )

        # ── Ingest source documents folder ─────────────────────────────────
        docs_dir_path = os.getenv("DOCS_DIR", "./docs")
        docs_dir = Path(docs_dir_path)
        if docs_dir.is_dir():
            logger.info("Ingesting folder via RAGAnything wrapper: %s", docs_dir)
            await insert_folder(rag, docs_dir)
        else:
            logger.warning(
                "Documents folder not found at '%s' — skipping ingest. "
                "Set DOCS_DIR in .env to override.",
                docs_dir_path,
            )

        # ── Run all 4 query modes ──────────────────────────────────────────
        await run_all_modes(rag, DEMO_QUERY)

    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        raise

    finally:
        # llm_response_cache lives on the inner LightRAG instance.
        # RAGAnything.finalize_storages() handles both layers.
        if lightrag is not None and hasattr(lightrag, "llm_response_cache"):
            await lightrag.llm_response_cache.index_done_callback()
        if rag is not None:
            await rag.finalize_storages()
            logger.info("Storages finalised. Goodbye.")
        elif lightrag is not None:
            await lightrag.finalize_storages()
            logger.info("Storages finalised. Goodbye.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local multimodal Graph RAG — LightRAG + RAGAnything + Ollama"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete stale LightRAG index files before starting.",
    )
    args = parser.parse_args()

    asyncio.run(main(fresh=args.fresh))
    print("\nDone!")
