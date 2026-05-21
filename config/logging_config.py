"""
config/logging_config.py — Rotating file + stderr logging for the RAG pipeline.

Configures:
    - A RotatingFileHandler  (10 MB max, 5 backups, detailed format)
    - A StreamHandler        (stderr, short format)
    - INFO level on both "lightrag" and "raganything" loggers

All paths and rotation settings are read from environment variables.
"""

from __future__ import annotations

import os
import logging
import logging.config
import logging.handlers


def configure_logging(log_filename: str = "rag_pipeline.log") -> None:
    """Configure logging for the multimodal Graph RAG pipeline.

    Must be called before any other module is imported so that the
    "lightrag" and "raganything" loggers pick up the correct handlers.

    Environment variables consumed:
        LOG_DIR          — directory for the rotating log file (default: cwd)
        LOG_MAX_BYTES    — max file size before rotation (default: 10 485 760 = 10 MB)
        LOG_BACKUP_COUNT — number of backup files to keep (default: 5)
        VERBOSE_DEBUG    — "true" to enable LightRAG verbose debug (default: false)

    Args:
        log_filename: Base name of the log file. Default: "rag_pipeline.log"
    """
    # Reset any pre-existing handlers to ensure clean state
    for logger_name in [
        "uvicorn", "uvicorn.access", "uvicorn.error",
        "lightrag", "raganything",
    ]:
        inst = logging.getLogger(logger_name)
        inst.handlers = []
        inst.filters = []

    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, log_filename))
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    print(f"\nLog file → {log_file_path}\n")

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "short": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "short",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "detailed",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "raganything": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Sync LightRAG's internal logger object with the new level
    try:
        from lightrag.utils import logger as lightrag_logger, set_verbose_debug
        lightrag_logger.setLevel(logging.INFO)
        set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")
    except ImportError:
        pass  # lightrag not yet installed — fine during scaffolding
