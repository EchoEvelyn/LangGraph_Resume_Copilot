"""
src/utils/logging.py

Structured logging for the LangGraph workflow.
"""

import logging
import os
import sys
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for a given module."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, log_level, logging.INFO))
    return logger


def log_node_entry(logger: logging.Logger, node_name: str, state_keys: list[str]) -> None:
    """Log when a node begins execution."""
    logger.info(f"[{node_name}] ▶ entering | state keys available: {state_keys}")


def log_node_exit(logger: logging.Logger, node_name: str, result: dict[str, Any]) -> None:
    """Log when a node completes execution."""
    keys = list(result.keys())
    logger.info(f"[{node_name}] ✓ completed | wrote: {keys}")
