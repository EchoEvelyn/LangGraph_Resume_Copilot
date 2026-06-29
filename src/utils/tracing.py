"""
src/utils/tracing.py

LangSmith tracing setup.
Call init_tracing() once at application startup (app.py and workflow.py).

LangSmith traces every LLM call, node execution, and graph invocation
automatically once LANGCHAIN_TRACING_V2=true is set in the environment.

No code changes needed in nodes — LangChain instruments everything via
the callback system when the env vars are present.
"""

from __future__ import annotations

import os
from src.utils.logging import get_logger

logger = get_logger(__name__)


def init_tracing() -> bool:
    """
    Activate LangSmith tracing if env vars are configured.
    Returns True if tracing is enabled, False otherwise.

    Required env vars:
      LANGCHAIN_TRACING_V2=true
      LANGCHAIN_API_KEY=ls__...
      LANGCHAIN_PROJECT=langgraph-resume-copilot  (optional, defaults to 'default')
    """
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    api_key = os.getenv("LANGCHAIN_API_KEY", "")

    if not tracing_enabled:
        logger.info("LangSmith tracing disabled (LANGCHAIN_TRACING_V2 not set to true)")
        return False

    if not api_key:
        logger.warning("LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is missing — tracing skipped")
        return False

    project = os.getenv("LANGCHAIN_PROJECT", "langgraph-resume-copilot")
    logger.info(f"LangSmith tracing enabled → project: '{project}'")
    return True


def get_run_metadata(resume_length: int, jd_length: int) -> dict:
    """
    Return metadata dict to attach to a graph run.
    Passed as config to graph.invoke() so each trace is annotated.

    Usage:
        graph.invoke(state, config={"metadata": get_run_metadata(...)})
    """
    return {
        "resume_chars": resume_length,
        "jd_chars": jd_length,
        "app_env": os.getenv("APP_ENV", "development"),
    }