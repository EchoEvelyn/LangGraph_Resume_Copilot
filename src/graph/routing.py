"""
src/graph/routing.py — simplified routing, no grounding/revision edges.
"""

from __future__ import annotations
from src.graph.state import GraphState
from src.utils.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def route_after_input_validation(state: GraphState) -> str:
    if state.get("errors"):
        return "error_end"
    return "jd_keyword_extractor_node"


def route_after_jd_extraction(state: GraphState) -> str:
    if state.get("jd_keywords") is None:
        if state.get("retry_count", 0) < Config.MAX_RETRIES:
            return "jd_keyword_extractor_node"
        return "error_end"
    return "resume_keyword_extractor_node"


def route_after_resume_extraction(state: GraphState) -> str:
    if state.get("resume_keywords") is None:
        if state.get("retry_count", 0) < Config.MAX_RETRIES:
            return "resume_keyword_extractor_node"
        return "error_end"
    return "keyword_matcher_node"