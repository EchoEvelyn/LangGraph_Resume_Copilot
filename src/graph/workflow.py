"""
src/graph/workflow.py

Lean 6-node LangGraph workflow — 3 LLM calls max, SQLite cache.

Graph topology:
  START
    → input_validation_node          (pure Python)
        ─ [error] → END
        ─ [ok]    → jd_keyword_extractor_node    (LLM #1, cached)
                      → resume_keyword_extractor_node  (LLM #2, cached)
                          → keyword_matcher_node       (pure Python)
                              → match_score_node       (pure Python)
                                  → bullet_rewriter_node  (LLM #3, 1 bulk call)
                                      → END

Cache behaviour:
  - Same resume text → resume_keyword_extractor skips LLM, loads from SQLite
  - Same JD text     → jd_keyword_extractor skips LLM, loads from SQLite
  - Second run on same resume + new JD → only 2 LLM calls
  - Same resume + same JD             → only 1 LLM call (rewriter)
"""

from __future__ import annotations
from langgraph.graph import StateGraph, END

from src.graph.state import GraphState
from src.graph.nodes import (
    input_validation_node,
    jd_keyword_extractor_node,
    resume_keyword_extractor_node,
    keyword_matcher_node,
    match_score_node,
    bullet_rewriter_node,
)
from src.graph.routing import (
    route_after_input_validation,
    route_after_jd_extraction,
    route_after_resume_extraction,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    """Build and compile the lean resume copilot workflow."""
    graph = StateGraph(GraphState)

    graph.add_node("input_validation_node",          input_validation_node)
    graph.add_node("jd_keyword_extractor_node",      jd_keyword_extractor_node)
    graph.add_node("resume_keyword_extractor_node",  resume_keyword_extractor_node)
    graph.add_node("keyword_matcher_node",           keyword_matcher_node)
    graph.add_node("match_score_node",               match_score_node)
    graph.add_node("bullet_rewriter_node",           bullet_rewriter_node)

    graph.set_entry_point("input_validation_node")

    graph.add_conditional_edges(
        "input_validation_node",
        route_after_input_validation,
        {"error_end": END, "jd_keyword_extractor_node": "jd_keyword_extractor_node"},
    )
    graph.add_conditional_edges(
        "jd_keyword_extractor_node",
        route_after_jd_extraction,
        {
            "jd_keyword_extractor_node":     "jd_keyword_extractor_node",
            "resume_keyword_extractor_node": "resume_keyword_extractor_node",
            "error_end": END,
        },
    )
    graph.add_conditional_edges(
        "resume_keyword_extractor_node",
        route_after_resume_extraction,
        {
            "resume_keyword_extractor_node": "resume_keyword_extractor_node",
            "keyword_matcher_node":          "keyword_matcher_node",
            "error_end": END,
        },
    )

    graph.add_edge("keyword_matcher_node", "match_score_node")
    graph.add_edge("match_score_node",     "bullet_rewriter_node")
    graph.add_edge("bullet_rewriter_node", END)

    compiled = graph.compile()
    logger.info("Lean workflow compiled: 6 nodes, 3 LLM calls max")
    return compiled


def run_workflow(resume: str, job_description: str, run_name: str | None = None) -> GraphState:
    """Build and run the graph synchronously."""
    from src.utils.tracing import get_run_metadata

    graph = build_graph()
    initial_state: GraphState = {
        "raw_resume":          resume,
        "raw_job_description": job_description,
        "errors":              [],
        "retry_count":         0,
        "missing_keywords":    [],
        "weak_keywords":       [],
        "rewritten_bullets":   [],
        "resume_cache_hit":    False,
        "jd_cache_hit":        False,
    }
    config = {
        "run_name": run_name or "resume-copilot-run",
        "metadata": get_run_metadata(
            resume_length=len(resume),
            jd_length=len(job_description),
        ),
    }
    return graph.invoke(initial_state, config=config)