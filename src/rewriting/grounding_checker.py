"""
src/rewriting/grounding_checker.py

Pure Python helpers for grounding check results.
The actual LLM grounding call now lives in grounding_checker_node (nodes.py)
as a single bulk call — replacing the old per-bullet approach.
"""

from __future__ import annotations
from src.schemas.outputs import GroundingCheckResult
from src.utils.config import Config

CONFIDENCE_THRESHOLD = Config.GROUNDING_CONFIDENCE_THRESHOLD


def any_bullets_need_revision(results: list[GroundingCheckResult]) -> bool:
    """Return True if any bullet fails the confidence threshold."""
    return any(
        not r.supported or r.confidence_score < CONFIDENCE_THRESHOLD
        for r in results
    )


def grounding_summary(results: list[GroundingCheckResult]) -> str:
    """Plain-English summary of grounding results."""
    total = len(results)
    passed = sum(
        1 for r in results
        if r.supported and r.confidence_score >= CONFIDENCE_THRESHOLD
    )
    summary = f"{passed}/{total} bullets passed grounding check (threshold={CONFIDENCE_THRESHOLD})."
    issues = [
        f"Bullet {i+1}: confidence={r.confidence_score:.2f}, unsupported={r.unsupported_claims}"
        for i, r in enumerate(results)
        if not r.supported or r.confidence_score < CONFIDENCE_THRESHOLD
    ]
    if issues:
        summary += " Issues: " + "; ".join(issues)
    return summary