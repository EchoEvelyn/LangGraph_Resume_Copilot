"""
src/graph/state.py
"""

from __future__ import annotations
from typing import Optional
from typing_extensions import TypedDict

from src.schemas.outputs import (
    JDKeywordAnalysis,
    ResumeKeywordAnalysis,
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
    SkillMatchResult,
    KeywordMatchResult,
    MatchScore,
    RewrittenBullet,
)


class GraphState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────
    raw_resume: str
    raw_job_description: str

    # ── Cache flags ───────────────────────────────────────────────
    resume_cache_hit: bool
    jd_cache_hit: bool

    # ── Structured resume sections (verbatim) ─────────────────────
    education_entries: list[EducationEntry]
    experience_entries: list[ExperienceEntry]
    project_entries: list[ProjectEntry]

    # ── Flat keyword extraction (for matching/scoring) ────────────
    jd_keywords: Optional[JDKeywordAnalysis]
    resume_keywords: Optional[ResumeKeywordAnalysis]

    # ── Matching & scoring ────────────────────────────────────────
    skill_match: Optional[SkillMatchResult]       # detailed required/nice split
    matched_keywords: Optional[KeywordMatchResult] # for scorer compat
    missing_keywords: list[str]
    weak_keywords: list[str]
    match_score: Optional[MatchScore]

    # ── Bullet rewriting ─────────────────────────────────────────
    rewritten_bullets: list[RewrittenBullet]

    # ── Control ───────────────────────────────────────────────────
    errors: list[str]
    retry_count: int