"""
src/schemas/outputs.py
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# JD Extraction
# ─────────────────────────────────────────────────────────────────────────────

class JDKeywordAnalysis(BaseModel):
    role_title: str = Field(description="Exact job title")
    required_skills: list[str] = Field(default_factory=list,
        description="Core technical skills explicitly required. Concise, 1-4 words each.")
    nice_to_have_skills: list[str] = Field(default_factory=list,
        description="Preferred/bonus skills. Concise, 1-4 words each.")
    tools: list[str] = Field(default_factory=list,
        description="Specific tools, libraries, platforms (e.g. PyTorch, Kafka, FAISS)")
    # Internal fields for scoring — not shown in UI
    frameworks: list[str] = Field(default_factory=list)
    domain_keywords: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    action_verbs: list[str] = Field(default_factory=list)
    seniority_signals: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Resume Extraction
# ─────────────────────────────────────────────────────────────────────────────

class EducationEntry(BaseModel):
    raw_text: str
    institution: str = Field(default="")
    degree: str = Field(default="")
    period: str = Field(default="")


class ExperienceEntry(BaseModel):
    raw_text: str
    company: str = Field(default="")
    title: str = Field(default="")
    period: str = Field(default="")
    bullets: list[str] = Field(default_factory=list)


class ProjectEntry(BaseModel):
    raw_text: str
    project_name: str = Field(default="")
    bullets: list[str] = Field(default_factory=list)


class ResumeKeywordAnalysis(BaseModel):
    technical_skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    domain_experience: list[str] = Field(default_factory=list)
    project_keywords: list[str] = Field(default_factory=list)
    action_verbs: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    bullet_points: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Match Results — 4-level system
# ─────────────────────────────────────────────────────────────────────────────

MatchLevel = Literal["strong_match", "partial_match", "related_experience", "missing"]

MATCH_LEVEL_SCORES: dict[str, float] = {
    "strong_match":       1.0,
    "partial_match":      0.6,
    "related_experience": 0.4,
    "missing":            0.0,
}

MATCH_LEVEL_LABELS: dict[str, str] = {
    "strong_match":       "✅ Strong Match",
    "partial_match":      "🟡 Partial Match",
    "related_experience": "🔵 Related Experience",
    "missing":            "❌ Missing",
}


class KeywordMatchDetail(BaseModel):
    """Detailed match result for a single JD keyword."""
    jd_keyword: str
    match_level: MatchLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    matched_resume_terms: list[str] = Field(default_factory=list,
        description="Specific terms from resume that matched")
    resume_evidence: str = Field(default="",
        description="The resume sentence/phrase supporting this match")
    explanation: str = Field(default="",
        description="Human-readable explanation of why this match level was assigned")


class SkillMatchResult(BaseModel):
    """Full match results across all JD keyword categories."""
    # Per-category detailed matches
    required_matches: list[KeywordMatchDetail] = Field(default_factory=list)
    nice_to_have_matches: list[KeywordMatchDetail] = Field(default_factory=list)
    tool_matches: list[KeywordMatchDetail] = Field(default_factory=list)

    # Convenience flat lists for UI and scoring
    required_matched: list[str] = Field(default_factory=list)
    required_missing: list[str] = Field(default_factory=list)
    required_coverage: float = Field(default=0.0, ge=0, le=100)
    nice_matched: list[str] = Field(default_factory=list)
    nice_missing: list[str] = Field(default_factory=list)
    nice_coverage: float = Field(default=0.0, ge=0, le=100)
    tool_matched: list[str] = Field(default_factory=list)
    tool_missing: list[str] = Field(default_factory=list)

    # Legacy compat
    all_missing: list[str] = Field(default_factory=list)
    weak_keywords: list[str] = Field(default_factory=list)
    recommended_keywords_to_emphasize: list[str] = Field(default_factory=list)
    evidence: dict[str, dict] = Field(default_factory=dict)


class KeywordMatchResult(BaseModel):
    """Kept for scoring compatibility."""
    exact_matches: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    weak_keywords: list[str] = Field(default_factory=list)
    recommended_keywords_to_emphasize: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

class MatchScore(BaseModel):
    overall_score: float = Field(ge=0, le=100)
    required_skill_score: float = Field(ge=0, le=100)
    nice_to_have_score: float = Field(ge=0, le=100)
    tool_overlap_score: float = Field(ge=0, le=100)
    domain_score: float = Field(ge=0, le=100)
    responsibility_alignment_score: float = Field(ge=0, le=100)
    explanation: str


# ─────────────────────────────────────────────────────────────────────────────
# Bullet Rewriting
# ─────────────────────────────────────────────────────────────────────────────

class RewrittenBullet(BaseModel):
    original_bullet: str
    rewritten_bullet: str
    jd_keywords_used: list[str] = Field(default_factory=list)
    source_section: str = Field(default="Experience")
    reason_for_rewrite: str = Field(default="")


class BulkRewriteResult(BaseModel):
    rewritten_bullets: list[RewrittenBullet] = Field(default_factory=list)
    keywords_incorporated: list[str] = Field(default_factory=list)