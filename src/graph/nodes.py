"""
src/graph/nodes.py

6-node workflow — 3 LLM calls max, SQLite cache, section-aware extraction.

Node map:
  1. input_validation_node         — pure Python
  2. jd_keyword_extractor_node     — LLM #1 (cached)
  3. resume_keyword_extractor_node — LLM #2 (cached, section-aware)
  4. keyword_matcher_node          — pure Python
  5. match_score_node              — pure Python
  6. bullet_rewriter_node          — LLM #3 (bulk)
"""

from __future__ import annotations
import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from src.graph.state import GraphState
from src.schemas.outputs import (
    JDKeywordAnalysis,
    ResumeKeywordAnalysis,
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
    RewrittenBullet,
    BulkRewriteResult,
)
from src.prompts.prompts import (
    JD_KEYWORD_SYSTEM, JD_KEYWORD_USER,
    RESUME_KEYWORD_SYSTEM, RESUME_KEYWORD_USER,
    EDUCATION_EXTRACT_SYSTEM, EDUCATION_EXTRACT_USER,
    EXPERIENCE_EXTRACT_SYSTEM, EXPERIENCE_EXTRACT_USER,
    PROJECTS_EXTRACT_SYSTEM, PROJECTS_EXTRACT_USER,
    SKILLS_EXTRACT_SYSTEM, SKILLS_EXTRACT_USER,
    BULK_REWRITER_SYSTEM, BULK_REWRITER_USER,
)
from src.scoring.match_score import (
    compute_skill_match,
    skill_match_to_keyword_match,
    calculate_match_score,
)
from src.storage.db import get_cached_jd, cache_jd, get_cached_resume, cache_resume
from src.utils.llm import get_structured_llm, get_llm
from src.utils.section_splitter import split_resume_sections
from src.utils.logging import get_logger, log_node_entry, log_node_exit

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Input Validation
# ─────────────────────────────────────────────────────────────────────────────

def input_validation_node(state: GraphState) -> dict[str, Any]:
    log_node_entry(logger, "input_validation_node", list(state.keys()))
    errors = list(state.get("errors", []))
    if not state.get("raw_resume", "").strip():
        errors.append("Resume is missing or empty.")
    if not state.get("raw_job_description", "").strip():
        errors.append("Job description is missing or empty.")
    result = {"errors": errors, "retry_count": state.get("retry_count", 0)}
    log_node_exit(logger, "input_validation_node", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. JD Keyword Extractor  (LLM #1, cached)
# ─────────────────────────────────────────────────────────────────────────────

def jd_keyword_extractor_node(state: GraphState) -> dict[str, Any]:
    log_node_entry(logger, "jd_keyword_extractor_node", list(state.keys()))
    jd_text = state["raw_job_description"]

    cached = get_cached_jd(jd_text)
    if cached:
        return {
            "jd_keywords": JDKeywordAnalysis(**cached),
            "jd_cache_hit": True,
            "errors": list(state.get("errors", [])),
        }

    llm = get_structured_llm(JDKeywordAnalysis)
    try:
        jd_keywords: JDKeywordAnalysis = llm.invoke([
            SystemMessage(content=JD_KEYWORD_SYSTEM),
            HumanMessage(content=JD_KEYWORD_USER.format(job_description=jd_text)),
        ])
        cache_jd(jd_text, jd_keywords.model_dump())
        result = {"jd_keywords": jd_keywords, "jd_cache_hit": False,
                  "errors": list(state.get("errors", []))}
    except Exception as e:
        result = {
            "errors": list(state.get("errors", [])) + [f"JD extraction failed: {e}"],
            "retry_count": state.get("retry_count", 0) + 1,
        }

    log_node_exit(logger, "jd_keyword_extractor_node", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Resume Keyword Extractor  (LLM #2, cached, section-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_entries(raw: str, model_cls, key: str = "entries") -> list:
    """Parse a JSON array from LLM response into a list of model instances."""
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        entries = data.get(key, data) if isinstance(data, dict) else data
        return [model_cls(**e) for e in entries if isinstance(e, dict)]
    except Exception as ex:
        logger.warning(f"Entry parse failed ({model_cls.__name__}): {ex}")
        return []


def resume_keyword_extractor_node(state: GraphState) -> dict[str, Any]:
    """
    Section-aware resume extraction:
      - Education, Experience, Projects → structured verbatim entries
      - Skills → flat keyword lists for matching
      - All results cached by resume hash
    """
    log_node_entry(logger, "resume_keyword_extractor_node", list(state.keys()))
    resume_text = state["raw_resume"]

    # ── Cache check ───────────────────────────────────────────────────────────
    cached = get_cached_resume(resume_text)
    if cached:
        # Detect legacy cache format (flat dict without section keys).
        # If sections are missing, fall through to re-extract so the UI
        # can display education / experience / project entries properly.
        has_sections = (
            "education"  in cached or
            "experience" in cached or
            "projects"   in cached
        )
        if has_sections:
            resume_keywords = ResumeKeywordAnalysis(**cached.get("keywords", cached))
            education   = [EducationEntry(**e)   for e in cached.get("education", [])]
            experience  = [ExperienceEntry(**e)  for e in cached.get("experience", [])]
            projects    = [ProjectEntry(**e)     for e in cached.get("projects", [])]
            logger.info(
                f"Cache hit — {len(education)} education, "
                f"{len(experience)} experience, {len(projects)} project entries"
            )
            return {
                "resume_keywords":    resume_keywords,
                "education_entries":  education,
                "experience_entries": experience,
                "project_entries":    projects,
                "resume_cache_hit":   True,
                "errors":             list(state.get("errors", [])),
            }
        else:
            logger.info("Legacy cache format detected (no sections); re-extracting")

    # ── Section split ─────────────────────────────────────────────────────────
    sections = split_resume_sections(resume_text)
    llm = get_llm()

    education_entries:  list[EducationEntry]  = []
    experience_entries: list[ExperienceEntry] = []
    project_entries:    list[ProjectEntry]    = []
    merged = ResumeKeywordAnalysis()

    try:
        # Education
        if "education" in sections:
            resp = llm.invoke([
                SystemMessage(content=EDUCATION_EXTRACT_SYSTEM),
                HumanMessage(content=EDUCATION_EXTRACT_USER.format(
                    text=sections["education"]
                )),
            ])
            education_entries = _parse_entries(resp.content, EducationEntry)
            logger.info(f"Education: {len(education_entries)} entries")

        # Experience
        if "experience" in sections:
            resp = llm.invoke([
                SystemMessage(content=EXPERIENCE_EXTRACT_SYSTEM),
                HumanMessage(content=EXPERIENCE_EXTRACT_USER.format(
                    text=sections["experience"]
                )),
            ])
            experience_entries = _parse_entries(resp.content, ExperienceEntry)
            for e in experience_entries:
                merged.bullet_points += e.bullets
                merged.domain_experience += []   # filled via skills
            logger.info(f"Experience: {len(experience_entries)} entries")

        # Projects
        if "projects" in sections:
            resp = llm.invoke([
                SystemMessage(content=PROJECTS_EXTRACT_SYSTEM),
                HumanMessage(content=PROJECTS_EXTRACT_USER.format(
                    text=sections["projects"]
                )),
            ])
            project_entries = _parse_entries(resp.content, ProjectEntry)
            for p in project_entries:
                merged.bullet_points += p.bullets
                merged.project_keywords.append(p.project_name)
            logger.info(f"Projects: {len(project_entries)} entries")

        # Skills
        if "skills" in sections:
            resp = llm.invoke([
                SystemMessage(content=SKILLS_EXTRACT_SYSTEM),
                HumanMessage(content=SKILLS_EXTRACT_USER.format(
                    text=sections["skills"]
                )),
            ])
            try:
                d = json.loads(resp.content.strip())
                merged.technical_skills = d.get("technical_skills", [])
                merged.tools            = d.get("tools", [])
                merged.frameworks       = d.get("frameworks", [])
            except Exception:
                logger.warning("Skills parse failed")

        # Fallback: if no sections detected, extract whole document
        if "full" in sections:
            logger.info("No sections found; falling back to full-document extraction")
            llm_structured = get_structured_llm(ResumeKeywordAnalysis)
            merged = llm_structured.invoke([
                SystemMessage(content=RESUME_KEYWORD_SYSTEM),
                HumanMessage(content=RESUME_KEYWORD_USER.format(resume=resume_text)),
            ])

        # Dedup bullets
        merged.bullet_points = list(dict.fromkeys(merged.bullet_points))
        merged.project_keywords = list(dict.fromkeys(merged.project_keywords))

        # Cache: store keywords + all section entries together
        cache_resume(resume_text, {
            "keywords":   merged.model_dump(),
            "education":  [e.model_dump() for e in education_entries],
            "experience": [e.model_dump() for e in experience_entries],
            "projects":   [e.model_dump() for e in project_entries],
        })

        result = {
            "resume_keywords":    merged,
            "education_entries":  education_entries,
            "experience_entries": experience_entries,
            "project_entries":    project_entries,
            "resume_cache_hit":   False,
            "errors":             list(state.get("errors", [])),
        }

    except Exception as e:
        result = {
            "errors":      list(state.get("errors", [])) + [f"Resume extraction failed: {e}"],
            "retry_count": state.get("retry_count", 0) + 1,
        }

    log_node_exit(logger, "resume_keyword_extractor_node", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Keyword Matcher  (pure Python)
# ─────────────────────────────────────────────────────────────────────────────

def keyword_matcher_node(state: GraphState) -> dict[str, Any]:
    log_node_entry(logger, "keyword_matcher_node", list(state.keys()))

    skill_match = compute_skill_match(
        jd=state["jd_keywords"],
        resume=state["resume_keywords"],
        raw_resume=state.get("raw_resume", ""),
    )
    kw_match = skill_match_to_keyword_match(skill_match)

    result = {
        "skill_match":     skill_match,
        "matched_keywords": kw_match,
        "missing_keywords": skill_match.all_missing,
        "weak_keywords":    skill_match.weak_keywords,
        "errors":           list(state.get("errors", [])),
    }
    log_node_exit(logger, "keyword_matcher_node", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Match Score  (pure Python)
# ─────────────────────────────────────────────────────────────────────────────

def match_score_node(state: GraphState) -> dict[str, Any]:
    log_node_entry(logger, "match_score_node", list(state.keys()))

    score = calculate_match_score(
        jd=state["jd_keywords"],
        resume=state["resume_keywords"],
        skill_match=state["skill_match"],
        raw_resume=state.get("raw_resume", ""),
    )
    result = {"match_score": score, "errors": list(state.get("errors", []))}
    log_node_exit(logger, "match_score_node", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. Bullet Rewriter  (LLM #3 — all bullets in 1 call)
# ─────────────────────────────────────────────────────────────────────────────

def bullet_rewriter_node(state: GraphState) -> dict[str, Any]:
    log_node_entry(logger, "bullet_rewriter_node", list(state.keys()))

    jd = state["jd_keywords"]
    jd_keywords_flat = (
        jd.required_skills + jd.nice_to_have_skills
        + jd.tools + jd.frameworks + jd.domain_keywords
    )[:20]

    bullets = state["resume_keywords"].bullet_points
    if not bullets:
        return {"rewritten_bullets": [], "errors": list(state.get("errors", []))}

    bullets_text = "\n".join(f"{i+1}. {b}" for i, b in enumerate(bullets))
    llm = get_structured_llm(BulkRewriteResult)

    try:
        bulk: BulkRewriteResult = llm.invoke([
            SystemMessage(content=BULK_REWRITER_SYSTEM),
            HumanMessage(content=BULK_REWRITER_USER.format(
                jd_keywords=", ".join(jd_keywords_flat),
                bullets=bullets_text,
            )),
        ])
        rewritten = bulk.rewritten_bullets
        for i in range(len(rewritten), len(bullets)):
            rewritten.append(RewrittenBullet(
                original_bullet=bullets[i], rewritten_bullet=bullets[i],
                source_section="Experience",
                reason_for_rewrite="Not returned; original preserved",
            ))
        logger.info(f"Bulk rewrite: {len(rewritten)} bullets")
        result = {"rewritten_bullets": rewritten, "errors": list(state.get("errors", []))}
    except Exception as e:
        originals = [
            RewrittenBullet(original_bullet=b, rewritten_bullet=b,
                            source_section="Experience",
                            reason_for_rewrite="Rewrite failed")
            for b in bullets
        ]
        result = {
            "rewritten_bullets": originals,
            "errors": list(state.get("errors", [])) + [f"Rewrite failed: {e}"],
        }

    log_node_exit(logger, "bullet_rewriter_node", result)
    return result