"""
src/rewriting/bullet_rewriter.py

Handles bullet point selection and LLM-powered rewriting.
Each rewrite is grounded in the original bullet — no fabrication.
"""

from __future__ import annotations

import json
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.messages import SystemMessage, HumanMessage

from src.schemas.outputs import RewrittenBullet
from src.prompts.prompts import (
    BULLET_REWRITER_SYSTEM,
    BULLET_REWRITER_USER,
    BULLET_SELECTOR_SYSTEM,
    BULLET_SELECTOR_USER,
)
from src.utils.llm import get_llm, get_structured_llm
from src.utils.logging import get_logger

logger = get_logger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
def select_bullets_to_rewrite(
    missing_keywords: list[str],
    weak_keywords: list[str],
    recommended_keywords: list[str],
    all_bullets: list[str],
    max_bullets: int = 8,
) -> list[str]:
    """
    Ask the LLM to choose which bullets are most worth rewriting.
    Returns a list of selected bullet strings.
    Falls back to returning all bullets if parsing fails.
    """
    llm = get_llm()

    messages = [
        SystemMessage(content=BULLET_SELECTOR_SYSTEM),
        HumanMessage(
            content=BULLET_SELECTOR_USER.format(
                missing_keywords="\n".join(f"- {k}" for k in missing_keywords),
                weak_keywords="\n".join(f"- {k}" for k in weak_keywords),
                recommended_keywords="\n".join(f"- {k}" for k in recommended_keywords),
                bullet_points="\n".join(f"• {b}" for b in all_bullets),
            )
        ),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    try:
        selected: list[str] = json.loads(content)
        # Validate that returned bullets actually exist in the original list
        valid = [b for b in selected if b in all_bullets]
        if not valid:
            logger.warning("Bullet selector returned bullets not in original list; using all")
            return all_bullets[:max_bullets]
        logger.info(f"Selected {len(valid)} bullets for rewriting")
        return valid[:max_bullets]
    except json.JSONDecodeError:
        logger.warning("Could not parse bullet selector JSON; defaulting to all bullets")
        return all_bullets[:max_bullets]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
def rewrite_single_bullet(
    original_bullet: str,
    jd_keywords_to_use: list[str],
    source_section: str,
) -> RewrittenBullet:
    """
    Rewrite a single bullet using structured output.
    Applies the strict no-fabrication rules from the system prompt.
    """
    llm = get_structured_llm(RewrittenBullet)

    messages = [
        SystemMessage(content=BULLET_REWRITER_SYSTEM),
        HumanMessage(
            content=BULLET_REWRITER_USER.format(
                jd_keywords_to_use=", ".join(jd_keywords_to_use),
                original_bullet=original_bullet,
                source_section=source_section,
            )
        ),
    ]

    result: RewrittenBullet = llm.invoke(messages)

    # Always preserve the original bullet in the schema
    result.original_bullet = original_bullet
    result.source_section = source_section

    logger.info(f"Rewrote bullet: {original_bullet[:60]}…")
    return result


def rewrite_all_bullets(
    selected_bullets: list[str],
    jd_keywords: list[str],
    resume_keywords_blob: str,
    source_section: str = "Experience",
) -> list[RewrittenBullet]:
    """
    Rewrite every selected bullet.
    Limits jd_keywords passed to each call to the top-10 most relevant
    to avoid prompt bloat.
    """
    rewrites: list[RewrittenBullet] = []
    top_keywords = jd_keywords[:10]

    for bullet in selected_bullets:
        try:
            rewritten = rewrite_single_bullet(
                original_bullet=bullet,
                jd_keywords_to_use=top_keywords,
                source_section=source_section,
            )
            rewrites.append(rewritten)
        except Exception as e:
            logger.error(f"Failed to rewrite bullet '{bullet[:40]}…': {e}")
            # Keep original as a fallback rewrite
            rewrites.append(
                RewrittenBullet(
                    original_bullet=bullet,
                    rewritten_bullet=bullet,
                    jd_keywords_used=[],
                    source_section=source_section,
                    reason_for_rewrite="Rewrite failed; original preserved",
                )
            )

    return rewrites
