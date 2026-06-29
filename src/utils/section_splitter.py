"""
src/utils/section_splitter.py

Rule-based resume section splitter.
Splits resume text into named sections before LLM extraction,
so each LLM call gets a shorter, more focused input.

Supported section headers (case-insensitive, flexible spacing):
  Experience, Work Experience, Professional Experience
  Projects, Side Projects, Personal Projects
  Skills, Technical Skills, Core Skills
  Education
  Summary, Profile, About
  Certifications, Awards, Publications
"""

from __future__ import annotations
import re
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Ordered list of canonical section names and their regex aliases
SECTION_PATTERNS: list[tuple[str, str]] = [
    ("summary",          r"summary|profile|about|objective"),
    ("experience",       r"experience|work experience|professional experience|employment"),
    ("projects",         r"projects?|side projects?|personal projects?|open.?source"),
    ("skills",           r"skills?|technical skills?|core skills?|competencies|technologies"),
    ("education",        r"education|academic|degree"),
    ("certifications",   r"certifications?|awards?|publications?|achievements?"),
]

# A section header is a short line (≤60 chars) that matches one of the patterns
# and is followed by content. We allow optional punctuation / underscores / dashes.
_HEADER_RE = re.compile(
    r"^[ \t]*("
    + "|".join(alias for _, alias in SECTION_PATTERNS)
    + r")[:\-–—_\s]*$",
    re.IGNORECASE | re.MULTILINE,
)


def split_resume_sections(text: str) -> dict[str, str]:
    """
    Split resume text into a dict of {section_name: section_text}.

    Sections not matching any known header are grouped under "other".
    If no headers are detected, returns {"full": text} so the caller
    can fall back to whole-document extraction.

    Example output:
        {
            "summary":    "Experienced ML engineer...",
            "experience": "Acme AI (2023–present)\\n• Built RAG pipeline...",
            "projects":   "LangGraph Resume Copilot\\n• Built a...",
            "skills":     "Python, LangChain, FAISS...",
            "education":  "B.Sc. Computer Science, UC Berkeley 2022",
        }
    """
    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    current_section = "other"
    current_lines: list[str] = []

    for line in lines:
        match = _HEADER_RE.match(line.strip())
        if match and len(line.strip()) <= 60:
            # Save previous section
            if current_lines:
                sections.setdefault(current_section, []).extend(current_lines)
            current_section = _canonical_name(line.strip())
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    if current_lines:
        sections.setdefault(current_section, []).extend(current_lines)

    # Convert lists to strings, strip blank lines
    result = {
        k: "\n".join(v).strip()
        for k, v in sections.items()
        if "\n".join(v).strip()
    }

    if not result or list(result.keys()) == ["other"]:
        logger.warning("No section headers detected; using full-document extraction")
        return {"full": text}

    logger.info(f"Resume sections detected: {list(result.keys())}")
    return result


def _canonical_name(header_line: str) -> str:
    """Map a raw header line to its canonical section name."""
    clean = header_line.lower().strip(" :-–—_")
    for canonical, alias_pattern in SECTION_PATTERNS:
        if re.fullmatch(alias_pattern, clean, re.IGNORECASE):
            return canonical
    return "other"


def get_bullets_section(sections: dict[str, str]) -> str:
    """
    Return the text most likely to contain bullet points
    (experience + projects combined).
    Used by bullet_rewriter_node to focus the rewrite on relevant content.
    """
    parts = []
    for key in ("experience", "projects", "other"):
        if key in sections:
            parts.append(sections[key])
    return "\n\n".join(parts)