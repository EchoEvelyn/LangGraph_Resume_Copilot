"""
src/utils/config.py

Central configuration loaded from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    # Retry logic
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

    # Scoring thresholds
    SCORE_THRESHOLD_WARN: float = float(os.getenv("SCORE_THRESHOLD_WARN", "50.0"))
    GROUNDING_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("GROUNDING_CONFIDENCE_THRESHOLD", "0.75")
    )

    # Storage
    DB_PATH: str = os.getenv("DB_PATH", "./data/results.db")

    # Scoring weights (must sum to 1.0)
    # Responsibility removed — redistributed to required skills and nice-to-have
    WEIGHT_REQUIRED_SKILLS: float = 0.50
    WEIGHT_NICE_TO_HAVE: float = 0.20
    WEIGHT_TOOL_OVERLAP: float = 0.20
    WEIGHT_DOMAIN: float = 0.10
    WEIGHT_RESPONSIBILITY: float = 0.00  # unused, kept for compat

    @classmethod
    def scoring_weights(cls) -> dict[str, float]:
        return {
            "required_skills": cls.WEIGHT_REQUIRED_SKILLS,
            "nice_to_have": cls.WEIGHT_NICE_TO_HAVE,
            "tool_overlap": cls.WEIGHT_TOOL_OVERLAP,
            "domain": cls.WEIGHT_DOMAIN,
            "responsibility": cls.WEIGHT_RESPONSIBILITY,
        }