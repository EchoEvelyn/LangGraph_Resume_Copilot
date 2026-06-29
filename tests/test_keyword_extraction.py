"""
tests/test_keyword_extraction.py

Unit tests for JD and resume keyword extraction logic.
Uses mock LLM responses so no API key is needed.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.schemas.outputs import JDKeywordAnalysis, ResumeKeywordAnalysis


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_jd_keywords():
    return JDKeywordAnalysis(
        role_title="AI Engineer",
        required_skills=["Python", "LangChain", "RAG", "LLM evaluation"],
        nice_to_have_skills=["LangGraph", "Kubernetes", "open-source"],
        tools=["FAISS", "MLflow", "Docker"],
        frameworks=["LangChain", "LangGraph", "FastAPI"],
        domain_keywords=["RAG", "NLP", "prompt engineering", "fine-tuning"],
        responsibilities=["build LLM pipelines", "deploy models", "evaluate RAG systems"],
        action_verbs=["design", "build", "deploy", "own", "collaborate"],
        seniority_signals=["3+ years", "production", "own", "cross-functional"],
    )


@pytest.fixture
def sample_resume_keywords():
    return ResumeKeywordAnalysis(
        technical_skills=["Python", "SQL", "Bash"],
        tools=["FAISS", "Docker", "MLflow", "Weights & Biases"],
        frameworks=["LangChain", "LangGraph", "FastAPI", "Streamlit"],
        domain_experience=["NLP", "RAG", "LLM evaluation", "prompt engineering"],
        project_keywords=["RAG pipeline", "evaluation framework", "Streamlit interface"],
        action_verbs=["built", "implemented", "designed", "fine-tuned", "containerised"],
        metrics=["28% ticket volume reduction", "18% retrieval precision improvement"],
        bullet_points=[
            "Built a RAG pipeline using LangChain and FAISS to answer support queries, reducing ticket volume by 28%.",
            "Implemented an LLM evaluation framework using custom metrics tracked with MLflow.",
        ],
    )


# ─── Schema validation tests ──────────────────────────────────────────────────

class TestJDKeywordAnalysisSchema:
    def test_creates_valid_instance(self, sample_jd_keywords):
        assert sample_jd_keywords.role_title == "AI Engineer"
        assert "Python" in sample_jd_keywords.required_skills
        assert "LangGraph" in sample_jd_keywords.nice_to_have_skills

    def test_defaults_to_empty_lists(self):
        minimal = JDKeywordAnalysis(role_title="Test Role")
        assert minimal.required_skills == []
        assert minimal.tools == []
        assert minimal.domain_keywords == []

    def test_serialises_to_dict(self, sample_jd_keywords):
        d = sample_jd_keywords.model_dump()
        assert "role_title" in d
        assert isinstance(d["required_skills"], list)


class TestResumeKeywordAnalysisSchema:
    def test_creates_valid_instance(self, sample_resume_keywords):
        assert "Python" in sample_resume_keywords.technical_skills
        assert len(sample_resume_keywords.bullet_points) == 2

    def test_metrics_are_strings(self, sample_resume_keywords):
        for metric in sample_resume_keywords.metrics:
            assert isinstance(metric, str)

    def test_empty_resume_defaults(self):
        empty = ResumeKeywordAnalysis()
        assert empty.technical_skills == []
        assert empty.bullet_points == []
