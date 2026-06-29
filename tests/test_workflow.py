"""tests/test_workflow.py — updated for 6-node lean workflow."""

import pytest
from unittest.mock import MagicMock, patch
from src.schemas.outputs import JDKeywordAnalysis, ResumeKeywordAnalysis
from src.graph.routing import (
    route_after_input_validation,
    route_after_jd_extraction,
    route_after_resume_extraction,
)


class TestRouting:
    def test_validation_ok(self):
        assert route_after_input_validation({"errors": []}) == "jd_keyword_extractor_node"

    def test_validation_error(self):
        assert route_after_input_validation({"errors": ["missing"]}) == "error_end"

    def test_jd_ok(self):
        assert route_after_jd_extraction({"jd_keywords": MagicMock(), "retry_count": 0}) == "resume_keyword_extractor_node"

    def test_jd_retry(self):
        assert route_after_jd_extraction({"jd_keywords": None, "retry_count": 1}) == "jd_keyword_extractor_node"

    def test_jd_exhausted(self):
        assert route_after_jd_extraction({"jd_keywords": None, "retry_count": 99}) == "error_end"

    def test_resume_ok(self):
        assert route_after_resume_extraction({"resume_keywords": MagicMock(), "retry_count": 0}) == "keyword_matcher_node"

    def test_resume_retry(self):
        assert route_after_resume_extraction({"resume_keywords": None, "retry_count": 1}) == "resume_keyword_extractor_node"


class TestInputValidation:
    def test_both_missing(self):
        from src.graph.nodes import input_validation_node
        r = input_validation_node({"raw_resume": "", "raw_job_description": "", "errors": [], "retry_count": 0})
        assert len(r["errors"]) == 2

    def test_both_present(self):
        from src.graph.nodes import input_validation_node
        r = input_validation_node({"raw_resume": "resume", "raw_job_description": "jd", "errors": [], "retry_count": 0})
        assert r["errors"] == []


class TestKeywordMatcher:
    def test_pure_python_no_llm(self):
        from src.graph.nodes import keyword_matcher_node
        state = {
            "jd_keywords": JDKeywordAnalysis(
                role_title="AI Engineer",
                required_skills=["python", "langchain"],
                tools=["faiss"], frameworks=["fastapi"], domain_keywords=["rag"],
            ),
            "resume_keywords": ResumeKeywordAnalysis(
                technical_skills=["python"], tools=["faiss"],
                frameworks=["langchain", "fastapi"], domain_experience=["rag"],
                bullet_points=["built rag pipeline"],
            ),
            "errors": [],
        }
        r = keyword_matcher_node(state)
        assert "python" in r["matched_keywords"].exact_matches
        assert r["errors"] == []


class TestCacheLayer:
    def test_cache_miss_then_hit(self, tmp_path):
        """Same resume text → second call returns cached result."""
        import os
        os.environ["DB_PATH"] = str(tmp_path / "test.db")

        from src.storage import db as db_module
        db_module.DB_PATH = str(tmp_path / "test.db")

        from src.storage.db import get_cached_resume, cache_resume
        text = "Jane Doe, AI Engineer, Python, LangChain"
        keywords = {"technical_skills": ["python"], "tools": [], "frameworks": [],
                    "domain_experience": [], "project_keywords": [], "action_verbs": [],
                    "metrics": [], "bullet_points": []}

        assert get_cached_resume(text) is None          # miss
        cache_resume(text, keywords)
        result = get_cached_resume(text)
        assert result is not None                        # hit
        assert result["technical_skills"] == ["python"]

    def test_different_text_different_hash(self, tmp_path):
        import os
        os.environ["DB_PATH"] = str(tmp_path / "test2.db")
        from src.storage import db as db_module
        db_module.DB_PATH = str(tmp_path / "test2.db")
        from src.storage.db import get_cached_resume, cache_resume

        keywords = {"technical_skills": [], "tools": [], "frameworks": [],
                    "domain_experience": [], "project_keywords": [], "action_verbs": [],
                    "metrics": [], "bullet_points": []}
        cache_resume("resume A", keywords)
        assert get_cached_resume("resume B") is None    # different text → miss