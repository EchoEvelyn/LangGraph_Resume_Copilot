"""tests/test_match_score.py — updated for SkillMatchResult."""

import pytest
from src.schemas.outputs import JDKeywordAnalysis, ResumeKeywordAnalysis
from src.scoring.match_score import compute_skill_match, calculate_match_score


@pytest.fixture
def good_match():
    jd = JDKeywordAnalysis(
        role_title="AI Engineer",
        required_skills=["python", "langchain", "rag"],
        nice_to_have_skills=["langgraph"],
        tools=["faiss", "docker"],
        frameworks=["langchain", "fastapi"],
        domain_keywords=["rag", "nlp"],
    )
    resume = ResumeKeywordAnalysis(
        technical_skills=["python"],
        tools=["faiss", "docker"],
        frameworks=["langchain", "fastapi", "langgraph"],
        domain_experience=["rag", "nlp"],
        bullet_points=["built rag pipeline with langchain and faiss"],
    )
    return jd, resume


@pytest.fixture
def poor_match():
    jd = JDKeywordAnalysis(
        role_title="ML Engineer",
        required_skills=["spark", "airflow", "pytorch"],
        nice_to_have_skills=["kubernetes"],
        tools=["feast", "sagemaker"],
        frameworks=["pytorch"],
        domain_keywords=["recommendation", "ranking"],
    )
    resume = ResumeKeywordAnalysis(
        technical_skills=["python"],
        tools=["faiss"],
        frameworks=["langchain"],
        domain_experience=["nlp"],
        bullet_points=["built rag pipeline"],
    )
    return jd, resume


class TestComputeSkillMatch:
    def test_required_matched(self, good_match):
        jd, resume = good_match
        sm = compute_skill_match(jd, resume)
        assert "python" in sm.required_matched
        assert "langchain" in sm.required_matched

    def test_required_missing(self, poor_match):
        jd, resume = poor_match
        sm = compute_skill_match(jd, resume)
        assert len(sm.required_missing) > 0
        assert "spark" in sm.required_missing

    def test_nice_matched(self, good_match):
        jd, resume = good_match
        sm = compute_skill_match(jd, resume)
        assert "langgraph" in sm.nice_matched

    def test_coverage_range(self, good_match):
        jd, resume = good_match
        sm = compute_skill_match(jd, resume)
        assert 0 <= sm.required_coverage <= 100
        assert 0 <= sm.nice_coverage <= 100

    def test_tool_matched(self, good_match):
        jd, resume = good_match
        sm = compute_skill_match(jd, resume)
        assert "faiss" in sm.tool_matched

    def test_all_missing_populated(self, poor_match):
        jd, resume = poor_match
        sm = compute_skill_match(jd, resume)
        assert len(sm.all_missing) > 0


class TestCalculateMatchScore:
    def test_good_match_high_score(self, good_match):
        jd, resume = good_match
        sm = compute_skill_match(jd, resume)
        score = calculate_match_score(jd, resume, sm)
        assert score.overall_score >= 60

    def test_poor_match_low_score(self, poor_match):
        jd, resume = poor_match
        sm = compute_skill_match(jd, resume)
        score = calculate_match_score(jd, resume, sm)
        assert score.overall_score < 50

    def test_score_in_range(self, good_match):
        jd, resume = good_match
        sm = compute_skill_match(jd, resume)
        score = calculate_match_score(jd, resume, sm)
        assert 0 <= score.overall_score <= 100

    def test_explanation_contains_score(self, good_match):
        jd, resume = good_match
        sm = compute_skill_match(jd, resume)
        score = calculate_match_score(jd, resume, sm)
        assert str(int(score.overall_score)) in score.explanation