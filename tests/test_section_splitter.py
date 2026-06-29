"""tests/test_section_splitter.py"""

import pytest
from src.utils.section_splitter import split_resume_sections, get_bullets_section

SAMPLE_RESUME = """
Jane Doe | jane@email.com

SUMMARY
ML engineer with 3 years of experience building NLP pipelines.

EXPERIENCE
AI Engineer — Acme AI (2023–present)
• Built a RAG pipeline using LangChain and FAISS.
• Implemented an LLM evaluation framework with MLflow.

ML Engineer Intern — DataCo (2022)
• Trained a BERT classifier with Hugging Face Transformers.

PROJECTS
LangGraph Resume Copilot
• Built an 11-node LangGraph workflow for resume optimization.
• Added SQLite caching to skip redundant LLM calls.

SKILLS
Python, LangChain, LangGraph, FAISS, Docker, MLflow

EDUCATION
B.Sc. Computer Science — UC Berkeley (2022)
"""

RESUME_NO_HEADERS = """
Jane Doe, ML engineer, Python, LangChain, built RAG pipeline,
trained BERT model, 3 years experience.
"""


class TestSplitResumeSections:
    def test_detects_main_sections(self):
        sections = split_resume_sections(SAMPLE_RESUME)
        assert "experience" in sections
        assert "projects" in sections
        assert "skills" in sections

    def test_experience_contains_bullets(self):
        sections = split_resume_sections(SAMPLE_RESUME)
        assert "RAG pipeline" in sections["experience"]
        assert "LangChain" in sections["experience"]

    def test_projects_contains_project_content(self):
        sections = split_resume_sections(SAMPLE_RESUME)
        assert "LangGraph" in sections["projects"]

    def test_skills_contains_skills(self):
        sections = split_resume_sections(SAMPLE_RESUME)
        assert "Python" in sections["skills"]

    def test_no_headers_returns_full(self):
        sections = split_resume_sections(RESUME_NO_HEADERS)
        assert "full" in sections
        assert len(sections) == 1

    def test_sections_are_non_empty_strings(self):
        sections = split_resume_sections(SAMPLE_RESUME)
        for k, v in sections.items():
            assert isinstance(v, str)
            assert v.strip() != ""

    def test_case_insensitive_headers(self):
        text = "experience\n• Did something\n\nskills\nPython"
        sections = split_resume_sections(text)
        assert "experience" in sections
        assert "skills" in sections


class TestGetBulletsSection:
    def test_combines_experience_and_projects(self):
        sections = split_resume_sections(SAMPLE_RESUME)
        bullets_text = get_bullets_section(sections)
        assert "RAG pipeline" in bullets_text
        assert "LangGraph" in bullets_text

    def test_returns_string(self):
        sections = split_resume_sections(SAMPLE_RESUME)
        assert isinstance(get_bullets_section(sections), str)