"""
app.py — LangGraph Resume Copilot

UI sections:
  - Step 1: Input (PDF / paste + JD)
  - Step 2: Analysis
      Tab 1: Resume Sections (Education / Experience / Projects verbatim)
      Tab 2: Skill Match (Required / Nice-to-have / Tools split + matched/missing)
      Tab 3: Match Score
  - Step 3: Bullet Review
  - Step 4: Report
"""

from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(".env.example")

import streamlit as st

st.set_page_config(
    page_title="LangGraph Resume Copilot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.graph.workflow import build_graph
from src.graph.state import GraphState
from src.schemas.outputs import (
    RewrittenBullet,
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
    SkillMatchResult,
)
from src.storage.db import save_run, list_runs
from src.utils.tracing import init_tracing, get_run_metadata

_tracing_enabled = init_tracing()

SAMPLE_RESUME = Path("data/sample_resume.txt").read_text()
SAMPLE_JD     = Path("data/sample_jd_ai_engineer.txt").read_text()


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_pdf_text(uploaded_file) -> str:
    import pdfplumber
    with pdfplumber.open(uploaded_file) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages).strip()


def score_badge(score: float) -> str:
    color = "green" if score >= 75 else ("orange" if score >= 50 else "red")
    return (
        f'<span style="background:{color};color:white;'
        f'padding:4px 12px;border-radius:8px;font-weight:bold">'
        f'{score:.0f}/100</span>'
    )


def pct_badge(pct: float) -> str:
    color = "green" if pct >= 70 else ("orange" if pct >= 40 else "red")
    return f'<span style="color:{color};font-weight:bold">{pct:.0f}%</span>'


def apply_feedback(bullets: list[RewrittenBullet], feedback: dict) -> list[RewrittenBullet]:
    final = []
    for i, b in enumerate(bullets):
        d = feedback.get(str(i), "accept")
        if d == "reject":
            final.append(RewrittenBullet(
                original_bullet=b.original_bullet,
                rewritten_bullet=b.original_bullet,
                source_section=b.source_section,
                reason_for_rewrite="Rejected; original restored",
            ))
        elif d.startswith("edit:"):
            final.append(RewrittenBullet(
                original_bullet=b.original_bullet,
                rewritten_bullet=d[5:],
                jd_keywords_used=b.jd_keywords_used,
                source_section=b.source_section,
                reason_for_rewrite="Manually edited",
            ))
        else:
            final.append(b)
    return final


def report_to_markdown(result: dict, final_bullets: list[RewrittenBullet]) -> str:
    score = result.get("match_score")
    sm: SkillMatchResult | None = result.get("skill_match")
    lines = ["# Resume Optimization Report"]
    if score:
        lines += [
            f"\n## Match Score: {score.overall_score:.0f}/100",
            f"\n{score.explanation}",
            "\n## Score Breakdown",
            f"- Required Skills (50%): {score.required_skill_score:.1f}",
            f"- Tools/Frameworks (20%): {score.tool_overlap_score:.1f}",
            f"- Nice-to-have (20%): {score.nice_to_have_score:.1f}",
            f"- Domain (10%): {score.domain_score:.1f}",
        ]
    if sm:
        lines += [
            "\n## Required Skills",
            f"Matched: {', '.join(sm.required_matched) or 'none'}",
            f"Missing: {', '.join(sm.required_missing) or 'none'}",
            "\n## Nice-to-have Skills",
            f"Matched: {', '.join(sm.nice_matched) or 'none'}",
            f"Missing: {', '.join(sm.nice_missing) or 'none'}",
        ]
    lines += ["\n## Rewritten Bullets"]
    for b in final_bullets:
        lines += [f"\n**Original:** {b.original_bullet}",
                  f"**Final:** {b.rewritten_bullet}"]
    return "\n".join(lines)


# ── Session state ─────────────────────────────────────────────────────────────
for key in ["step", "pipeline_result", "final_bullets", "run_id"]:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state["step"] is None:
    st.session_state["step"] = "input"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    use_samples = st.checkbox("Load sample data", value=False)
    st.markdown("### Past Runs")
    try:
        runs = list_runs(limit=10)
        if runs:
            for r in runs:
                st.markdown(
                    f"**#{r['id']}** {r.get('job_title','?')} "
                    f"— {r.get('match_score',0):.0f}/100"
                )
        else:
            st.caption("No saved runs yet.")
    except Exception:
        st.caption("Database not initialised.")
    st.markdown("---")
    st.markdown("Built with **LangGraph** · 3 LLM calls · SQLite cache")
    st.markdown("---")
    if _tracing_enabled:
        st.success("🔍 LangSmith ON")
        st.caption(f"Project: {os.getenv('LANGCHAIN_PROJECT','langgraph-resume-copilot')}")
    else:
        st.warning("🔍 LangSmith OFF")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Input
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["step"] == "input":
    st.title("📄 LangGraph Resume Copilot")
    st.markdown(
        "Upload your resume and a job description. "
        "The workflow extracts sections, scores the match, and rewrites bullets — "
        "**3 LLM calls max**. Same resume is cached automatically."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Your Resume")
        input_mode = st.radio("Input method", ["Upload PDF", "Paste text"], horizontal=True)
        resume_text = ""
        if input_mode == "Upload PDF":
            pdf = st.file_uploader("Upload resume (PDF)", type=["pdf"])
            if pdf:
                try:
                    resume_text = extract_pdf_text(pdf)
                    st.success(f"✅ {len(resume_text):,} characters extracted")
                    with st.expander("Preview"):
                        st.text(resume_text[:2000] + ("…" if len(resume_text) > 2000 else ""))
                except ImportError:
                    st.error("Run: `pip install pdfplumber`")
                except Exception as e:
                    st.error(f"PDF error: {e}")
            elif use_samples:
                resume_text = SAMPLE_RESUME
                st.info("Using sample resume")
        else:
            resume_text = st.text_area(
                "Paste resume", value=SAMPLE_RESUME if use_samples else "",
                height=400, placeholder="Paste your full resume…",
            )

    with col2:
        st.subheader("💼 Job Description")
        jd_text = st.text_area(
            "Paste job description", value=SAMPLE_JD if use_samples else "",
            height=400, placeholder="Paste the full JD…",
        )

    st.markdown("---")
    if st.button("🚀 Analyse & Optimise", type="primary"):
        if not resume_text.strip() or not jd_text.strip():
            st.error("Please provide both a resume and a job description.")
        else:
            with st.spinner("Running… (~20–30s first run, faster with cache)"):
                try:
                    graph = build_graph()
                    initial_state: GraphState = {
                        "raw_resume":          resume_text,
                        "raw_job_description": jd_text,
                        "errors":              [],
                        "retry_count":         0,
                        "missing_keywords":    [],
                        "weak_keywords":       [],
                        "rewritten_bullets":   [],
                        "resume_cache_hit":    False,
                        "jd_cache_hit":        False,
                        "education_entries":   [],
                        "experience_entries":  [],
                        "project_entries":     [],
                    }
                    config = {
                        "run_name": f"copilot | {jd_text.split(chr(10))[0][:40].strip()}",
                        "metadata": get_run_metadata(
                            resume_length=len(resume_text),
                            jd_length=len(jd_text),
                        ),
                    }
                    result = graph.invoke(initial_state, config=config)
                    st.session_state["pipeline_result"] = result
                    st.session_state["step"] = "analysis"
                    st.rerun()
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state["step"] == "analysis":
    result = st.session_state["pipeline_result"]

    errors = result.get("errors", [])
    if errors:
        st.error("Pipeline errors:")
        for e in errors:
            st.markdown(f"- {e}")
        if st.button("↩ Start Over"):
            st.session_state["step"] = "input"
            st.rerun()
        st.stop()

    # Cache status
    c1, c2 = st.columns(2)
    c1.caption(f"Resume: {'⚡ cached' if result.get('resume_cache_hit') else '🔄 LLM extracted'}")
    c2.caption(f"JD: {'⚡ cached' if result.get('jd_cache_hit') else '🔄 LLM extracted'}")

    st.markdown("---")

    tab_resume, tab_jd = st.tabs(["📋 Resume", "🔍 JD Analysis & Match"])

    # ── Tab 1: Resume ─────────────────────────────────────────────────────────
    with tab_resume:
        edu:  list[EducationEntry]  = result.get("education_entries", [])
        exp:  list[ExperienceEntry] = result.get("experience_entries", [])
        proj: list[ProjectEntry]    = result.get("project_entries", [])

        st.markdown("### 🎓 Education")
        if edu:
            for e in edu:
                with st.expander(
                    f"{e.degree or 'Degree'} — {e.institution or 'Institution'} {e.period or ''}",
                    expanded=True,
                ):
                    st.text(e.raw_text)
        else:
            st.caption("No education entries extracted.")

        st.markdown("---")
        st.markdown("### 💼 Work / Internship Experience")
        if exp:
            for e in exp:
                label = f"{e.title or 'Role'} @ {e.company or 'Company'}"
                if e.period:
                    label += f" · {e.period}"
                with st.expander(label, expanded=True):
                    st.text(e.raw_text)
                    if e.bullets:
                        st.markdown("**Bullets extracted:**")
                        for b in e.bullets:
                            st.markdown(f"- {b}")
        else:
            st.caption("No experience entries extracted.")

        st.markdown("---")
        st.markdown("### 🛠️ Projects")
        if proj:
            for p in proj:
                with st.expander(p.project_name or "Project", expanded=True):
                    st.text(p.raw_text)
                    if p.bullets:
                        st.markdown("**Bullets extracted:**")
                        for b in p.bullets:
                            st.markdown(f"- {b}")
        else:
            st.caption("No project entries extracted.")

    # ── Tab 2: JD Analysis & Match ────────────────────────────────────────────
    with tab_jd:
        sm: SkillMatchResult | None = result.get("skill_match")
        jd_kw = result.get("jd_keywords")
        score = result.get("match_score")

        # Match score banner
        if score:
            score_col, exp_col = st.columns([1, 3])
            with score_col:
                st.markdown(score_badge(score.overall_score), unsafe_allow_html=True)
            with exp_col:
                st.caption(score.explanation)

            import pandas as pd
            df = pd.DataFrame({
                "Dimension": [
                    "Required Skills (50%)", "Tools/Frameworks (20%)",
                    "Nice-to-have (20%)", "Domain (10%)",
                ],
                "Score": [
                    score.required_skill_score, score.tool_overlap_score,
                    score.nice_to_have_score, score.domain_score,
                ],
            })
            st.bar_chart(df.set_index("Dimension"))
            st.markdown("---")


        if sm and jd_kw:
            from src.schemas.outputs import MATCH_LEVEL_LABELS

            LEVEL_COLORS = {
                "strong_match":       "#2e7d32",
                "partial_match":      "#f57f17",
                "related_experience": "#1565c0",
                "missing":            "#c62828",
            }
            LEVEL_SCORES = {"strong_match": 1.0, "partial_match": 0.6,
                            "related_experience": 0.4, "missing": 0.0}

            def _render_detail_group(title: str, details: list, coverage: float | None) -> None:
                st.markdown(f"### {title}")
                if coverage is not None:
                    n_nm = sum(1 for d in details if d.match_level != "missing")
                    st.markdown(
                        f"Weighted coverage: {pct_badge(coverage)} ({n_nm}/{len(details)} non-missing)",
                        unsafe_allow_html=True,
                    )
                groups = {"strong_match": [], "partial_match": [], "related_experience": [], "missing": []}
                for d in details:
                    groups[d.match_level].append(d)

                for level, level_details in groups.items():
                    if not level_details:
                        continue
                    color = LEVEL_COLORS[level]
                    label = MATCH_LEVEL_LABELS[level]
                    st.markdown(
                        f'<span style="color:{color};font-weight:bold">{label} ({len(level_details)})</span>',
                        unsafe_allow_html=True,
                    )
                    for d in level_details:
                        with st.expander(f"`{d.jd_keyword}`  — conf: {d.confidence_score:.0%}", expanded=False):
                            st.caption(f"**Why:** {d.explanation}")
                            if d.matched_resume_terms:
                                st.caption(f"**Matched on:** `{', '.join(d.matched_resume_terms)}`")
                            if d.resume_evidence:
                                st.caption(f"**Resume evidence:** _{d.resume_evidence}_")

            _render_detail_group("🔴 Required Skills", sm.required_matches, sm.required_coverage)
            st.markdown("---")
            _render_detail_group("🟡 Nice-to-have Skills", sm.nice_to_have_matches, sm.nice_coverage)
            st.markdown("---")
            _render_detail_group("🔧 Tools & Frameworks", sm.tool_matches, coverage=None)
            st.markdown("---")

            with st.expander("📄 Full JD Keyword Extraction", expanded=False):
                st.json({
                    "role_title":          jd_kw.role_title,
                    "required_skills":     jd_kw.required_skills,
                    "nice_to_have_skills": jd_kw.nice_to_have_skills,
                    "tools":               jd_kw.tools,
                })

    st.markdown("---")

    # ── Bullet Review ─────────────────────────────────────────────────────────
    st.markdown("## ✏️ Review Rewritten Bullets")
    st.markdown("**Accept** · **Reject** · **Edit**")

    bullets: list[RewrittenBullet] = result.get("rewritten_bullets") or []
    feedback: dict[str, str] = {}

    if bullets:
        if st.button("✅ Accept All"):
            for j in range(len(bullets)):
                st.session_state[f"decision_{j}"] = "Accept"

    for i, bullet in enumerate(bullets):
        changed = bullet.original_bullet != bullet.rewritten_bullet
        icon = "✏️" if changed else "—"
        preview = bullet.rewritten_bullet[:80] + ("…" if len(bullet.rewritten_bullet) > 80 else "")

        with st.expander(f"{icon} #{i+1} — {preview}", expanded=False):
            c_orig, c_new = st.columns(2)
            with c_orig:
                st.markdown("**Original**")
                st.info(bullet.original_bullet)
            with c_new:
                st.markdown("**Rewritten**")
                st.success(bullet.rewritten_bullet) if changed else st.info(bullet.rewritten_bullet)

            if bullet.jd_keywords_used:
                kw_str = ", ".join(bullet.jd_keywords_used)
                st.caption(f"Keywords used: {kw_str}")
            if bullet.reason_for_rewrite and bullet.reason_for_rewrite != "No change needed":
                st.caption(f"Why: {bullet.reason_for_rewrite}")

            decision = st.radio(
                "Decision", ["Accept", "Reject", "Edit"],
                horizontal=True, key=f"decision_{i}",
            )
            if decision == "Edit":
                custom = st.text_area("Your version", value=bullet.rewritten_bullet, key=f"edit_{i}")
                feedback[str(i)] = f"edit:{custom}"
            elif decision == "Reject":
                feedback[str(i)] = "reject"
            else:
                feedback[str(i)] = "accept"

    st.markdown("---")
    gen_col, back_col = st.columns([1, 5])
    with gen_col:
        done = st.button("✅ Finalise & View Report", type="primary", use_container_width=True)
    with back_col:
        if st.button("↩ Start Over"):
            st.session_state["step"] = "input"
            st.rerun()

    if done:
        final_bullets = apply_feedback(bullets, feedback)
        state_copy = dict(result)
        state_copy["final_bullets"] = final_bullets
        try:
            run_id = save_run(state_copy)
            st.session_state["run_id"] = run_id
        except Exception:
            pass
        st.session_state["pipeline_result"] = state_copy
        st.session_state["final_bullets"] = final_bullets
        st.session_state["step"] = "report"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Report
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state["step"] == "report":
    result       = st.session_state["pipeline_result"]
    final_bullets = st.session_state.get("final_bullets") or []

    run_id = st.session_state.get("run_id")
    if run_id:
        st.success(f"✅ Run #{run_id} saved")

    st.markdown("# 📄 Resume Optimization Report")

    score = result.get("match_score")
    sm: SkillMatchResult | None = result.get("skill_match")

    if score:
        st.markdown(f"## {score_badge(score.overall_score)}", unsafe_allow_html=True)
        st.caption(score.explanation)
        cols = st.columns(4)
        for col, (label, val) in zip(cols, [
            ("Required\nSkills (50%)", score.required_skill_score),
            ("Tools /\nFrameworks (20%)", score.tool_overlap_score),
            ("Nice-to-\nhave (20%)", score.nice_to_have_score),
            ("Domain (10%)", score.domain_score),
        ]):
            col.metric(label, f"{val:.0f}")

    if sm:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ❌ Missing Required Skills")
            for k in sm.required_missing:
                st.markdown(f"- `{k}`")
        with c2:
            st.markdown("### 🔑 Recommended to Add")
            for k in sm.recommended_keywords_to_emphasize:
                st.markdown(f"- `{k}`")

    st.markdown("---")
    st.markdown("### ✏️ Final Bullets")
    for b in final_bullets:
        changed = b.original_bullet != b.rewritten_bullet
        with st.expander(
            f"{'✏️' if changed else '—'} {b.rewritten_bullet[:70]}…",
            expanded=False,
        ):
            if changed:
                st.markdown(f"**Original:** {b.original_bullet}")
            st.markdown(f"**Final:** {b.rewritten_bullet}")

    st.markdown("---")
    st.download_button(
        "⬇️ Download Report (Markdown)",
        data=report_to_markdown(result, final_bullets),
        file_name="resume_optimization_report.md",
        mime="text/markdown",
    )
    if st.button("🔄 Optimise Another Resume"):
        for key in ["step", "pipeline_result", "final_bullets", "run_id"]:
            st.session_state[key] = None
        st.session_state["step"] = "input"
        st.rerun()