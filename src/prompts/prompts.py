"""
src/prompts/prompts.py
"""

# ─── JD Extractor ─────────────────────────────────────────────────────────────

JD_KEYWORD_SYSTEM = """You are an expert technical recruiter.
Extract structured keyword data from the job description precisely.
Only extract what is explicitly stated. Return only JSON."""

JD_KEYWORD_USER = """Extract structured keywords from this job description.

JOB DESCRIPTION:
{job_description}

Extract ONLY these 4 fields:
- role_title: exact job title (string)
- required_skills: concise technical keywords, 1-4 words each. Extract the CORE SKILL not the full sentence.
  Good: "fine-tuning", "Python", "evaluation pipelines", "post-training"
  Bad: "hands-on experience fine-tuning and evaluating large language models", "strong programming skills in Python"
- nice_to_have_skills: same rule — concise keywords, 1-4 words each
- tools: specific tool/software/platform names only (e.g. PyTorch, Spark, Hugging Face, Kafka)

For the remaining schema fields (frameworks, domain_keywords, responsibilities, action_verbs, seniority_signals),
return empty lists [].
"""

# ─── Resume Section Extractors ────────────────────────────────────────────────

EDUCATION_EXTRACT_SYSTEM = """You are a resume parser.
Extract education entries from the text below.
Preserve the full original text of each entry verbatim."""

EDUCATION_EXTRACT_USER = """Extract all education entries from this section.

TEXT:
{text}

Return a JSON array called "entries". Each entry has:
- raw_text: the complete original text of this entry (verbatim)
- institution: school or university name
- degree: degree type and field (e.g. "B.Sc. Computer Science")
- period: dates (e.g. "2019-2022")

Return: {{"entries": [...]}}
"""

EXPERIENCE_EXTRACT_SYSTEM = """You are a resume parser.
Extract work and internship entries from the text below.
Preserve bullet points verbatim."""

EXPERIENCE_EXTRACT_USER = """Extract all work/internship entries from this section.

TEXT:
{text}

Return a JSON array called "entries". Each entry has:
- raw_text: complete original text of this entry including all bullets (verbatim)
- company: company name
- title: job title
- period: dates (e.g. "Jan 2023 – Present")
- bullets: list of bullet points verbatim (each starting with the original text)

Return: {{"entries": [...]}}
"""

PROJECTS_EXTRACT_SYSTEM = """You are a resume parser.
Extract project entries from the text below.
Preserve bullet points verbatim."""

PROJECTS_EXTRACT_USER = """Extract all project entries from this section.

TEXT:
{text}

Return a JSON array called "entries". Each entry has:
- raw_text: complete original text of this project including all bullets (verbatim)
- project_name: name of the project
- bullets: list of bullet points verbatim

Return: {{"entries": [...]}}
"""

SKILLS_EXTRACT_SYSTEM = """You are a resume parser.
Extract only skills, tools, and frameworks explicitly listed."""

SKILLS_EXTRACT_USER = """Extract from this Skills section:

{text}

Return JSON with keys: technical_skills, tools, frameworks.
Each is a list of strings. No other keys."""

# ─── Flat keyword extraction (fallback / summary) ────────────────────────────

RESUME_KEYWORD_SYSTEM = """You are an expert resume analyst.
Extract structured information from the candidate's resume.
Extract only what is present. Preserve bullet points verbatim."""

RESUME_KEYWORD_USER = """Extract all relevant keywords and bullet points from this resume.

RESUME:
{resume}

Extract: technical_skills, tools, frameworks, domain_experience,
project_keywords, action_verbs, metrics, bullet_points (verbatim).
"""

# ─── Bulk Bullet Rewriter ─────────────────────────────────────────────────────

BULK_REWRITER_SYSTEM = """You are an expert technical resume writer for AI/ML engineers.

RULES:
1. Use strong, specific action verbs.
2. Keep each bullet concise (1–2 lines max).
3. Naturally incorporate JD keywords — no keyword stuffing.
4. Preserve any quantified impact already present.
5. NEVER invent tools, metrics, scale, users, revenue, or production deployment.
6. Every claim must be traceable to the original bullet.
7. If a bullet cannot be improved for this JD, return it unchanged."""

BULK_REWRITER_USER = """Rewrite these resume bullets to align with the JD.
Return ALL bullets — improve where possible, keep unchanged if not applicable.

JD KEYWORDS:
{jd_keywords}

RESUME BULLETS:
{bullets}

For each bullet return:
- original_bullet: exact original text
- rewritten_bullet: improved version (or same if no change)
- jd_keywords_used: list of JD keywords incorporated
- source_section: "Experience" or "Projects"
- reason_for_rewrite: brief explanation or "No change needed"
"""