# LangGraph Resume Copilot

An AI-powered resume optimization workflow that analyzes a resume against a job description, identifies skill gaps, scores alignment, rewrites bullet points, and generates a structured improvement report.

This project demonstrates production-style LLM application design: LangGraph orchestration, structured extraction, deterministic scoring, SQLite caching, LangSmith tracing, and human-in-the-loop review.

---

## What this project demonstrates

- Designing multi-step LLM workflows with LangGraph
- Using structured outputs with Pydantic schemas
- Separating LLM reasoning from deterministic business logic
- Building transparent scoring systems instead of relying on LLM-generated numbers
- Reducing latency and cost through SQLite caching
- Adding observability with LangSmith tracing
- Writing testable AI application components with mocked LLM calls

## Architecture

```
input_validation          (pure Python)
    ↓
jd_keyword_extractor      (LLM #1 — cached by JD hash)
    ↓
resume_keyword_extractor  (LLM #2 — cached by resume hash)
    ↓
keyword_matcher           (pure Python — 4-level match engine)
    ↓
match_score               (pure Python — weighted formula)
    ↓
bullet_rewriter           (LLM #3 — all bullets in 1 call)
    ↓
END
```

**6 nodes · 3 LLM calls max · SQLite cache**

---

## Features

### Resume & JD Understanding
- Section-aware resume extraction
- JD keyword extraction with structured outputs
- PDF upload and text input support

### Matching & Scoring
- 4-level keyword matching: Strong / Partial / Related / Missing
- Deterministic weighted scoring in pure Python
- Alias and synonym-aware matching for AI/ML keywords

### Workflow & Engineering
- LangGraph-based multi-node workflow
- SQLite caching by resume and JD hash
- LangSmith tracing for latency, token usage, and node-level monitoring
- Unit tests with mocked LLM calls

---

## Why LangGraph?

This project uses LangGraph because the resume optimization process is not a single prompt. It requires a controlled multi-step workflow:

1. Validate inputs
2. Extract JD requirements
3. Extract resume evidence
4. Match keywords deterministically
5. Compute scores
6. Rewrite bullets with human review

LangGraph makes this workflow explicit, testable, and observable through typed state, node-level separation, conditional routing, and retry logic.

---

## Match Scoring

Scoring is computed deterministically in `src/scoring/match_score.py` — To avoid hallucinated scores and inconsistent reasoning, all match scoring is computed deterministically in Python. The LLM is only used for extraction and rewriting, while scoring remains reproducible and testable.

| Dimension | Weight |
|---|---|
| Required skills | 50% |
| Tools / frameworks | 20% |
| Nice-to-have skills | 20% |
| Domain relevance | 10% |

### 4-Level Match System

| Level | Score | Criteria |
|---|---|---|
| ✅ Strong Match | 1.0 | All core tokens present or alias match |
| 🟡 Partial Match | 0.6 | Some core tokens present |
| 🔵 Related Experience | 0.4 | Adjacent concept found |
| ❌ Missing | 0.0 | No evidence |

### Alias Dictionary

The system knows that:
- `vector database` → FAISS, Chroma, Pinecone, vector search
- `model monitoring` → evaluation framework, performance tracking
- `model deployment` → inference workflow, production pipeline
- `workflow orchestration` → LangGraph, StateGraph, Airflow
- `data quality` → quality filtering, data validation, human calibration
- `rag` → retrieval augmented generation, RAG pipeline, RAG system

---

## Performance Optimization: SQLite Caching

Resume and JD extractions are cached by SHA-256 hash. This avoids repeated LLM calls when the same resume is evaluated against multiple job descriptions.

This is useful for real job-search workflows where one resume is repeatedly compared against many roles.

---

## Normaliser

`src/utils/normaliser.py` is a domain knowledge base for the ML/AI job market:

- **80+ abbreviation expansions** — SFT, RLHF, DPO, QLoRA, PEFT, RAG, vLLM, FSDP...
- **60+ synonym mappings** — fine-tuning variants, Hugging Face spellings, distributed training aliases...
- **Phrase-level matching** — 2-word phrases require both core tokens; long phrases require ≥1 meaningful token
- **Stopword filtering** — generic words like "data", "model", "system" don't trigger false matches alone


---

## Repository Structure

```text
src/
├── graph/        # LangGraph state, nodes, routing, workflow assembly
├── schemas/      # Pydantic structured output schemas
├── scoring/      # Deterministic keyword matching and scoring
├── rewriting/    # Bullet grounding checks
├── storage/      # SQLite cache and run history
├── prompts/      # LLM prompts
└── utils/        # LLM provider, PDF parsing, config, tracing
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/EchoEvelyn/LangGraph_Resume_Copilot
cd LangGraph_Resume_Copilot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=sk-...          # Required
LANGCHAIN_TRACING_V2=true      # Optional — LangSmith tracing
LANGCHAIN_API_KEY=ls__...      # Optional
LANGCHAIN_PROJECT=langgraph-resume-copilot
```

### 3. Run the app

```bash
python -m streamlit run app.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

No API key required — all LLM calls are mocked.

### 5. Run benchmark

```bash
# Add your resume and 5 JDs to data/
# my_resume.txt, jd_1.txt ... jd_5.txt

python benchmark.py
```

---

## Swap LLM Provider

Change `LLM_PROVIDER` in `.env`:

```bash
LLM_PROVIDER=openai      # default
LLM_PROVIDER=anthropic   # pip install langchain-anthropic
LLM_PROVIDER=together    # pip install langchain-together
```

No node code changes needed.

---

## Tech Stack

Python · LangGraph · LangChain · OpenAI API · Pydantic v2 · Streamlit · SQLite · LangSmith · pdfplumber · pytest