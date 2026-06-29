# 📄 LangGraph Resume Copilot

An AI-powered resume optimization system built with **LangGraph**. Upload your resume and a job description — the workflow extracts keywords, scores the match with a 4-level confidence system, rewrites your bullet points, and lets you review before generating a final report.

Built as a portfolio project for AI Engineer / Applied AI / LLM Engineer job searches.

---

## Features

- **Section-aware resume extraction** — Education, Experience, and Projects extracted separately for higher precision
- **4-level keyword matching** — Strong Match / Partial Match / Related Experience / Missing, each with confidence score and resume evidence
- **Transparent match scoring** — 5-dimension weighted formula computed in pure Python (no LLM arithmetic)
- **Bulk bullet rewriting** — All bullets rewritten in a single LLM call with JD keyword alignment
- **SQLite caching** — Same resume across multiple JDs skips re-extraction entirely
- **LangSmith tracing** — Per-node latency, token usage, and LLM call monitoring
- **PDF upload support** — Upload resume as PDF or paste as text

---

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

### Why LangGraph?

| Feature | How it's used |
|---|---|
| Typed `StateGraph` | `GraphState` TypedDict flows through every node |
| Conditional edges | Missing inputs → early exit; extraction failures → retry |
| Node modularity | Each node has exactly one responsibility, independently testable |
| Retry logic | Failed structured output parsing retries up to 3 times |

---

## Match Scoring

Scoring is computed deterministically in `src/scoring/match_score.py` — the LLM never does arithmetic.

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

## Caching

Resume and JD keyword extractions are cached in SQLite by SHA-256 hash. Same resume across multiple JDs = only 1 LLM call per run.

**Benchmark results (1 resume × 5 JDs × 3 runs = 15 total):**

| | Cache OFF | Cache ON |
|---|---|---|
| Avg latency | 51.5s | 19.0s |
| Avg LLM calls | 3.0 | 1.0 |
| Latency reduction | — | **63%** |
| LLM call reduction | — | **67%** |
| Cache hit rate | — | **100%** |

---

## Repository Structure

```
langgraph-resume-copilot/
├── app.py                          # Streamlit UI
├── benchmark.py                    # Cache ON vs OFF benchmark
├── requirements.txt
├── .env.example
├── src/
│   ├── graph/
│   │   ├── state.py                # GraphState TypedDict
│   │   ├── nodes.py                # 6 node functions
│   │   ├── workflow.py             # Graph assembly
│   │   └── routing.py             # Conditional edge functions
│   ├── schemas/
│   │   └── outputs.py             # Pydantic schemas
│   ├── prompts/
│   │   └── prompts.py             # All LLM prompts
│   ├── scoring/
│   │   └── match_score.py         # 4-level match engine + weighted scorer
│   ├── rewriting/
│   │   └── grounding_checker.py   # Grounding check helpers
│   ├── storage/
│   │   └── db.py                  # SQLite cache + run history
│   └── utils/
│       ├── llm.py                  # Swappable LLM provider factory
│       ├── normaliser.py           # ML/AI domain keyword normaliser
│       ├── section_splitter.py     # Resume section parser
│       ├── config.py               # Configuration
│       ├── tracing.py              # LangSmith setup
│       └── logging.py             # Structured logging
├── tests/                          # 37 unit tests, 0 API calls required
└── data/
    ├── sample_resume.txt
    ├── sample_jd_ai_engineer.txt
    └── sample_jd_ml_engineer.txt
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

## Normaliser

`src/utils/normaliser.py` is a domain knowledge base for the ML/AI job market:

- **80+ abbreviation expansions** — SFT, RLHF, DPO, QLoRA, PEFT, RAG, vLLM, FSDP...
- **60+ synonym mappings** — fine-tuning variants, Hugging Face spellings, distributed training aliases...
- **Phrase-level matching** — 2-word phrases require both core tokens; long phrases require ≥1 meaningful token
- **Stopword filtering** — generic words like "data", "model", "system" don't trigger false matches alone

---

## Tech Stack

Python · LangGraph · LangChain · OpenAI API · Pydantic v2 · Streamlit · SQLite · LangSmith · pdfplumber · pytest