# 📄 LangGraph Resume Copilot

> An AI-powered resume optimization system built with **LangGraph**, demonstrating multi-node workflow orchestration, structured state management, hallucination prevention, and human-in-the-loop review.

---

## Project Summary

LangGraph Resume Copilot compares a candidate's resume against a job description and produces targeted, grounded rewrites of experience bullets. It is designed as a **portfolio project** for AI Engineer / Applied AI / LLM Engineer job searches — the codebase is intentionally modular and interview-explainable.

**What it does:**
1. Extracts structured keywords from the JD and resume using structured LLM outputs
2. Computes a transparent, code-driven match score (no LLM arithmetic)
3. Selects the highest-value bullets to rewrite
4. Rewrites them with JD keywords — without fabricating experience
5. Grounding-checks every rewrite against the original resume
6. Routes flagged bullets to a revision node before surfacing them for human review
7. Generates a final report with match score, gaps, and download

---

## Why LangGraph?

| LangGraph Feature | How This Project Uses It |
|---|---|
| **Typed `StateGraph`** | `GraphState` TypedDict flows through every node; each node reads and writes a well-defined slice |
| **Conditional edges** | Missing inputs → early exit; grounding failures → revision; retries on extraction errors |
| **Node modularity** | 11 single-responsibility nodes — easy to swap, test, or extend independently |
| **Human-in-the-loop** | `human_review_node` merges Streamlit feedback back into state before the final report |
| **Retry logic** | Failed structured output parsing retries up to `MAX_RETRIES` with `tenacity` |

A plain LangChain chain cannot express this kind of conditional, stateful, retry-aware workflow cleanly. LangGraph's `StateGraph` makes the control flow explicit and inspectable.

---

## Architecture Diagram

```
START
  │
  ▼
input_validation_node ──[missing inputs]──► END (error)
  │
  ▼
jd_keyword_extractor_node ──[retry/fail]──► END (error)
  │
  ▼
resume_keyword_extractor_node ──[retry/fail]──► END (error)
  │
  ▼
keyword_matcher_node
  │
  ▼
match_score_node  (pure Python arithmetic — no LLM)
  │
  ▼
bullet_selector_node
  │
  ▼
bullet_rewriter_node
  │
  ▼
grounding_checker_node
  │
  ├──[needs_revision=True]──► revision_node
  │                                │
  └──[all grounded]────────────────┤
                                   ▼
                           human_review_node  ◄── Streamlit UI
                                   │
                                   ▼
                           final_report_node
                                   │
                                   ▼
                                  END
```

---

## Repository Structure

```
langgraph-resume-copilot/
├── app.py                          # Streamlit UI (3-step flow)
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── graph/
│   │   ├── state.py                # GraphState TypedDict
│   │   ├── nodes.py                # All 11 node functions
│   │   ├── workflow.py             # Graph assembly & compile
│   │   └── routing.py             # Conditional edge functions
│   ├── schemas/
│   │   └── outputs.py             # 7 Pydantic output schemas
│   ├── prompts/
│   │   └── prompts.py             # All system + user prompts
│   ├── scoring/
│   │   └── match_score.py         # Transparent weighted scoring
│   ├── rewriting/
│   │   ├── bullet_rewriter.py     # Bullet selection + rewriting
│   │   └── grounding_checker.py   # Hallucination detection
│   ├── storage/
│   │   └── db.py                  # SQLite run persistence
│   └── utils/
│       ├── llm.py                  # Swappable LLM provider factory
│       ├── config.py               # Centralised configuration
│       └── logging.py              # Structured logging
├── tests/
│   ├── test_keyword_extraction.py
│   ├── test_match_score.py
│   ├── test_grounding_checker.py
│   └── test_workflow.py
└── data/
    ├── sample_resume.txt
    ├── sample_jd_ai_engineer.txt
    └── sample_jd_ml_engineer.txt
```

---

## Setup Instructions

### 1. Clone and install

```bash
git clone https://github.com/yourname/langgraph-resume-copilot
cd langgraph-resume-copilot
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY (or ANTHROPIC_API_KEY / TOGETHER_API_KEY)
# Change LLM_PROVIDER=openai to anthropic or together to swap providers
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

### 4. Run the test suite

```bash
pytest tests/ -v
```

Tests do **not** require an API key — all LLM calls are mocked.

---

## Swapping LLM Provider

Set `LLM_PROVIDER` in `.env`:

```bash
LLM_PROVIDER=openai      # default — requires OPENAI_API_KEY
LLM_PROVIDER=anthropic   # requires ANTHROPIC_API_KEY + pip install langchain-anthropic
LLM_PROVIDER=together    # requires TOGETHER_API_KEY + pip install langchain-together
```

No node code changes needed. The factory in `src/utils/llm.py` handles the switch.

---

## Scoring Formula

The match score is computed deterministically in `src/scoring/match_score.py` — the LLM is never asked to do arithmetic:

| Dimension | Weight | Measurement |
|---|---|---|
| Required skill coverage | **40%** | Fraction of JD required skills found in resume text |
| Tool / framework overlap | **20%** | Fraction of JD tools + frameworks found in resume |
| Responsibility alignment | **15%** | Fraction of JD responsibilities + action verbs found |
| Nice-to-have coverage | **15%** | Fraction of preferred skills found |
| Domain relevance | **10%** | Fraction of JD domain keywords found |

Coverage uses token-level matching (lower-cased, punctuation-stripped) so "PyTorch" matches "pytorch", and multi-word phrases like "retrieval augmented generation" match as a unit.

---

## Bullet Rewriting Rules

The system prompt enforces these rules on every rewrite:

- ✅ Strong, specific action verbs
- ✅ Concise (1–2 lines)
- ✅ JD keywords incorporated naturally
- ✅ Quantified impact preserved if already present
- ❌ Never invent tools, metrics, or scale
- ❌ Never claim Kubernetes, millions of users, enterprise scope, or production deployment unless in original
- ❌ Never claim team leadership unless in original
- ❌ Every claim must be traceable to the original bullet

**Example (AI Engineer JD):**

| | Text |
|---|---|
| **Original** | "Built an interactive Streamlit interface to enable rapid feedback collection and iterative system improvement." |
| **Rewritten** | "Built a Streamlit review interface for an LLM-powered RAG system, enabling structured user feedback collection and iterative evaluation of retrieval quality." |
| **Rejected** | ~~"Deployed a production-scale LLM platform serving enterprise customers."~~ — unsupported |

---

## Example Input / Output

**Input:** Senior ML engineer resume + AI Engineer JD (see `data/`)

**Output:**
```
Match Score: 78/100

Required Skills:    92%  (Python, LangChain, RAG, FAISS ✓)
Tools/Frameworks:   85%  (FAISS, Docker, MLflow, Weights & Biases ✓)
Responsibility:     70%  (build, deploy, evaluate ✓)
Nice-to-have:       60%  (LangGraph ✓, Kubernetes ✗)
Domain:             80%  (RAG, NLP, prompt engineering ✓)

Missing: Kubernetes, streaming, async Python, safety/alignment
Weak: fine-tuning (mentioned but not emphasised)

Rewritten bullets: 6 (5 accepted, 1 edited by user)
Grounding: 6/6 passed (confidence ≥ 0.75)
```

---

## Evaluation Metrics

| Metric | Measurement |
|---|---|
| **Keyword coverage** | % of JD required keywords present in resume after rewrite |
| **Grounding pass rate** | % of rewritten bullets with confidence ≥ threshold |
| **Match score delta** | Score increase from original to optimised resume |
| **Human accept rate** | % of rewrites accepted without edit in review step |

---

## Future Improvements

- [ ] **FAISS semantic search** — retrieve most relevant resume sections per JD keyword
- [ ] **Multi-turn revision** — allow user to iterate on bullets before final report
- [ ] **PDF / DOCX input** — parse uploaded resume files instead of plain text
- [ ] **Cover letter generation** — new graph branch after final report
- [ ] **LangSmith tracing** — add trace IDs for debugging individual runs
- [ ] **Async streaming** — stream LLM responses in the Streamlit UI
- [ ] **Batch mode** — run against multiple JDs and rank by match score
- [ ] **Fine-tuned rewriter** — LoRA-tune a smaller model on high-quality rewrite examples

---

## Resume Bullet Points for This Project

Use these on your own resume when applying for AI Engineer roles:

```
• Built a LangGraph multi-node workflow (11 nodes, conditional edges, typed GraphState) 
  for automated resume optimization against job descriptions.

• Implemented structured LLM outputs with Pydantic schemas for 7 output types; 
  added tenacity-based retry logic for robust production-grade parsing.

• Designed a hallucination prevention layer that grounding-checks each rewritten 
  bullet against the source resume before surfacing to the user.

• Engineered a transparent, code-driven match scoring formula (weighted across 
  5 dimensions) that quantifies resume-JD alignment without relying on LLM arithmetic.

• Built a human-in-the-loop review step in Streamlit allowing accept / reject / edit 
  decisions per bullet, integrated as a LangGraph state transition.

• Made the LLM provider fully swappable (OpenAI / Anthropic / Together) via 
  a factory pattern in a single utility module.
```

---

## License

MIT — free to use, fork, and add to your portfolio.
