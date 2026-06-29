"""
src/scoring/match_score.py

4-level keyword matching engine with phrase-level precision.

Match levels:
  strong_match       — exact/alias phrase match or high token overlap (≥0.8)
  partial_match      — most core tokens present (≥0.5) or alias partial hit
  related_experience — adjacent concept present (≥0.3)
  missing            — no reliable evidence

Scoring weights:
  required_skills    50%
  tools/frameworks   20%
  nice_to_have       20%
  domain             10%
"""

from __future__ import annotations
import re
from src.schemas.outputs import (
    JDKeywordAnalysis,
    ResumeKeywordAnalysis,
    KeywordMatchDetail,
    SkillMatchResult,
    KeywordMatchResult,
    MatchScore,
    MATCH_LEVEL_SCORES,
    MatchLevel,
)
from src.utils.normaliser import normalise_phrase, normalise_token_set
from src.utils.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)

W_REQUIRED = Config.WEIGHT_REQUIRED_SKILLS   # 0.50
W_NICE     = Config.WEIGHT_NICE_TO_HAVE      # 0.20
W_TOOLS    = Config.WEIGHT_TOOL_OVERLAP      # 0.20
W_DOMAIN   = Config.WEIGHT_DOMAIN            # 0.10
W_RESP     = Config.WEIGHT_RESPONSIBILITY    # 0.00


# ─────────────────────────────────────────────────────────────────────────────
# Domain stopwords — excluded from core-token overlap calculation
# These words are too common to count as skill evidence on their own
# but ARE kept for context (unlike generic stopwords in normaliser.py)
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_STOPWORDS: set[str] = {
    "experi", "knowledg", "understand", "profici", "familiar",
    "abil", "strong", "modern", "advanc", "basic",
    "skill", "background",
    "in", "with", "for", "of", "and", "or", "to", "a", "an", "the",
    "at", "on", "by", "as", "from", "into", "via", "per",
    "design", "build", "work", "develop", "us", "creat", "implement",
    "manag", "support", "perform", "ensur", "collabor", "communic",
}


# ─────────────────────────────────────────────────────────────────────────────
# Alias dictionary — maps JD concepts to resume equivalents
# ─────────────────────────────────────────────────────────────────────────────

ALIAS_MAP: dict[str, list[str]] = {
    # Vector / Retrieval
    "vector database":          ["faiss", "chroma", "pinecone", "weaviate", "qdrant",
                                  "vector search", "embedding search", "similarity search",
                                  "ann", "approximate nearest neighbor"],
    "vector search":            ["faiss", "vector database", "embedding search",
                                  "similarity search", "semantic search"],
    "embeddings":               ["embedding model", "sentence transformer", "text embedding",
                                  "word embedding", "dense retrieval"],

    # LLM Applications
    "llm application":          ["llm powered", "ai agent", "rag system", "chatbot",
                                  "language model", "generative ai"],
    "llm powered":              ["llm application", "ai agent", "rag system"],
    "rag system":               ["retrieval augmented", "rag pipeline", "rag"],
    "rag":                      ["retrieval augmented generation", "rag pipeline",
                                  "rag system", "retrieval pipeline"],

    # Model Lifecycle
    "model monitoring":         ["model evaluation", "performance tracking",
                                  "evaluation framework", "model performance",
                                  "drift detection", "monitoring"],
    "model deployment":         ["inference workflow", "production", "api integration",
                                  "model serving", "deploy", "deployment"],
    "model evaluation":         ["evaluation framework", "eval", "model monitoring",
                                  "performance metrics", "benchmarking"],
    "evaluation framework":     ["model evaluation", "eval framework", "ragas",
                                  "benchmarking", "model monitoring"],

    # Training
    "fine tuning":              ["fine-tuning", "finetuning", "instruction tuning",
                                  "sft", "qlora", "lora", "peft", "rlhf"],
    "pre training":             ["pretraining", "pre-training", "continued pretraining"],
    "post training":            ["post-training", "posttraining", "rlhf", "dpo",
                                  "preference optimization", "sft"],
    "rlhf":                     ["reinforcement learning human feedback", "reward model",
                                  "preference optimization", "dpo", "ppo"],
    "distributed training":     ["data parallel", "model parallel", "deepspeed",
                                  "fsdp", "multi gpu", "multi node"],

    # Orchestration
    "workflow orchestration":   ["langgraph", "stategraph", "airflow", "prefect",
                                  "dagster", "conditional routing", "pipeline orchestration"],
    "ml pipeline":              ["training pipeline", "inference pipeline",
                                  "data pipeline", "etl pipeline"],

    # Data
    "data quality":             ["quality filtering", "data validation",
                                  "human calibration", "data cleaning", "data curation"],
    "data pipeline":            ["etl", "data engineering", "feature pipeline",
                                  "data ingestion", "data processing"],
    "feature engineering":      ["feature extraction", "feature store",
                                  "feature selection", "feature creation"],

    # Infrastructure
    "mlops":                    ["ml pipeline", "model deployment", "model monitoring",
                                  "ci/cd", "experiment tracking", "mlflow", "wandb"],
    "inference":                ["model serving", "model deployment", "vllm",
                                  "triton", "tgi", "onnx", "torchscript"],

    # NLP/ML concepts
    "natural language processing": ["nlp", "text classification", "ner", "sentiment",
                                     "text processing", "language model"],
    "machine learning":         ["ml", "supervised learning", "model training",
                                  "scikit learn", "sklearn"],
    "deep learning":            ["dl", "neural network", "pytorch", "tensorflow",
                                  "transformer", "cnn", "rnn"],

    # Soft / Process
    "experiment tracking":      ["mlflow", "wandb", "weights biases", "comet",
                                  "neptune", "experiment management"],
    "prompt engineering":       ["prompting", "few shot", "zero shot",
                                  "chain of thought", "system prompt"],
    "ab testing":               ["ab testing", "experimentation", "hypothesis testing",
                                  "statistical testing", "online experiment"],
}

# Reverse alias lookup: resume term → JD concepts it can satisfy
def _build_reverse_aliases() -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}
    for jd_concept, aliases in ALIAS_MAP.items():
        for alias in aliases:
            norm_alias = normalise_phrase(alias)
            reverse.setdefault(norm_alias, []).append(jd_concept)
    return reverse

_REVERSE_ALIASES = _build_reverse_aliases()


# ─────────────────────────────────────────────────────────────────────────────
# Core matching engine
# ─────────────────────────────────────────────────────────────────────────────

def _core_tokens(norm_phrase: str) -> set[str]:
    """Meaningful tokens after removing domain stopwords."""
    tokens = set(norm_phrase.split())
    core = tokens - _DOMAIN_STOPWORDS
    return core if core else tokens


def _build_resume_token_set(
    resume: ResumeKeywordAnalysis,
    raw_resume: str = "",
) -> set[str]:
    all_phrases = (
        resume.technical_skills + resume.tools + resume.frameworks
        + resume.domain_experience + resume.project_keywords
        + resume.action_verbs + resume.bullet_points
    )
    tokens = normalise_token_set(all_phrases)
    if raw_resume:
        tokens.update(normalise_token_set([raw_resume]))
    return tokens


def _find_best_evidence(
    keyword: str,
    resume: ResumeKeywordAnalysis,
    raw_resume: str,
) -> tuple[str, list[str]]:
    """
    Find the best resume sentence and matched terms for a keyword.
    Returns (snippet, matched_terms).
    """
    norm_kw = normalise_phrase(keyword)
    kw_core = _core_tokens(norm_kw)

    best_snippet = ""
    best_terms: list[str] = []
    best_score = 0

    candidates = resume.bullet_points + resume.technical_skills + resume.tools + resume.frameworks + resume.domain_experience
    if raw_resume:
        candidates += [l.strip() for l in re.split(r"[.\n]", raw_resume) if l.strip()]

    for cand in candidates:
        norm_cand = normalise_phrase(cand)
        cand_tokens = set(norm_cand.split())
        hit = kw_core & cand_tokens
        if len(hit) > best_score:
            best_score = len(hit)
            snippet = cand.strip()
            best_snippet = snippet[:120] + "…" if len(snippet) > 120 else snippet
            best_terms = sorted(hit)

    return best_snippet, best_terms


def _token_overlap(norm_a: str, token_set_b: set[str]) -> float:
    """Fraction of core tokens of A that appear in B."""
    core_a = _core_tokens(norm_a)
    if not core_a:
        return 0.0
    return len(core_a & token_set_b) / len(core_a)


def _check_aliases(keyword: str, resume_tokens: set[str]) -> tuple[bool, list[str]]:
    """
    Check if any alias of this keyword appears in the resume token set.
    Returns (found, matched_alias_terms).
    """
    norm_kw = normalise_phrase(keyword)
    aliases = ALIAS_MAP.get(norm_kw, []) + ALIAS_MAP.get(keyword.lower(), [])

    matched_aliases: list[str] = []
    for alias in aliases:
        norm_alias = normalise_phrase(alias)
        alias_core = _core_tokens(norm_alias)
        if alias_core and alias_core.issubset(resume_tokens):
            matched_aliases.append(alias)

    # Also check reverse: does any resume token map back to this concept?
    norm_kw_core = _core_tokens(norm_kw)
    for res_tok in resume_tokens:
        for jd_concept in _REVERSE_ALIASES.get(res_tok, []):
            norm_concept = normalise_phrase(jd_concept)
            if _token_overlap(norm_kw, set(norm_concept.split())) >= 0.5:
                matched_aliases.append(res_tok)

    return bool(matched_aliases), list(set(matched_aliases))[:3]


def match_single_keyword(
    keyword: str,
    resume: ResumeKeywordAnalysis,
    resume_tokens: set[str],
    raw_resume: str = "",
) -> KeywordMatchDetail:
    """
    Assign a match level to a single JD keyword against the resume.

    Decision logic:
    1. Normalise keyword → get core tokens
    2. Compute token overlap ratio
    3. Check alias dictionary
    4. Assign level:
       - strong_match:       overlap ≥ 0.8  OR  exact alias hit
       - partial_match:      overlap ≥ 0.5  OR  partial alias hit
       - related_experience: overlap ≥ 0.3  OR  any alias token present
       - missing:            overlap < 0.3  AND  no alias evidence
    """
    norm_kw = normalise_phrase(keyword)
    kw_core = _core_tokens(norm_kw)
    n_core = len(kw_core)

    # Edge case: all tokens are stopwords → can't meaningfully match
    if not kw_core:
        return KeywordMatchDetail(
            jd_keyword=keyword,
            match_level="missing",
            confidence_score=0.0,
            matched_resume_terms=[],
            resume_evidence="",
            explanation="Keyword consists only of generic words; no specific skill to match.",
        )

    overlap = _token_overlap(norm_kw, resume_tokens)
    alias_found, alias_terms = _check_aliases(keyword, resume_tokens)

    evidence_snippet, matched_terms = _find_best_evidence(keyword, resume, raw_resume)

    # ── Determine match level ──────────────────────────────────────────────
    # For 2-word keywords: require both core tokens (unless alias saves it)
    # For 3+ word keywords: token overlap threshold applies

    if n_core <= 2:
        # Short phrase: strict
        if kw_core.issubset(resume_tokens):
            level: MatchLevel = "strong_match"
            confidence = 0.9 + 0.1 * overlap
            explanation = f"All core tokens {sorted(kw_core)} found in resume."
        elif alias_found:
            level = "strong_match"
            confidence = 0.85
            explanation = f"Matched via alias: {alias_terms[0] if alias_terms else 'related term'}."
        else:
            # Single generic token hit is not enough for 2-word phrase
            partial_hit = kw_core & resume_tokens
            if partial_hit and len(partial_hit) < n_core:
                # Only some core tokens matched
                level = "partial_match"
                confidence = 0.5
                explanation = f"Partial match: {sorted(partial_hit)} found but not full concept."
            else:
                level = "missing"
                confidence = 0.0
                explanation = f"Core tokens {sorted(kw_core)} not found in resume."
    else:
        # Long phrase: threshold-based
        if overlap >= 0.8 or (alias_found and overlap >= 0.5):
            level = "strong_match"
            confidence = min(0.95, 0.7 + overlap * 0.3)
            explanation = f"High token overlap ({overlap:.0%}) with resume."
        elif overlap >= 0.5 or alias_found:
            level = "partial_match"
            confidence = 0.4 + overlap * 0.4
            explanation = (
                f"Partial token overlap ({overlap:.0%})."
                + (f" Alias hint: {alias_terms[0]}." if alias_terms else "")
            )
        elif overlap >= 0.3:
            level = "related_experience"
            confidence = 0.2 + overlap * 0.5
            explanation = f"Adjacent experience found (overlap {overlap:.0%})."
        else:
            level = "missing"
            confidence = 0.0
            explanation = "No reliable evidence found in resume."

    return KeywordMatchDetail(
        jd_keyword=keyword,
        match_level=level,
        confidence_score=round(confidence, 2),
        matched_resume_terms=matched_terms[:3],
        resume_evidence=evidence_snippet,
        explanation=explanation,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_skill_match(
    jd: JDKeywordAnalysis,
    resume: ResumeKeywordAnalysis,
    raw_resume: str = "",
) -> SkillMatchResult:
    resume_tokens = _build_resume_token_set(resume, raw_resume)

    def _match_group(keywords: list[str]) -> list[KeywordMatchDetail]:
        return [
            match_single_keyword(kw, resume, resume_tokens, raw_resume)
            for kw in keywords
        ]

    req_details  = _match_group(jd.required_skills)
    nice_details = _match_group(jd.nice_to_have_skills)
    tool_details = _match_group(jd.tools + jd.frameworks)

    def _weighted_score(details: list[KeywordMatchDetail]) -> float:
        if not details:
            return 100.0
        total = sum(
            MATCH_LEVEL_SCORES[d.match_level] * d.confidence_score
            for d in details
        )
        return round(total / len(details) * 100, 1)

    req_score  = _weighted_score(req_details)
    nice_score = _weighted_score(nice_details)

    req_matched  = [d.jd_keyword for d in req_details  if d.match_level != "missing"]
    req_missing  = [d.jd_keyword for d in req_details  if d.match_level == "missing"]
    nice_matched = [d.jd_keyword for d in nice_details if d.match_level != "missing"]
    nice_missing = [d.jd_keyword for d in nice_details if d.match_level == "missing"]
    tool_matched = [d.jd_keyword for d in tool_details if d.match_level != "missing"]
    tool_missing = [d.jd_keyword for d in tool_details if d.match_level == "missing"]

    all_missing  = list(dict.fromkeys(req_missing + nice_missing + tool_missing))
    recommended  = req_missing[:5] + [k for k in nice_missing if k not in req_missing][:3]

    # Build legacy evidence dict
    evidence = {
        d.jd_keyword: {"snippet": d.resume_evidence, "matched_tokens": ", ".join(d.matched_resume_terms)}
        for d in req_details + nice_details + tool_details
        if d.resume_evidence
    }

    logger.info(
        f"Match — required: {len(req_matched)}/{len(jd.required_skills)} ({req_score:.0f}%), "
        f"nice: {len(nice_matched)}/{len(jd.nice_to_have_skills)}, "
        f"tools: {len(tool_matched)}/{len(jd.tools + jd.frameworks)}"
    )

    return SkillMatchResult(
        required_matches=req_details,
        nice_to_have_matches=nice_details,
        tool_matches=tool_details,
        required_matched=req_matched,
        required_missing=req_missing,
        required_coverage=req_score,
        nice_matched=nice_matched,
        nice_missing=nice_missing,
        nice_coverage=nice_score,
        tool_matched=tool_matched,
        tool_missing=tool_missing,
        all_missing=all_missing,
        weak_keywords=[d.jd_keyword for d in req_details if d.match_level == "partial_match"],
        recommended_keywords_to_emphasize=recommended,
        evidence=evidence,
    )


def skill_match_to_keyword_match(sm: SkillMatchResult) -> KeywordMatchResult:
    return KeywordMatchResult(
        exact_matches=sm.required_matched + sm.nice_matched + sm.tool_matched,
        missing_keywords=sm.all_missing,
        weak_keywords=sm.weak_keywords,
        recommended_keywords_to_emphasize=sm.recommended_keywords_to_emphasize,
    )


def calculate_match_score(
    jd: JDKeywordAnalysis,
    resume: ResumeKeywordAnalysis,
    skill_match: SkillMatchResult,
    raw_resume: str = "",
) -> MatchScore:

    def _group_score(details: list[KeywordMatchDetail]) -> float:
        if not details:
            return 100.0
        return round(
            sum(MATCH_LEVEL_SCORES[d.match_level] * d.confidence_score for d in details)
            / len(details) * 100, 1
        )

    required_score = _group_score(skill_match.required_matches)
    nice_score     = _group_score(skill_match.nice_to_have_matches)
    tool_score     = _group_score(skill_match.tool_matches)
    domain_score   = skill_match.required_coverage  # proxy
    resp_score     = 0.0

    overall = round(
        required_score * W_REQUIRED + nice_score * W_NICE
        + tool_score * W_TOOLS + domain_score * W_DOMAIN
        + resp_score * W_RESP, 1
    )

    top_missing = skill_match.required_missing[:4]
    partial     = [d.jd_keyword for d in skill_match.required_matches if d.match_level == "partial_match"][:2]

    explanation = (
        f"Overall: {overall}/100. "
        f"Required skills: {required_score:.0f}% (50%). "
        f"Tools/frameworks: {tool_score:.0f}% (20%). "
        f"Nice-to-have: {nice_score:.0f}% (20%). "
        f"Domain: {domain_score:.0f}% (10%). "
    )
    if top_missing:
        explanation += f"Missing: {', '.join(top_missing[:3])}. "
    if partial:
        explanation += f"Partial: {', '.join(partial)}."

    return MatchScore(
        overall_score=min(100.0, overall),
        required_skill_score=required_score,
        nice_to_have_score=nice_score,
        tool_overlap_score=tool_score,
        domain_score=domain_score,
        responsibility_alignment_score=resp_score,
        explanation=explanation,
    )