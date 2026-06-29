"""
src/utils/normaliser.py

Domain knowledge base for ML / AI / Data Engineering keyword normalisation.

This module acts as a vocabulary for the AI Engineer / MLE job domain.
It maps the many ways the same concept is expressed in JDs and resumes
to a single canonical form, enabling accurate keyword matching.

Pipeline per keyword phrase:
  1. Lowercase + strip punctuation + hyphens → spaces
  2. Expand known abbreviations (whole phrase, then per token)
  3. Apply synonym / variant mapping
  4. Stem each token (simple suffix rules)
  5. Short phrase match: ALL meaningful tokens must be present
     Long phrase match: AT LEAST ONE meaningful token must be present
"""

from __future__ import annotations
import re


# ─────────────────────────────────────────────────────────────────────────────
# 1. ABBREVIATION EXPANSION
#    Expands acronyms to their full form before matching.
#    Keys: lowercase abbreviation  →  Value: expanded (lowercase, space-separated)
# ─────────────────────────────────────────────────────────────────────────────

ABBREV_MAP: dict[str, str] = {

    # ── LLM / Generative AI ────────────────────────────────────────────────
    "llm":      "large language model",
    "llms":     "large language models",
    "gpt":      "generative pretrained transformer",
    "vlm":      "vision language model",
    "mllm":     "multimodal large language model",
    "slm":      "small language model",
    "fm":       "foundation model",
    "genai":    "generative ai",

    # ── Training paradigms ─────────────────────────────────────────────────
    "sft":      "supervised fine tuning",
    "rlhf":     "reinforcement learning human feedback",
    "rlaif":    "reinforcement learning ai feedback",
    "dpo":      "direct preference optimization",
    "ppo":      "proximal policy optimization",
    "grpo":     "group relative policy optimization",
    "orpo":     "odds ratio preference optimization",
    "kto":      "kahneman tversky optimization",
    "ipo":      "identity preference optimization",
    "rft":      "reinforcement fine tuning",
    "cpt":      "continued pretraining",
    "pt":       "pretraining",

    # ── Parameter-efficient fine-tuning ───────────────────────────────────
    "peft":     "parameter efficient fine tuning",
    "lora":     "low rank adaptation",
    "qlora":    "quantized low rank adaptation",
    "dora":     "weight decomposed low rank adaptation",
    "ia3":      "infused adapter inhibited activations",
    "adapter":  "adapter fine tuning",
    "prefix":   "prefix tuning",
    "prompt":   "prompt tuning",

    # ── Retrieval & RAG ────────────────────────────────────────────────────
    "rag":      "retrieval augmented generation",
    "hyde":     "hypothetical document embeddings",
    "raptor":   "recursive abstractive processing tree organized retrieval",
    "colbert":  "contextualized late interaction bert",
    "ann":      "approximate nearest neighbor",
    "hnsw":     "hierarchical navigable small world",
    "ivf":      "inverted file index",

    # ── Evaluation ────────────────────────────────────────────────────────
    "rouge":    "recall oriented understudy gisting evaluation",
    "bleu":     "bilingual evaluation understudy",
    "bertscore": "bert score",
    "ragas":    "retrieval augmented generation assessment",
    "mtbench":  "mt bench",
    "hellaswag": "hella swag",
    "mmlu":     "massive multitask language understanding",
    "humaneval": "human eval",

    # ── Models (common abbreviations in JDs) ──────────────────────────────
    "bert":     "bidirectional encoder representations transformers",
    "gpt2":     "gpt 2",
    "gpt3":     "gpt 3",
    "gpt4":     "gpt 4",
    "t5":       "text to text transfer transformer",
    "moe":      "mixture of experts",
    "mla":      "multi head latent attention",

    # ── MLOps / Infrastructure ────────────────────────────────────────────
    "mlops":    "machine learning operations",
    "llmops":   "large language model operations",
    "cicd":     "continuous integration continuous deployment",
    "ci":       "continuous integration",
    "cd":       "continuous deployment",
    "iac":      "infrastructure as code",
    "k8s":      "kubernetes",

    # ── ML fundamentals ───────────────────────────────────────────────────
    "ml":       "machine learning",
    "dl":       "deep learning",
    "nlp":      "natural language processing",
    "cv":       "computer vision",
    "rl":       "reinforcement learning",
    "sl":       "supervised learning",
    "ul":       "unsupervised learning",
    "ssl":      "self supervised learning",
    "mtl":      "multi task learning",
    "tl":       "transfer learning",
    "fl":       "federated learning",
    "xai":      "explainable ai",
    "ai":       "artificial intelligence",

    # ── Neural net architectures ──────────────────────────────────────────
    "cnn":      "convolutional neural network",
    "rnn":      "recurrent neural network",
    "lstm":     "long short term memory",
    "gru":      "gated recurrent unit",
    "gcn":      "graph convolutional network",
    "gnn":      "graph neural network",
    "vae":      "variational autoencoder",
    "gan":      "generative adversarial network",
    "ddpm":     "denoising diffusion probabilistic model",

    # ── Data & Feature Engineering ────────────────────────────────────────
    "eda":      "exploratory data analysis",
    "etl":      "extract transform load",
    "elt":      "extract load transform",
    "pca":      "principal component analysis",
    "svd":      "singular value decomposition",
    "tsne":     "t distributed stochastic neighbor embedding",
    "umap":     "uniform manifold approximation projection",

    # ── Cloud & Infra ─────────────────────────────────────────────────────
    "aws":      "amazon web services",
    "gcp":      "google cloud platform",
    "az":       "azure",
    "ec2":      "elastic compute cloud",
    "s3":       "simple storage service",
    "emr":      "elastic map reduce",
    "gke":      "google kubernetes engine",
    "eks":      "elastic kubernetes service",
    "aks":      "azure kubernetes service",

    # ── APIs / Protocols ──────────────────────────────────────────────────
    "api":      "application programming interface",
    "rest":     "representational state transfer",
    "grpc":     "google remote procedure call",
    "http":     "hypertext transfer protocol",

    # ── Hardware ──────────────────────────────────────────────────────────
    "gpu":      "graphics processing unit",
    "tpu":      "tensor processing unit",
    "cpu":      "central processing unit",
    "fpga":     "field programmable gate array",

    # ── Quantization / Efficiency ─────────────────────────────────────────
    "int8":     "8 bit integer quantization",
    "fp16":     "16 bit floating point",
    "bf16":     "bfloat 16",
    "fp8":      "8 bit floating point",
    "awq":      "activation aware weight quantization",
    "gptq":     "generative pretrained transformer quantization",
    "gguf":     "ggml unified format",

    # ── Serving / Deployment ──────────────────────────────────────────────
    "vllm":     "vllm inference",
    "tgi":      "text generation inference",
    "triton":   "triton inference server",
    "onnx":     "open neural network exchange",
    "torchscript": "torch script",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. SYNONYM / VARIANT MAPPING
#    Maps spelling variants, hyphenation differences, and near-synonyms
#    to a single canonical form (lowercase, space-separated).
# ─────────────────────────────────────────────────────────────────────────────

SYNONYM_MAP: dict[str, str] = {

    # ── Fine-tuning variants ───────────────────────────────────────────────
    "finetuning":           "fine tuning",
    "fine-tuning":          "fine tuning",
    "finetune":             "fine tuning",
    "fine-tune":            "fine tuning",
    "fine tuned":           "fine tuning",
    "finetuned":            "fine tuning",
    "model finetuning":     "fine tuning",
    "model fine-tuning":    "fine tuning",
    "instruction tuning":   "fine tuning",
    "instruction-tuning":   "fine tuning",

    # ── Pre-training variants ──────────────────────────────────────────────
    "pretraining":          "pre training",
    "pre-training":         "pre training",
    "pretrain":             "pre training",
    "post training":        "post training",
    "post-training":        "post training",
    "posttraining":         "post training",

    # ── Hugging Face variants ─────────────────────────────────────────────
    "huggingface":          "hugging face",
    "hugging-face":         "hugging face",
    "hf":                   "hugging face",
    "huggingface hub":      "hugging face",
    "hugging face hub":     "hugging face",

    # ── PyTorch variants ──────────────────────────────────────────────────
    "torch":                "pytorch",
    "pytorch lightning":    "pytorch",

    # ── TensorFlow variants ───────────────────────────────────────────────
    "tf":                   "tensorflow",
    "tensorflow 2":         "tensorflow",
    "tf2":                  "tensorflow",
    "keras":                "tensorflow",  # Keras is now TF-native

    # ── Vector DB / Search ────────────────────────────────────────────────
    "vector store":         "vector database",
    "vector db":            "vector database",
    "vectordb":             "vector database",
    "vector search":        "vector search",
    "semantic search":      "vector search",
    "embedding search":     "vector search",
    "similarity search":    "vector search",

    # ── Evaluation ────────────────────────────────────────────────────────
    "eval":                 "evaluation",
    "evals":                "evaluation",
    "model eval":           "model evaluation",
    "model evaluation":     "model evaluation",
    "llm eval":             "llm evaluation",
    "llm evaluation":       "llm evaluation",

    # ── Prompt engineering ────────────────────────────────────────────────
    "prompt eng":           "prompt engineering",
    "prompting":            "prompt engineering",
    "chain of thought":     "chain of thought prompting",
    "cot":                  "chain of thought prompting",
    "few shot":             "few shot prompting",
    "zero shot":            "zero shot prompting",

    # ── RAG variants ──────────────────────────────────────────────────────
    "retrieval augmented":  "retrieval augmented generation",
    "retrieval-augmented":  "retrieval augmented generation",

    # ── Embeddings ────────────────────────────────────────────────────────
    "text embedding":       "embeddings",
    "sentence embedding":   "embeddings",
    "word embedding":       "embeddings",
    "embedding model":      "embeddings",

    # ── Distributed training ──────────────────────────────────────────────
    "dist training":            "distributed training",
    "distributed ml":           "distributed training",
    "data parallel":            "distributed training",
    "model parallel":           "distributed training",
    "tensor parallel":          "distributed training",
    "pipeline parallel":        "distributed training",
    "deepspeed":                "distributed training",
    "megatron":                 "distributed training",
    "fsdp":                     "distributed training",
    "ddp":                      "distributed training",

    # ── Inference / Serving ───────────────────────────────────────────────
    "model serving":        "model deployment",
    "model inference":      "inference",
    "llm serving":          "inference",
    "inference optimization": "inference",
    "model optimization":   "inference",

    # ── Data pipeline ─────────────────────────────────────────────────────
    "data pipelines":       "data pipeline",
    "ml pipeline":          "ml pipeline",
    "training pipeline":    "ml pipeline",
    "inference pipeline":   "ml pipeline",

    # ── MLOps tools ───────────────────────────────────────────────────────
    "weights and biases":   "weights biases",
    "weights & biases":     "weights biases",
    "wandb":                "weights biases",
    "ml flow":              "mlflow",
    "comet ml":             "comet",

    # ── Feature engineering ───────────────────────────────────────────────
    "feature store":        "feature engineering",
    "feature extraction":   "feature engineering",

    # ── Monitoring ────────────────────────────────────────────────────────
    "model monitoring":     "monitoring",
    "drift detection":      "monitoring",
    "data drift":           "monitoring",

    # ── A/B testing ───────────────────────────────────────────────────────
    "ab testing":           "ab testing",
    "a/b testing":          "ab testing",
    "a b testing":          "ab testing",
    # "experimentation" not mapped — too broad, causes false matches
    # "online experimentation" removed — too broad

    # ── General preference ─────────────────────────────────────────────────
    "preference optimisation": "preference optimization",
    "preference learning":     "preference optimization",
    "reward modeling":         "reward model",
    "reward modelling":        "reward model",
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. SUFFIX STEMMING
#    Lightweight suffix stripping — no external dependencies.
# ─────────────────────────────────────────────────────────────────────────────

_SUFFIX_RULES: list[tuple[str, str]] = [
    ("ization", "ize"),
    ("isation", "ize"),
    ("ations",  "ate"),
    ("ation",   "ate"),
    ("nesses",  ""),
    ("ness",    ""),
    ("ments",   "ment"),
    ("ities",   "ity"),
    ("ity",     "ity"),
    ("ings",    "ing"),
    ("ers",     "er"),
    ("ies",     "y"),
    ("es",      ""),
    ("s",       ""),
]

_MIN_STEM_LEN = 4


def _stem_token(token: str) -> str:
    if len(token) <= _MIN_STEM_LEN:
        return token
    for suffix, replacement in _SUFFIX_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM_LEN:
            return token[: len(token) - len(suffix)] + replacement
    return token


# ─────────────────────────────────────────────────────────────────────────────
# 4. STOPWORDS
#    Words that carry no skill signal on their own.
#    A match on ONLY these words is a false positive.
# ─────────────────────────────────────────────────────────────────────────────

_RAW_STOPWORDS: set[str] = {
    # Articles / prepositions
    "in", "with", "for", "of", "and", "or", "to", "a", "an", "the",
    "at", "on", "by", "as", "from", "into", "via", "per", "over",
    # Generic skill descriptors
    "skill", "experience", "knowledge", "understanding", "proficiency",
    "familiarity", "ability", "background", "basic", "advanced",
    "strong", "modern", "proven", "solid", "expert",
    # Generic verbs
    "design", "build", "work", "develop", "use", "create", "implement",
    "manage", "support", "perform", "ensure", "collaborate", "communicate",
    "deliver", "drive", "lead", "own", "define", "write", "review",
    # High-frequency generic nouns (appear in almost every resume/JD)
    "data", "model", "system", "large", "scale", "distributed",
    "solution", "product", "service", "environment", "result",
    "output", "input", "application", "approach", "method",
    "tool", "technology", "platform", "framework",
    # Soft skill words
    "problem", "team", "cross", "functional", "stakeholder",
    "communication", "collaboration",
    # Adjectives that appear as modifiers everywhere
    "real", "time", "high", "low", "end", "side",
    # Overly broad technical adjectives / generic verbs that appear everywhere
    "structured", "unstructured", "query", "language",
    "experiment", "systematic", "iterativ", "advanc", "scale",
}

# Build normalised stopword set (applied after stemming)
_STOPWORDS_NORM: set[str] = {_stem_token(w) for w in _RAW_STOPWORDS}


# ─────────────────────────────────────────────────────────────────────────────
# 5. PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def normalise_phrase(phrase: str) -> str:
    """
    Normalise a keyword phrase for comparison:
      1. Lowercase + hyphens/slashes → spaces + strip punctuation
      2. Whole-phrase abbreviation expansion
      3. Whole-phrase synonym mapping
      4. Per-token abbreviation + synonym expansion
      5. Stem each token
    """
    text = phrase.lower()
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in ABBREV_MAP:
        text = ABBREV_MAP[text]
    if text in SYNONYM_MAP:
        text = SYNONYM_MAP[text]

    tokens = text.split()
    expanded: list[str] = []
    for tok in tokens:
        tok = ABBREV_MAP.get(tok, tok)
        tok = SYNONYM_MAP.get(tok, tok)
        expanded.extend(tok.split())

    stemmed = [_stem_token(t) for t in expanded]
    return " ".join(stemmed)


def normalise_token_set(phrases: list[str]) -> set[str]:
    """Normalise a list of phrases → flat set of all tokens."""
    tokens: set[str] = set()
    for phrase in phrases:
        tokens.update(normalise_phrase(phrase).split())
    return tokens


def phrases_match(
    jd_keyword: str,
    resume_token_set: set[str],
    *,
    short_threshold: int = 2,
) -> bool:
    """
    Match a JD keyword against the resume token set.

    Strategy:
      - Extract meaningful (non-stopword) tokens from the normalised phrase
      - Short phrase (≤ short_threshold meaningful tokens):
          ALL meaningful tokens must be present  →  precision
      - Long phrase (> short_threshold meaningful tokens):
          At least ONE meaningful token must be present  →  recall
      - If no meaningful tokens exist: require ALL tokens (strict fallback)
    """
    norm = normalise_phrase(jd_keyword)
    norm_tokens = set(norm.split())

    if not norm_tokens:
        return False

    meaningful = norm_tokens - _STOPWORDS_NORM

    if not meaningful:
        # All tokens are stopwords — keyword is too generic to match meaningfully.
        # e.g. "structured data" → meaningful={} → reject rather than false-positive
        return False

    if len(meaningful) <= short_threshold:
        return meaningful.issubset(resume_token_set)
    else:
        return bool(meaningful & resume_token_set)