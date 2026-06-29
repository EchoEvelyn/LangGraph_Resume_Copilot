"""
src/utils/llm.py

Swappable LLM provider factory.
Change LLM_PROVIDER in .env to switch between OpenAI, Anthropic, Together, etc.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_llm(
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """
    Return a cached LangChain chat model.

    Swapping providers only requires changing env vars — no node code changes.

    Supported providers (set LLM_PROVIDER):
      openai      → ChatOpenAI
      anthropic   → ChatAnthropic
      together    → ChatTogether (via langchain_together)
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = temperature if temperature is not None else float(
        os.getenv("LLM_TEMPERATURE", "0.2")
    )
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "together":
        from langchain_together import ChatTogether  # pip install langchain-together
        return ChatTogether(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Choose openai / anthropic / together.")


def get_structured_llm(schema: type, **kwargs) -> BaseChatModel:
    """Return an LLM bound to a Pydantic schema for structured output."""
    return get_llm(**kwargs).with_structured_output(schema)
