"""Gemini-backed research assistant, grounded in the paper's full content."""
from __future__ import annotations

import os
from typing import Iterator, List, Optional

from research import paper_context

MODEL_ID = "gemini-3.8-flash"

SUGGESTED_QUESTIONS = [
    "Why does removing dropout cause the generalization gap to jump to 2.02?",
    "What was the compute budget and GPU runtime for these experiments?",
    "How does the empirical baseline relate to character set size?",
]

SYSTEM_PROMPT = f"""You are the research assistant for a single paper, embedded in its interactive
dashboard. Your audience includes admissions committees, recruiters and ML engineers.

Ground rules:
1. Answer from the PAPER CONTEXT below. Quote exact numbers from it (nats/char unless stated).
2. When you add general ML background that the paper does not state (for example how dropout
   works mechanically), say so explicitly ("General background, not from the paper: ...").
3. Never invent results, seeds, runs, hyperparameters or citations that are not in the context.
   If something was not measured, say it was not measured and, where useful, point to the
   relevant item in the paper's Limitations.
4. Respect the paper's own caveats: one seed per setting, evaluation every 500 iterations,
   truncated schedule, only two dropout values. Do not overstate conclusions.
5. Be concise: short paragraphs or bullets, and LaTeX ($...$) for formulas where it helps.

PAPER CONTEXT
=============
{paper_context()}
"""


def resolve_api_key(user_key: Optional[str] = None) -> Optional[str]:
    """User-entered key > Streamlit secrets > environment variables."""
    if user_key and user_key.strip():
        return user_key.strip()
    try:
        import streamlit as st

        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if name in st.secrets:
                return str(st.secrets[name])
    except Exception:
        pass  # no secrets.toml present
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class AssistantError(RuntimeError):
    pass


def stream_reply(api_key: str, history: List[dict]) -> Iterator[str]:
    """
    history: [{"role": "user"|"assistant", "content": str}, ...], ending with the
    latest user turn. Yields text chunks as they arrive.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise AssistantError("The `google-genai` package is not installed.") from e

    try:
        client = genai.Client(api_key=api_key)
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in history
        ]
        stream = client.models.generate_content_stream(
            model=MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=1200,
            ),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        msg = str(e)
        if "API key" in msg or "API_KEY" in msg or "401" in msg or "403" in msg:
            raise AssistantError("The Gemini API rejected the key. Check that it is valid.") from e
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise AssistantError("Gemini rate limit reached. Wait a minute and try again.") from e
        raise AssistantError(f"Gemini request failed: {msg[:300]}") from e
    except Exception as e:
        msg = str(e)
        if "API key" in msg or "API_KEY" in msg or "401" in msg or "403" in msg:
            raise AssistantError("The Gemini API rejected the key. Check that it is valid.") from e
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise AssistantError("Gemini rate limit reached. Wait a minute and try again.") from e
        raise AssistantError(f"Gemini request failed: {msg[:300]}") from e
