"""LLM-based qualitative analysis of podcast transcripts."""

from __future__ import annotations

import json
import logging
import os

from openai import OpenAI

import config
from src.models import LLMAnalysis, Transcription

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_model_name() -> str:
    """Return the model name to use for chat completions."""
    if config.USE_LOCAL_LLM:
        return config.OLLAMA_MODEL
    return config.OPENAI_MODEL


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if config.USE_LOCAL_LLM:
            logger.info(
                "Using local LLM via Ollama at %s (model=%s)",
                config.OLLAMA_BASE_URL,
                config.OLLAMA_MODEL,
            )
            _client = OpenAI(
                base_url=config.OLLAMA_BASE_URL,
                api_key="ollama",  # Ollama doesn't need a real key
            )
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "Set the OPENAI_API_KEY environment variable to use LLM analysis.\n"
                    "Or use --local-llm to use Ollama instead (free, runs locally)."
                )
            _client = OpenAI(api_key=api_key)
    return _client


SYSTEM_PROMPT = """\
You are an expert in Spanish language pedagogy and linguistics.
You will be given a transcript excerpt from a Spanish-language podcast.
Analyze it and return a JSON object (no markdown fences) with these fields:

{
  "slang_score": <float 0-1, 0 = no slang, 1 = heavy slang/colloquialisms>,
  "slang_examples": [<list of slang/colloquial phrases found>],
  "topic_complexity": <float 0-1, 0 = simple everyday topics, 1 = highly specialized/technical>,
  "topic_summary": "<brief description of the main topics discussed>",
  "estimated_cefr": "<one of A1, A2, B1, B2, C1, C2>",
  "explanation": "<2-3 sentence explanation of why you chose that CEFR level>",
  "idiom_count": <int, number of idiomatic expressions found>,
  "formality_score": <float 0-1, 0 = very informal, 1 = very formal>
}

Be precise and ground your analysis in the actual text provided.
Only return the JSON object, nothing else.
"""


def analyze_with_llm(transcript: Transcription) -> LLMAnalysis:
    """Send a transcript excerpt to an LLM for qualitative analysis.

    The transcript is truncated to LLM_MAX_TRANSCRIPT_CHARS to control cost.
    """
    text = transcript.full_text[: config.LLM_MAX_TRANSCRIPT_CHARS]
    if not text.strip():
        logger.warning("Empty transcript for LLM analysis")
        return LLMAnalysis()

    client = _get_client()

    model_name = _get_model_name()
    logger.info(
        "Sending %d chars to %s for analysis...", len(text), model_name
    )

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Analyze this Spanish podcast transcript excerpt:\n\n{text}"
                ),
            },
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model includes them anyway
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]  # drop first line
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON: %s", raw[:200])
        return LLMAnalysis(explanation=f"Failed to parse LLM response: {raw[:200]}")

    analysis = LLMAnalysis(
        slang_score=float(data.get("slang_score", 0)),
        slang_examples=data.get("slang_examples", []),
        topic_complexity=float(data.get("topic_complexity", 0)),
        topic_summary=data.get("topic_summary", ""),
        estimated_cefr=data.get("estimated_cefr", ""),
        explanation=data.get("explanation", ""),
        idiom_count=int(data.get("idiom_count", 0)),
        formality_score=float(data.get("formality_score", 0)),
    )

    logger.info(
        "LLM analysis: CEFR=%s, slang=%.2f, topic=%.2f",
        analysis.estimated_cefr,
        analysis.slang_score,
        analysis.topic_complexity,
    )
    return analysis
