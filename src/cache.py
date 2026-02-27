"""Transcription and analysis caching.

Cache files live in data/cache/ as JSON.  The cache key is a hash of the
episode URL plus the Whisper parameters that affect the output.  If you
change the model size, beam size, skip seconds, or transcription length,
the old cache is ignored and a fresh transcription is produced.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from src.models import (
    CachedAnalysis,
    CachedLLMAnalysis,
    CachedTranscription,
    LLMAnalysis,
    StructuralMetrics,
    Transcription,
    WhisperParams,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

def current_whisper_params() -> WhisperParams:
    """Snapshot the active Whisper settings from config."""
    return WhisperParams(
        model_size=config.WHISPER_MODEL_SIZE,
        beam_size=config.WHISPER_BEAM_SIZE,
        language=config.WHISPER_LANGUAGE,
        skip_intro_seconds=config.SKIP_INTRO_SECONDS,
        max_transcribe_minutes=config.MAX_TRANSCRIBE_MINUTES,
        first_half_only=config.FIRST_HALF_ONLY,
    )


def _cache_key(episode_url: str, params: WhisperParams) -> str:
    """Deterministic hash of episode URL + whisper params."""
    params_json = json.dumps(params.model_dump(), sort_keys=True)
    blob = f"{episode_url}|{params_json}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _transcription_path(episode_url: str, params: WhisperParams) -> Path:
    return config.CACHE_DIR / f"tx_{_cache_key(episode_url, params)}.json"


def _analysis_path(episode_url: str, params: WhisperParams) -> Path:
    return config.CACHE_DIR / f"an_{_cache_key(episode_url, params)}.json"


def _llm_path(episode_url: str, params: WhisperParams) -> Path:
    return config.CACHE_DIR / f"llm_{_cache_key(episode_url, params)}.json"


# ------------------------------------------------------------------
#  Transcription cache
# ------------------------------------------------------------------

def load_cached_transcription(
    episode_url: str,
    params: WhisperParams | None = None,
) -> Transcription | None:
    """Return a cached Transcription if one exists for these params, else None."""
    params = params or current_whisper_params()
    path = _transcription_path(episode_url, params)
    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cached = CachedTranscription.model_validate(raw)
        # Double-check the stored params still match
        if cached.whisper_params != params:
            logger.debug("Cache params mismatch for %s, ignoring", episode_url)
            return None
        logger.info("Loaded cached transcription for '%s'", cached.transcription.episode_title)
        return cached.transcription
    except Exception as exc:
        logger.warning("Failed to load cache %s: %s", path.name, exc)
        return None


def save_cached_transcription(
    episode_url: str,
    transcription: Transcription,
    params: WhisperParams | None = None,
) -> Path:
    """Persist a transcription to the cache and return the file path."""
    params = params or current_whisper_params()
    cached = CachedTranscription(
        episode_title=transcription.episode_title,
        whisper_params=params,
        episode_url=episode_url,
        transcription=transcription,
        cached_at=datetime.now(timezone.utc).isoformat(),
    )
    path = _transcription_path(episode_url, params)
    path.write_text(cached.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Cached transcription -> %s", path.name)
    return path


# ------------------------------------------------------------------
#  Structural-analysis cache
# ------------------------------------------------------------------

def load_cached_analysis(
    episode_url: str,
    params: WhisperParams | None = None,
) -> StructuralMetrics | None:
    """Return cached StructuralMetrics if available, else None."""
    params = params or current_whisper_params()
    path = _analysis_path(episode_url, params)
    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cached = CachedAnalysis.model_validate(raw)
        if cached.whisper_params != params:
            return None
        logger.info("Loaded cached structural analysis for %s", episode_url[:60])
        return cached.structural_metrics
    except Exception as exc:
        logger.warning("Failed to load analysis cache %s: %s", path.name, exc)
        return None


def save_cached_analysis(
    episode_url: str,
    metrics: StructuralMetrics,
    params: WhisperParams | None = None,
    episode_title: str = "",
    podcast_title: str = "",
    feed_url: str = "",
) -> Path:
    """Persist structural metrics to the cache."""
    params = params or current_whisper_params()
    cached = CachedAnalysis(
        episode_title=episode_title,
        podcast_title=podcast_title,
        feed_url=feed_url,
        episode_url=episode_url,
        whisper_params=params,
        structural_metrics=metrics,
        cached_at=datetime.now(timezone.utc).isoformat(),
    )
    path = _analysis_path(episode_url, params)
    path.write_text(cached.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Cached structural analysis -> %s", path.name)
    return path


# ------------------------------------------------------------------
#  LLM analysis cache
# ------------------------------------------------------------------

def load_cached_llm_analysis(
    episode_url: str,
    params: WhisperParams | None = None,
) -> LLMAnalysis | None:
    """Return cached LLMAnalysis if available, else None."""
    params = params or current_whisper_params()
    path = _llm_path(episode_url, params)
    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cached = CachedLLMAnalysis.model_validate(raw)
        logger.info("Loaded cached LLM analysis for '%s'", cached.episode_title)
        return cached.llm_analysis
    except Exception as exc:
        logger.warning("Failed to load LLM cache %s: %s", path.name, exc)
        return None


def save_cached_llm_analysis(
    episode_url: str,
    llm_analysis: LLMAnalysis,
    params: WhisperParams | None = None,
    episode_title: str = "",
    podcast_title: str = "",
    feed_url: str = "",
) -> Path:
    """Persist LLM analysis to the cache."""
    params = params or current_whisper_params()
    cached = CachedLLMAnalysis(
        episode_title=episode_title,
        podcast_title=podcast_title,
        feed_url=feed_url,
        episode_url=episode_url,
        llm_analysis=llm_analysis,
        cached_at=datetime.now(timezone.utc).isoformat(),
    )
    path = _llm_path(episode_url, params)
    path.write_text(cached.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Cached LLM analysis -> %s", path.name)
    return path


# ------------------------------------------------------------------
#  Full cache scan (for --rescore)
# ------------------------------------------------------------------

def scan_cached_analyses() -> dict[str, list[CachedAnalysis]]:
    """Scan data/cache/ for all an_*.json files and group by feed_url.

    Returns a dict mapping feed_url → list[CachedAnalysis].
    Episodes with no feed_url (legacy cache files) are grouped under
    an empty-string key.
    """
    results: dict[str, list[CachedAnalysis]] = {}
    cache_dir = config.CACHE_DIR
    if not cache_dir.exists():
        return results

    for path in sorted(cache_dir.glob("an_*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            cached = CachedAnalysis.model_validate(raw)
            key = cached.feed_url or ""
            results.setdefault(key, []).append(cached)
        except Exception as exc:
            logger.warning("Skipping invalid cache file %s: %s", path.name, exc)

    logger.info(
        "Cache scan: found %d analysis files across %d feed(s)",
        sum(len(v) for v in results.values()),
        len(results),
    )
    return results


def scan_cached_llm_analyses() -> dict[str, CachedLLMAnalysis]:
    """Scan for all llm_*.json files, returning a dict keyed by episode_url."""
    results: dict[str, CachedLLMAnalysis] = {}
    cache_dir = config.CACHE_DIR
    if not cache_dir.exists():
        return results

    for path in sorted(cache_dir.glob("llm_*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            cached = CachedLLMAnalysis.model_validate(raw)
            results[cached.episode_url] = cached
        except Exception as exc:
            logger.warning("Skipping invalid LLM cache file %s: %s", path.name, exc)

    return results
