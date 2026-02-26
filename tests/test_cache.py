"""Tests for src/cache.py – cache key generation, save/load round-trip,
and parameter-mismatch invalidation."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.cache import (
    _cache_key,
    current_whisper_params,
    load_cached_analysis,
    load_cached_llm_analysis,
    load_cached_transcription,
    save_cached_analysis,
    save_cached_llm_analysis,
    save_cached_transcription,
    scan_cached_analyses,
    scan_cached_llm_analyses,
)
from src.models import (
    LLMAnalysis,
    StructuralMetrics,
    Transcription,
    TranscriptionSegment,
    WhisperParams,
)


# ===================================================================
# Cache key
# ===================================================================

class TestCacheKey:
    """Deterministic hash of episode URL + whisper params."""

    def test_same_inputs_same_key(self):
        params = WhisperParams()
        k1 = _cache_key("https://example.com/ep.mp3", params)
        k2 = _cache_key("https://example.com/ep.mp3", params)
        assert k1 == k2

    def test_different_url_different_key(self):
        params = WhisperParams()
        k1 = _cache_key("https://example.com/ep1.mp3", params)
        k2 = _cache_key("https://example.com/ep2.mp3", params)
        assert k1 != k2

    def test_different_model_different_key(self):
        url = "https://example.com/ep.mp3"
        k1 = _cache_key(url, WhisperParams(model_size="small"))
        k2 = _cache_key(url, WhisperParams(model_size="medium"))
        assert k1 != k2

    def test_different_beam_size_different_key(self):
        url = "https://example.com/ep.mp3"
        k1 = _cache_key(url, WhisperParams(beam_size=1))
        k2 = _cache_key(url, WhisperParams(beam_size=5))
        assert k1 != k2

    def test_different_skip_different_key(self):
        url = "https://example.com/ep.mp3"
        k1 = _cache_key(url, WhisperParams(skip_intro_seconds=30))
        k2 = _cache_key(url, WhisperParams(skip_intro_seconds=60))
        assert k1 != k2

    def test_different_max_minutes_different_key(self):
        url = "https://example.com/ep.mp3"
        k1 = _cache_key(url, WhisperParams(max_transcribe_minutes=10))
        k2 = _cache_key(url, WhisperParams(max_transcribe_minutes=0))
        assert k1 != k2

    def test_key_is_hex_string(self):
        key = _cache_key("https://example.com/ep.mp3", WhisperParams())
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_key_deterministic_across_param_order(self):
        """sort_keys=True ensures dict ordering doesn't affect the hash."""
        p1 = WhisperParams(model_size="small", beam_size=1, language="es")
        p2 = WhisperParams(language="es", beam_size=1, model_size="small")
        url = "https://example.com/ep.mp3"
        assert _cache_key(url, p1) == _cache_key(url, p2)


# ===================================================================
# Transcription cache round-trip
# ===================================================================

class TestTranscriptionCache:
    """Save and reload transcriptions."""

    def test_round_trip(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        params = WhisperParams(model_size="small", beam_size=1)
        url = "https://example.com/ep.mp3"
        transcript = Transcription(
            episode_title="Test Episode",
            full_text="Hola mundo, esto es una prueba.",
            segments=[
                TranscriptionSegment(text="Hola mundo", start=0.0, end=1.5, avg_log_prob=-0.3),
                TranscriptionSegment(text="esto es una prueba", start=1.5, end=3.0, avg_log_prob=-0.4),
            ],
            duration_seconds=3.0,
            language="es",
            language_probability=0.99,
        )

        save_cached_transcription(url, transcript, params)
        loaded = load_cached_transcription(url, params)

        assert loaded is not None
        assert loaded.episode_title == "Test Episode"
        assert loaded.full_text == transcript.full_text
        assert len(loaded.segments) == 2
        assert loaded.segments[0].avg_log_prob == pytest.approx(-0.3)
        assert loaded.duration_seconds == 3.0
        assert loaded.language_probability == pytest.approx(0.99)

    def test_cache_miss_returns_none(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        result = load_cached_transcription(
            "https://example.com/nonexistent.mp3",
            WhisperParams(),
        )
        assert result is None

    def test_param_mismatch_returns_none(self, tmp_path: Path, monkeypatch):
        """Saved with small model → looking up with medium should miss."""
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        url = "https://example.com/ep.mp3"
        transcript = Transcription(
            episode_title="Test",
            full_text="Hola",
        )
        save_cached_transcription(url, transcript, WhisperParams(model_size="small"))
        result = load_cached_transcription(url, WhisperParams(model_size="medium"))
        assert result is None

    def test_cache_file_is_valid_json(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        url = "https://example.com/ep.mp3"
        transcript = Transcription(episode_title="JSON Test", full_text="Prueba")
        path = save_cached_transcription(url, transcript, WhisperParams())
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["episode_title"] == "JSON Test"
        assert "whisper_params" in data
        assert "cached_at" in data

    def test_word_timestamps_preserved(self, tmp_path: Path, monkeypatch):
        from src.models import WordTimestamp
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        url = "https://example.com/ep.mp3"
        transcript = Transcription(
            episode_title="Words",
            full_text="hola mundo",
            segments=[
                TranscriptionSegment(
                    text="hola mundo",
                    start=0.0,
                    end=1.0,
                    words=[
                        WordTimestamp(word="hola", start=0.0, end=0.4, probability=0.95),
                        WordTimestamp(word="mundo", start=0.5, end=1.0, probability=0.88),
                    ],
                ),
            ],
        )
        save_cached_transcription(url, transcript, WhisperParams())
        loaded = load_cached_transcription(url, WhisperParams())
        assert loaded is not None
        assert len(loaded.segments[0].words) == 2
        assert loaded.segments[0].words[0].word == "hola"
        assert loaded.segments[0].words[1].probability == pytest.approx(0.88)


# ===================================================================
# Analysis cache round-trip
# ===================================================================

class TestAnalysisCache:
    """Save and reload structural metrics."""

    def test_round_trip(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        params = WhisperParams()
        url = "https://example.com/ep.mp3"
        metrics = StructuralMetrics(
            words_per_minute=145.3,
            total_words=1200,
            unique_words=420,
            lexical_diversity=0.35,
            avg_sentence_length=14.2,
            avg_parse_depth=4.1,
            pct_outside_top_5k=0.18,
            subjunctive_ratio=0.05,
            avg_segment_confidence=-0.45,
        )
        save_cached_analysis(url, metrics, params, episode_title="Metrics EP")
        loaded = load_cached_analysis(url, params)

        assert loaded is not None
        assert loaded.words_per_minute == pytest.approx(145.3)
        assert loaded.total_words == 1200
        assert loaded.lexical_diversity == pytest.approx(0.35)
        assert loaded.avg_parse_depth == pytest.approx(4.1)

    def test_cache_miss(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        result = load_cached_analysis("https://example.com/x.mp3", WhisperParams())
        assert result is None

    def test_param_mismatch(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        url = "https://example.com/ep.mp3"
        metrics = StructuralMetrics(words_per_minute=100.0)
        save_cached_analysis(url, metrics, WhisperParams(beam_size=1))
        result = load_cached_analysis(url, WhisperParams(beam_size=5))
        assert result is None

    def test_episode_title_in_json(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        url = "https://example.com/ep.mp3"
        metrics = StructuralMetrics()
        path = save_cached_analysis(url, metrics, WhisperParams(), episode_title="Mi Episodio")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["episode_title"] == "Mi Episodio"


# ===================================================================
# current_whisper_params
# ===================================================================

class TestCurrentWhisperParams:
    """Ensure params are snapshot from config."""

    def test_reflects_config(self, monkeypatch):
        monkeypatch.setattr("config.WHISPER_MODEL_SIZE", "large-v3")
        monkeypatch.setattr("config.WHISPER_BEAM_SIZE", 5)
        monkeypatch.setattr("config.SKIP_INTRO_SECONDS", 30)
        monkeypatch.setattr("config.MAX_TRANSCRIBE_MINUTES", 0)
        params = current_whisper_params()
        assert params.model_size == "large-v3"
        assert params.beam_size == 5
        assert params.skip_intro_seconds == 30
        assert params.max_transcribe_minutes == 0


# ===================================================================
# LLM analysis cache
# ===================================================================

class TestLLMAnalysisCache:
    """Save and reload LLM qualitative analysis."""

    def test_round_trip(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        params = WhisperParams()
        url = "https://example.com/ep.mp3"
        llm = LLMAnalysis(
            slang_score=0.3,
            slang_examples=["mola", "tío"],
            topic_complexity=0.5,
            estimated_cefr="B1",
            idiom_count=4,
            formality_score=0.6,
        )
        save_cached_llm_analysis(url, llm, params, episode_title="LLM Test")
        loaded = load_cached_llm_analysis(url, params)
        assert loaded is not None
        assert loaded.slang_score == pytest.approx(0.3)
        assert loaded.slang_examples == ["mola", "tío"]
        assert loaded.estimated_cefr == "B1"

    def test_cache_miss(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        result = load_cached_llm_analysis("https://example.com/x.mp3", WhisperParams())
        assert result is None

    def test_metadata_in_json(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        url = "https://example.com/ep.mp3"
        llm = LLMAnalysis(slang_score=0.1)
        path = save_cached_llm_analysis(
            url, llm, WhisperParams(),
            episode_title="EP1",
            podcast_title="My Podcast",
            feed_url="https://example.com/feed.xml",
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["episode_title"] == "EP1"
        assert data["podcast_title"] == "My Podcast"
        assert data["feed_url"] == "https://example.com/feed.xml"


# ===================================================================
# Cache scan for rescore
# ===================================================================

class TestScanCachedAnalyses:
    """Test the scan_cached_analyses() function used by --rescore."""

    def test_empty_cache(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        result = scan_cached_analyses()
        assert result == {}

    def test_groups_by_feed_url(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        params = WhisperParams()

        # Two episodes from feed A
        save_cached_analysis(
            "https://example.com/ep1.mp3",
            StructuralMetrics(words_per_minute=120.0),
            params,
            episode_title="Ep1",
            podcast_title="Podcast A",
            feed_url="https://example.com/feedA.xml",
        )
        save_cached_analysis(
            "https://example.com/ep2.mp3",
            StructuralMetrics(words_per_minute=140.0),
            params,
            episode_title="Ep2",
            podcast_title="Podcast A",
            feed_url="https://example.com/feedA.xml",
        )
        # One episode from feed B
        save_cached_analysis(
            "https://example.com/ep3.mp3",
            StructuralMetrics(words_per_minute=180.0),
            params,
            episode_title="Ep3",
            podcast_title="Podcast B",
            feed_url="https://example.com/feedB.xml",
        )

        result = scan_cached_analyses()
        assert len(result) == 2
        assert len(result["https://example.com/feedA.xml"]) == 2
        assert len(result["https://example.com/feedB.xml"]) == 1

    def test_legacy_files_grouped_under_empty_key(self, tmp_path: Path, monkeypatch):
        """Cache files without feed_url (from older runs) go into the '' group."""
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        params = WhisperParams()
        # Save without feed_url (legacy)
        save_cached_analysis(
            "https://example.com/old.mp3",
            StructuralMetrics(),
            params,
        )
        result = scan_cached_analyses()
        assert "" in result
        assert len(result[""]) == 1

    def test_invalid_files_skipped(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        # Write an invalid JSON file with the right naming pattern
        (tmp_path / "an_badfile1234abcd.json").write_text("not json", encoding="utf-8")
        result = scan_cached_analyses()
        assert result == {}

    def test_preserves_structural_metrics(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        params = WhisperParams()
        save_cached_analysis(
            "https://example.com/ep.mp3",
            StructuralMetrics(
                words_per_minute=155.0,
                pct_outside_top_5k=0.22,
                avg_parse_depth=4.5,
            ),
            params,
            feed_url="https://example.com/feed.xml",
        )
        result = scan_cached_analyses()
        items = result["https://example.com/feed.xml"]
        assert len(items) == 1
        assert items[0].structural_metrics.words_per_minute == pytest.approx(155.0)
        assert items[0].structural_metrics.pct_outside_top_5k == pytest.approx(0.22)


class TestScanCachedLLMAnalyses:
    """Test scan_cached_llm_analyses() used by --rescore."""

    def test_empty_cache(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        result = scan_cached_llm_analyses()
        assert result == {}

    def test_keyed_by_episode_url(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        url = "https://example.com/ep.mp3"
        save_cached_llm_analysis(
            url,
            LLMAnalysis(slang_score=0.4, topic_complexity=0.6),
            WhisperParams(),
        )
        result = scan_cached_llm_analyses()
        assert url in result
        assert result[url].llm_analysis.slang_score == pytest.approx(0.4)


# ===================================================================
# Feed metadata in analysis cache
# ===================================================================

class TestAnalysisCacheFeedMetadata:
    """Verify feed_url and podcast_title are stored and retrievable."""

    def test_feed_metadata_in_json(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        url = "https://example.com/ep.mp3"
        path = save_cached_analysis(
            url,
            StructuralMetrics(),
            WhisperParams(),
            episode_title="Ep Title",
            podcast_title="Podcast Title",
            feed_url="https://example.com/feed.xml",
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["podcast_title"] == "Podcast Title"
        assert data["feed_url"] == "https://example.com/feed.xml"
        assert data["episode_title"] == "Ep Title"
