"""Tests for src/models.py – Pydantic model validation, defaults, and forward refs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import (
    CachedAnalysis,
    CachedLLMAnalysis,
    CachedTranscription,
    DifficultyScore,
    Episode,
    EpisodeAnalysis,
    LLMAnalysis,
    PodcastFeed,
    StructuralMetrics,
    Transcription,
    TranscriptionSegment,
    WhisperParams,
    WordTimestamp,
)


# ===================================================================
# Episode
# ===================================================================

class TestEpisode:
    def test_required_fields(self):
        ep = Episode(title="Ep 1", url="https://example.com/ep.mp3")
        assert ep.title == "Ep 1"
        assert ep.audio_path is None
        assert ep.duration_seconds is None

    def test_optional_fields(self):
        ep = Episode(
            title="Ep",
            url="https://example.com/ep.mp3",
            published="2025-01-01",
            duration_seconds=300.0,
            audio_path="/tmp/ep.mp3",
        )
        assert ep.published == "2025-01-01"
        assert ep.duration_seconds == 300.0

    def test_missing_required_raises(self):
        with pytest.raises(ValidationError):
            Episode(title="No URL")  # type: ignore


# ===================================================================
# PodcastFeed
# ===================================================================

class TestPodcastFeed:
    def test_defaults(self):
        f = PodcastFeed(title="Pod", feed_url="http://example.com/rss")
        assert f.episodes == []
        assert f.language is None

    def test_with_episodes(self):
        ep = Episode(title="Ep", url="http://e.com/1.mp3")
        f = PodcastFeed(title="Pod", feed_url="http://f.com", episodes=[ep])
        assert len(f.episodes) == 1


# ===================================================================
# TranscriptionSegment / WordTimestamp (forward refs)
# ===================================================================

class TestTranscriptionSegment:
    def test_basic(self):
        seg = TranscriptionSegment(text="hola", start=0.0, end=1.0)
        assert seg.words == []
        assert seg.avg_log_prob is None

    def test_with_words(self):
        """Forward reference to WordTimestamp should be resolved."""
        w = WordTimestamp(word="hola", start=0.0, end=0.5, probability=0.99)
        seg = TranscriptionSegment(text="hola", start=0.0, end=0.5, words=[w])
        assert len(seg.words) == 1
        assert seg.words[0].word == "hola"

    def test_serialisation_round_trip(self):
        w = WordTimestamp(word="mundo", start=0.1, end=0.5, probability=0.88)
        seg = TranscriptionSegment(
            text="mundo", start=0.1, end=0.5, words=[w], avg_log_prob=-0.3
        )
        data = seg.model_dump()
        restored = TranscriptionSegment.model_validate(data)
        assert restored.words[0].probability == pytest.approx(0.88)
        assert restored.avg_log_prob == pytest.approx(-0.3)


# ===================================================================
# Transcription
# ===================================================================

class TestTranscription:
    def test_defaults(self):
        t = Transcription(episode_title="Ep", full_text="Hola")
        assert t.language == "es"
        assert t.duration_seconds == 0.0
        assert t.segments == []

    def test_with_segments(self):
        seg = TranscriptionSegment(text="Hola", start=0.0, end=1.0)
        t = Transcription(
            episode_title="Ep",
            full_text="Hola",
            segments=[seg],
            duration_seconds=1.0,
        )
        assert len(t.segments) == 1


# ===================================================================
# WhisperParams
# ===================================================================

class TestWhisperParams:
    def test_defaults(self):
        p = WhisperParams()
        assert p.model_size == "small"
        assert p.beam_size == 1
        assert p.language == "es"
        assert p.skip_intro_seconds == 45
        assert p.max_transcribe_minutes == 10

    def test_equality(self):
        p1 = WhisperParams(model_size="small", beam_size=1)
        p2 = WhisperParams(model_size="small", beam_size=1)
        assert p1 == p2

    def test_inequality(self):
        p1 = WhisperParams(model_size="small")
        p2 = WhisperParams(model_size="medium")
        assert p1 != p2


# ===================================================================
# StructuralMetrics
# ===================================================================

class TestStructuralMetrics:
    def test_all_defaults_zero(self):
        sm = StructuralMetrics()
        assert sm.words_per_minute == 0.0
        assert sm.total_words == 0
        assert sm.lexical_diversity == 0.0
        assert sm.avg_parse_depth == 0.0
        assert sm.subjunctive_ratio == 0.0

    def test_custom_values(self):
        sm = StructuralMetrics(
            words_per_minute=145.0,
            pct_outside_top_5k=0.18,
        )
        assert sm.words_per_minute == 145.0
        assert sm.pct_outside_top_5k == 0.18


# ===================================================================
# LLMAnalysis
# ===================================================================

class TestLLMAnalysis:
    def test_defaults(self):
        la = LLMAnalysis()
        assert la.slang_score == 0.0
        assert la.slang_examples == []
        assert la.estimated_cefr == ""

    def test_with_values(self):
        la = LLMAnalysis(
            slang_score=0.7,
            slang_examples=["guay", "mola"],
            estimated_cefr="B2",
        )
        assert len(la.slang_examples) == 2


# ===================================================================
# EpisodeAnalysis
# ===================================================================

class TestEpisodeAnalysis:
    def test_optional_fields_none(self):
        ep = Episode(title="Ep", url="http://e.com/1.mp3")
        ea = EpisodeAnalysis(episode=ep)
        assert ea.transcription is None
        assert ea.structural_metrics is None
        assert ea.llm_analysis is None


# ===================================================================
# DifficultyScore
# ===================================================================

class TestDifficultyScore:
    def test_defaults(self):
        ds = DifficultyScore(podcast_title="Pod", feed_url="http://f.com")
        assert ds.overall_score == 0.0
        assert ds.cefr_estimate == ""
        assert ds.component_scores == {}
        assert ds.episodes_analyzed == 0
        assert ds.episode_results == []

    def test_serialisation(self):
        ds = DifficultyScore(
            podcast_title="Test",
            feed_url="http://test.com",
            overall_score=0.42,
            cefr_estimate="B1",
            component_scores={"speech_rate": 0.5, "vocabulary_level": 0.3},
            episodes_analyzed=2,
        )
        data = ds.model_dump()
        restored = DifficultyScore.model_validate(data)
        assert restored.overall_score == pytest.approx(0.42)
        assert restored.component_scores["speech_rate"] == pytest.approx(0.5)


# ===================================================================
# CachedTranscription / CachedAnalysis
# ===================================================================

class TestCachedModels:
    def test_cached_transcription(self):
        ct = CachedTranscription(
            episode_title="Ep Title",
            whisper_params=WhisperParams(),
            episode_url="http://e.com/ep.mp3",
            transcription=Transcription(episode_title="Ep Title", full_text="Hola"),
            cached_at="2025-01-01T00:00:00Z",
        )
        assert ct.episode_title == "Ep Title"
        assert ct.whisper_params.model_size == "small"

    def test_cached_analysis(self):
        ca = CachedAnalysis(
            episode_title="Ep",
            episode_url="http://e.com/ep.mp3",
            whisper_params=WhisperParams(),
            structural_metrics=StructuralMetrics(words_per_minute=120.0),
        )
        assert ca.structural_metrics.words_per_minute == 120.0

    def test_cached_analysis_feed_metadata(self):
        ca = CachedAnalysis(
            episode_title="Ep",
            podcast_title="My Podcast",
            feed_url="http://e.com/feed.xml",
            episode_url="http://e.com/ep.mp3",
            whisper_params=WhisperParams(),
            structural_metrics=StructuralMetrics(),
        )
        assert ca.podcast_title == "My Podcast"
        assert ca.feed_url == "http://e.com/feed.xml"

    def test_cached_llm_analysis(self):
        cl = CachedLLMAnalysis(
            episode_title="Ep",
            podcast_title="Pod",
            feed_url="http://e.com/feed.xml",
            episode_url="http://e.com/ep.mp3",
            llm_analysis=LLMAnalysis(slang_score=0.4, estimated_cefr="B1"),
            cached_at="2025-01-01T00:00:00Z",
        )
        assert cl.llm_analysis.slang_score == 0.4
        assert cl.podcast_title == "Pod"
