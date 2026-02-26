"""Tests for src/scoring.py – normalisation, weights, CEFR mapping, and report."""

from __future__ import annotations

import pytest

import config
from src.models import (
    DifficultyScore,
    Episode,
    EpisodeAnalysis,
    LLMAnalysis,
    StructuralMetrics,
    Transcription,
)
from src.scoring import (
    CEFR_THRESHOLDS,
    NORM_RANGES,
    _cefr_from_score,
    _normalise,
    compute_podcast_score,
    format_report,
    score_episode,
)


# ===================================================================
# _normalise
# ===================================================================

class TestNormalise:
    """Tests for the _normalise helper."""

    def test_midpoint(self):
        assert _normalise(50.0, 0.0, 100.0) == pytest.approx(0.5)

    def test_at_lower_bound(self):
        assert _normalise(0.0, 0.0, 100.0) == pytest.approx(0.0)

    def test_at_upper_bound(self):
        assert _normalise(100.0, 0.0, 100.0) == pytest.approx(1.0)

    def test_below_lower_bound_clamps_to_zero(self):
        assert _normalise(-50.0, 0.0, 100.0) == pytest.approx(0.0)

    def test_above_upper_bound_clamps_to_one(self):
        assert _normalise(200.0, 0.0, 100.0) == pytest.approx(1.0)

    def test_negative_range(self):
        """Clarity uses a negative range like (-1.2, -0.1)."""
        assert _normalise(-0.65, -1.2, -0.1) == pytest.approx(0.5)

    def test_at_negative_lower_bound(self):
        assert _normalise(-1.2, -1.2, -0.1) == pytest.approx(0.0)

    def test_at_negative_upper_bound(self):
        assert _normalise(-0.1, -1.2, -0.1) == pytest.approx(1.0)

    def test_quarter_point(self):
        assert _normalise(25.0, 0.0, 100.0) == pytest.approx(0.25)

    def test_identical_bounds_returns_zero(self):
        """If lo == hi the division would be 0/0; should return 0.0 safely."""
        result = _normalise(5.0, 5.0, 5.0)
        assert result == 0.0


# ===================================================================
# _cefr_from_score
# ===================================================================

class TestCEFR:
    """Tests for CEFR mapping thresholds."""

    @pytest.mark.parametrize("score, expected", [
        (0.00, "A1"),
        (0.10, "A1"),
        (0.14, "A1"),
        (0.15, "A2"),
        (0.20, "A2"),
        (0.29, "A2"),
        (0.30, "B1"),
        (0.40, "B1"),
        (0.49, "B1"),
        (0.50, "B2"),
        (0.60, "B2"),
        (0.69, "B2"),
        (0.70, "C1"),
        (0.80, "C1"),
        (0.84, "C1"),
        (0.85, "C2"),
        (0.95, "C2"),
        (1.00, "C2"),
    ])
    def test_cefr_boundaries(self, score: float, expected: str):
        assert _cefr_from_score(score) == expected

    def test_thresholds_are_ordered(self):
        thresholds = [t for t, _ in CEFR_THRESHOLDS]
        assert thresholds == sorted(thresholds)


# ===================================================================
# score_episode
# ===================================================================

def _make_episode_analysis(
    *,
    wpm: float = 130.0,
    pct_outside_5k: float = 0.15,
    lexical_diversity: float = 0.30,
    avg_sentence_length: float = 12.0,
    avg_parse_depth: float = 4.0,
    avg_segment_confidence: float = -0.5,
    slang_score: float = 0.0,
    topic_complexity: float = 0.0,
) -> EpisodeAnalysis:
    """Build an EpisodeAnalysis with controllable metric values."""
    ep = Episode(title="Test Episode", url="https://example.com/ep.mp3")
    sm = StructuralMetrics(
        words_per_minute=wpm,
        total_words=1000,
        unique_words=300,
        lexical_diversity=lexical_diversity,
        avg_sentence_length=avg_sentence_length,
        avg_parse_depth=avg_parse_depth,
        pct_outside_top_5k=pct_outside_5k,
        avg_segment_confidence=avg_segment_confidence,
    )
    llm = LLMAnalysis(
        slang_score=slang_score,
        topic_complexity=topic_complexity,
    )
    return EpisodeAnalysis(
        episode=ep,
        structural_metrics=sm,
        llm_analysis=llm,
    )


class TestScoreEpisode:
    """Tests for per-episode component scoring."""

    def test_returns_all_components(self):
        ea = _make_episode_analysis()
        comps = score_episode(ea)
        expected_keys = set(config.SCORING_WEIGHTS.keys())
        assert set(comps.keys()) == expected_keys

    def test_all_values_between_0_and_1(self):
        ea = _make_episode_analysis()
        comps = score_episode(ea)
        for key, val in comps.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of range"

    def test_speech_rate_normalisation(self):
        lo, hi = NORM_RANGES["speech_rate"]
        # At minimum
        ea_slow = _make_episode_analysis(wpm=lo)
        assert score_episode(ea_slow)["speech_rate"] == pytest.approx(0.0)
        # At maximum
        ea_fast = _make_episode_analysis(wpm=hi)
        assert score_episode(ea_fast)["speech_rate"] == pytest.approx(1.0)

    def test_vocabulary_normalisation(self):
        lo, hi = NORM_RANGES["vocabulary_level"]
        ea_easy = _make_episode_analysis(pct_outside_5k=lo)
        assert score_episode(ea_easy)["vocabulary_level"] == pytest.approx(0.0)
        ea_hard = _make_episode_analysis(pct_outside_5k=hi)
        assert score_episode(ea_hard)["vocabulary_level"] == pytest.approx(1.0)

    def test_clarity_inversion(self):
        """Worse clarity (more negative) should yield a HIGHER difficulty."""
        ea_clear = _make_episode_analysis(avg_segment_confidence=-0.1)
        ea_muddy = _make_episode_analysis(avg_segment_confidence=-1.2)
        comps_clear = score_episode(ea_clear)
        comps_muddy = score_episode(ea_muddy)
        assert comps_muddy["clarity"] > comps_clear["clarity"]

    def test_clarity_best_case(self):
        """Best clarity (-0.1) → clarity difficulty ≈ 0."""
        ea = _make_episode_analysis(avg_segment_confidence=-0.1)
        assert score_episode(ea)["clarity"] == pytest.approx(0.0, abs=0.01)

    def test_clarity_worst_case(self):
        """Worst clarity (-1.2) → clarity difficulty ≈ 1."""
        ea = _make_episode_analysis(avg_segment_confidence=-1.2)
        assert score_episode(ea)["clarity"] == pytest.approx(1.0, abs=0.01)

    def test_empty_structural_metrics_gives_zeroes(self):
        """If no metrics are provided, defaults (0s) should normalise to 0."""
        ep = Episode(title="Empty", url="https://example.com/ep.mp3")
        ea = EpisodeAnalysis(episode=ep)
        comps = score_episode(ea)
        # With all-zero metrics most components should be at or near 0
        # (except clarity which inverts a default of -0.5)
        assert comps["speech_rate"] == pytest.approx(0.0)
        assert comps["vocabulary_level"] == pytest.approx(0.0)


# ===================================================================
# compute_podcast_score
# ===================================================================

class TestComputePodcastScore:
    """Tests for the full podcast scoring pipeline."""

    def test_empty_episodes_returns_zero(self):
        score = compute_podcast_score("Test", "http://test.com", [])
        assert score.overall_score == 0.0
        assert score.cefr_estimate == ""

    def test_single_episode_score(self):
        ea = _make_episode_analysis(
            wpm=130.0,
            pct_outside_5k=0.20,
            lexical_diversity=0.30,
            avg_sentence_length=12.0,
            avg_parse_depth=4.0,
        )
        score = compute_podcast_score("Test Pod", "http://test.com", [ea])
        assert 0.0 < score.overall_score < 1.0
        assert score.cefr_estimate in {"A1", "A2", "B1", "B2", "C1", "C2"}
        assert score.episodes_analyzed == 1

    def test_multiple_episodes_averaged(self):
        """Score from two episodes should be between the individual scores."""
        ea_easy = _make_episode_analysis(wpm=50.0, pct_outside_5k=0.0, avg_parse_depth=1.0)
        ea_hard = _make_episode_analysis(wpm=200.0, pct_outside_5k=0.40, avg_parse_depth=7.0)
        score_easy = compute_podcast_score("Easy", "http://e.com", [ea_easy])
        score_hard = compute_podcast_score("Hard", "http://h.com", [ea_hard])
        score_both = compute_podcast_score("Both", "http://b.com", [ea_easy, ea_hard])
        assert score_easy.overall_score < score_both.overall_score < score_hard.overall_score

    def test_weights_sum_to_one(self):
        total = sum(config.SCORING_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_overall_score_clamped_0_1(self):
        """Even with extreme values, overall must be in [0, 1]."""
        ea_extreme = _make_episode_analysis(
            wpm=500.0,
            pct_outside_5k=1.0,
            lexical_diversity=1.0,
            avg_sentence_length=50.0,
            avg_parse_depth=20.0,
            avg_segment_confidence=-5.0,
            slang_score=1.0,
            topic_complexity=1.0,
        )
        score = compute_podcast_score("Extreme", "http://x.com", [ea_extreme])
        assert 0.0 <= score.overall_score <= 1.0


# ===================================================================
# Weight redistribution (no LLM)
# ===================================================================

class TestWeightRedistribution:
    """When LLM is disabled, LLM weights should redistribute proportionally."""

    def test_no_llm_still_full_range(self):
        """Without LLM, an extreme-hard episode should still score near 1.0."""
        ea = _make_episode_analysis(
            wpm=220.0,
            pct_outside_5k=0.45,
            lexical_diversity=0.65,
            avg_sentence_length=30.0,
            avg_parse_depth=10.0,
            avg_segment_confidence=-1.2,
        )
        # Remove LLM analysis
        ea.llm_analysis = None
        score = compute_podcast_score("Hard", "http://h.com", [ea])
        assert score.overall_score >= 0.85, (
            f"Expected near 1.0 without LLM, got {score.overall_score}"
        )

    def test_no_llm_easy_still_low(self):
        """Without LLM, an easy episode should still score near 0."""
        ea = _make_episode_analysis(
            wpm=40.0,
            pct_outside_5k=0.0,
            lexical_diversity=0.05,
            avg_sentence_length=2.0,
            avg_parse_depth=1.0,
            avg_segment_confidence=-0.1,
        )
        ea.llm_analysis = None
        score = compute_podcast_score("Easy", "http://e.com", [ea])
        assert score.overall_score <= 0.15, (
            f"Expected near 0 without LLM, got {score.overall_score}"
        )

    def test_redistributed_weights_sum_to_one(self):
        """After redistribution, weights should still sum to 1.0."""
        weights = dict(config.SCORING_WEIGHTS)
        llm_keys = {"slang_score", "topic_complexity"}
        llm_weight = sum(weights[k] for k in llm_keys)
        structural_keys = [k for k in weights if k not in llm_keys]
        structural_total = sum(weights[k] for k in structural_keys)
        for k in structural_keys:
            weights[k] = weights[k] / structural_total * (structural_total + llm_weight)
        for k in llm_keys:
            weights[k] = 0.0
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_no_llm_excludes_slang_topic_from_components(self):
        """Without LLM, component_scores should not include slang/topic."""
        ea = _make_episode_analysis()
        ea.llm_analysis = None
        score = compute_podcast_score("Test", "http://t.com", [ea])
        assert "slang_score" not in score.component_scores
        assert "topic_complexity" not in score.component_scores


# ===================================================================
# format_report
# ===================================================================

class TestFormatReport:
    """Tests for the text report formatter."""

    def test_contains_title(self):
        score = DifficultyScore(
            podcast_title="Mi Podcast",
            feed_url="http://example.com",
            overall_score=0.42,
            cefr_estimate="B1",
            episodes_analyzed=3,
        )
        report = format_report(score)
        assert "Mi Podcast" in report

    def test_contains_cefr(self):
        score = DifficultyScore(
            podcast_title="Test",
            feed_url="http://example.com",
            overall_score=0.42,
            cefr_estimate="B1",
        )
        report = format_report(score)
        assert "B1" in report

    def test_no_llm_mode_label(self):
        score = DifficultyScore(
            podcast_title="Test",
            feed_url="http://example.com",
        )
        report = format_report(score)
        assert "Structural only" in report

    def test_llm_mode_label(self):
        ep = Episode(title="Ep", url="http://example.com/ep.mp3")
        ea = EpisodeAnalysis(
            episode=ep,
            llm_analysis=LLMAnalysis(slang_score=0.3),
        )
        score = DifficultyScore(
            podcast_title="Test",
            feed_url="http://example.com",
            episode_results=[ea],
        )
        report = format_report(score)
        assert "Structural + LLM" in report


# ===================================================================
# Outlier trimming
# ===================================================================

def _make_ea(title: str, wpm: float, outside_5k: float = 0.1) -> EpisodeAnalysis:
    """Helper to build an EpisodeAnalysis with key metrics."""
    return EpisodeAnalysis(
        episode=Episode(title=title, url=f"http://example.com/{title}.mp3"),
        structural_metrics=StructuralMetrics(
            words_per_minute=wpm,
            total_words=500,
            unique_words=200,
            lexical_diversity=0.4,
            avg_sentence_length=12.0,
            pct_outside_top_5k=outside_5k,
            avg_parse_depth=3.5,
            avg_segment_confidence=-0.4,
        ),
    )


class TestOutlierTrimming:
    """Outlier trimming: drop the highest-scoring episode when we have enough."""

    def test_trims_highest_scorer(self, monkeypatch):
        """The episode with the hardest metrics should be excluded."""
        monkeypatch.setattr("config.OUTLIER_TRIM_COUNT", 1)
        monkeypatch.setattr("config.OUTLIER_TRIM_MIN_EPISODES", 4)
        episodes = [
            _make_ea("easy1", wpm=60, outside_5k=0.05),
            _make_ea("easy2", wpm=65, outside_5k=0.06),
            _make_ea("easy3", wpm=70, outside_5k=0.07),
            _make_ea("easy4", wpm=55, outside_5k=0.04),
            _make_ea("hard_outlier", wpm=200, outside_5k=0.40),
        ]
        result = compute_podcast_score("Test", "http://test.com", episodes)
        assert "hard_outlier" in result.trimmed_episodes
        assert result.episodes_analyzed == 4

    def test_no_trim_below_minimum(self, monkeypatch):
        """With fewer episodes than the minimum, no trimming should occur."""
        monkeypatch.setattr("config.OUTLIER_TRIM_COUNT", 1)
        monkeypatch.setattr("config.OUTLIER_TRIM_MIN_EPISODES", 4)
        episodes = [
            _make_ea("ep1", wpm=60),
            _make_ea("ep2", wpm=200),  # outlier, but only 2 eps
        ]
        result = compute_podcast_score("Test", "http://test.com", episodes)
        assert result.trimmed_episodes == []
        assert result.episodes_analyzed == 2

    def test_trim_disabled(self, monkeypatch):
        """Setting OUTLIER_TRIM_COUNT=0 disables trimming entirely."""
        monkeypatch.setattr("config.OUTLIER_TRIM_COUNT", 0)
        episodes = [
            _make_ea("ep1", wpm=60),
            _make_ea("ep2", wpm=200),
            _make_ea("ep3", wpm=70),
            _make_ea("ep4", wpm=65),
            _make_ea("ep5", wpm=55),
        ]
        result = compute_podcast_score("Test", "http://test.com", episodes)
        assert result.trimmed_episodes == []
        assert result.episodes_analyzed == 5

    def test_trimming_lowers_score(self, monkeypatch):
        """Removing the hardest outlier should reduce the overall score."""
        monkeypatch.setattr("config.OUTLIER_TRIM_COUNT", 1)
        monkeypatch.setattr("config.OUTLIER_TRIM_MIN_EPISODES", 4)
        episodes = [
            _make_ea("ep1", wpm=60, outside_5k=0.05),
            _make_ea("ep2", wpm=65, outside_5k=0.06),
            _make_ea("ep3", wpm=70, outside_5k=0.07),
            _make_ea("ep4", wpm=55, outside_5k=0.04),
            _make_ea("hard", wpm=200, outside_5k=0.40),
        ]
        trimmed = compute_podcast_score("Test", "http://test.com", episodes)

        monkeypatch.setattr("config.OUTLIER_TRIM_COUNT", 0)
        untrimmed = compute_podcast_score("Test", "http://test.com", episodes)

        assert trimmed.overall_score < untrimmed.overall_score

    def test_report_shows_trimmed_label(self, monkeypatch):
        """The report should show [TRIMMED] next to excluded episodes."""
        monkeypatch.setattr("config.OUTLIER_TRIM_COUNT", 1)
        monkeypatch.setattr("config.OUTLIER_TRIM_MIN_EPISODES", 4)
        episodes = [
            _make_ea("ep1", wpm=60),
            _make_ea("ep2", wpm=65),
            _make_ea("ep3", wpm=70),
            _make_ea("ep4", wpm=55),
            _make_ea("hard_one", wpm=200, outside_5k=0.40),
        ]
        result = compute_podcast_score("Test", "http://test.com", episodes)
        report = format_report(result)
        assert "[TRIMMED]" in report
        assert "outlier trimmed" in report

    def test_all_episodes_in_results(self, monkeypatch):
        """All episodes (including trimmed) should appear in episode_results."""
        monkeypatch.setattr("config.OUTLIER_TRIM_COUNT", 1)
        monkeypatch.setattr("config.OUTLIER_TRIM_MIN_EPISODES", 4)
        episodes = [
            _make_ea("ep1", wpm=60),
            _make_ea("ep2", wpm=65),
            _make_ea("ep3", wpm=70),
            _make_ea("ep4", wpm=55),
            _make_ea("hard_one", wpm=200, outside_5k=0.40),
        ]
        result = compute_podcast_score("Test", "http://test.com", episodes)
        titles = [ea.episode.title for ea in result.episode_results]
        assert "hard_one" in titles  # still in the full list
        assert len(result.episode_results) == 5
        assert result.episodes_analyzed == 4  # but not counted
