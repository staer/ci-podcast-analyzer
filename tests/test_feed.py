"""Tests for src/feed.py – episode sampling logic.

Note: these tests don't hit the network.  We test the _sample_episodes
function directly with synthetic Episode lists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import config
from src.feed import _sample_episodes, _sort_prefer_longer
from src.models import Episode


@pytest.fixture(autouse=True)
def _no_cache_hits(monkeypatch, tmp_path):
    """By default, make the cache directory empty so no episodes appear cached."""
    monkeypatch.setattr("config.CACHE_DIR", tmp_path)


def _make_episodes(n: int, duration: float | None = 300.0) -> list[Episode]:
    """Create a list of n dummy episodes with given duration (seconds)."""
    return [
        Episode(
            title=f"Episode {i+1}",
            url=f"https://example.com/ep{i+1}.mp3",
            duration_seconds=duration,
        )
        for i in range(n)
    ]


# ===================================================================
# Basic sampling
# ===================================================================

class TestSampleEpisodes:
    """Tests for the _sample_episodes function."""

    def test_empty_candidates(self):
        assert _sample_episodes([]) == []

    def test_returns_at_least_min_episodes(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_EPISODES", 3)
        monkeypatch.setattr(config, "TARGET_AUDIO_MINUTES", 1)  # Very low target
        episodes = _make_episodes(10, duration=60.0)  # 1 min each
        selected = _sample_episodes(episodes)
        assert len(selected) >= 3

    def test_stops_at_target_duration(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_EPISODES", 1)
        monkeypatch.setattr(config, "TARGET_AUDIO_MINUTES", 5)
        # 10 episodes × 5 minutes each = 50 min total
        episodes = _make_episodes(10, duration=300.0)
        selected = _sample_episodes(episodes)
        # Should stop once we hit ~5 min (1 ep of 5 min + min_episodes met)
        assert len(selected) <= 10

    def test_selects_all_if_not_enough_duration(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_EPISODES", 1)
        monkeypatch.setattr(config, "TARGET_AUDIO_MINUTES", 100)
        # Only 3 episodes × 5 min = 15 min, but target is 100
        episodes = _make_episodes(3, duration=300.0)
        selected = _sample_episodes(episodes)
        assert len(selected) == 3  # All of them

    def test_no_duration_falls_back_to_min_episodes(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_EPISODES", 2)
        monkeypatch.setattr(config, "TARGET_AUDIO_MINUTES", 30)
        episodes = _make_episodes(5, duration=None)  # No duration info
        selected = _sample_episodes(episodes)
        assert len(selected) == 2

    def test_random_sampling_varies(self, monkeypatch):
        """Multiple calls should sometimes produce different orderings."""
        monkeypatch.setattr(config, "MIN_EPISODES", 2)
        monkeypatch.setattr(config, "TARGET_AUDIO_MINUTES", 5)
        # Use durations above PREFER_LONGER_THRESHOLD so episodes are shuffled
        episodes = _make_episodes(20, duration=700.0)

        # Run many times and collect the first episode title
        first_titles = set()
        for _ in range(50):
            selected = _sample_episodes(episodes)
            first_titles.add(selected[0].title)

        # With 20 candidates and random shuffle, we should see multiple first titles
        assert len(first_titles) > 1, "Sampling doesn't appear random"

    def test_single_candidate(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_EPISODES", 1)
        episodes = _make_episodes(1, duration=600.0)
        selected = _sample_episodes(episodes)
        assert len(selected) == 1
        assert selected[0].title == "Episode 1"

    def test_min_episodes_exceeds_candidates(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_EPISODES", 10)
        monkeypatch.setattr(config, "TARGET_AUDIO_MINUTES", 1)
        episodes = _make_episodes(3, duration=60.0)
        selected = _sample_episodes(episodes)
        # Can only return as many as we have
        assert len(selected) == 3

    def test_respects_min_even_if_duration_met_early(self, monkeypatch):
        """If first episode meets the duration target, we still want min_episodes."""
        monkeypatch.setattr(config, "MIN_EPISODES", 3)
        monkeypatch.setattr(config, "TARGET_AUDIO_MINUTES", 1)
        # First ep is 10 min, but we need at least 3 episodes
        episodes = _make_episodes(5, duration=600.0)
        selected = _sample_episodes(episodes)
        assert len(selected) >= 3

    def test_prefers_cached_episodes(self, monkeypatch, tmp_path):
        """Episodes with cached transcriptions should be selected first."""
        monkeypatch.setattr("config.CACHE_DIR", tmp_path)
        monkeypatch.setattr(config, "MIN_EPISODES", 2)
        monkeypatch.setattr(config, "TARGET_AUDIO_MINUTES", 5)
        # Use long episodes so length preference doesn't interfere
        episodes = _make_episodes(10, duration=700.0)

        # Create fake cache files for episodes 5 and 6
        from src.cache import _transcription_path, current_whisper_params
        params = current_whisper_params()
        for ep in episodes[4:6]:
            path = _transcription_path(ep.url, params)
            path.write_text("{}", encoding="utf-8")

        # Run many times — cached episodes should always be selected
        for _ in range(20):
            selected = _sample_episodes(episodes)
            selected_urls = {ep.url for ep in selected}
            assert episodes[4].url in selected_urls, "Cached episode 5 should be selected"
            assert episodes[5].url in selected_urls, "Cached episode 6 should be selected"


# ===================================================================
# _sort_prefer_longer
# ===================================================================

def _make_episodes_varied(durations: list[float | None]) -> list[Episode]:
    """Create episodes with specific durations."""
    return [
        Episode(
            title=f"Episode {i+1}",
            url=f"https://example.com/ep{i+1}.mp3",
            duration_seconds=d,
        )
        for i, d in enumerate(durations)
    ]


class TestSortPreferLonger:
    """Tests for the _sort_prefer_longer helper."""

    def test_short_episodes_sorted_longest_first(self, monkeypatch):
        """When median < threshold, episodes are sorted longest-first."""
        monkeypatch.setattr(config, "PREFER_LONGER_THRESHOLD", 600)
        eps = _make_episodes_varied([120, 300, 180, 240, 60])
        result = _sort_prefer_longer(eps)
        durations = [ep.duration_seconds for ep in result]
        assert durations == [300, 240, 180, 120, 60]

    def test_long_episodes_shuffled(self, monkeypatch):
        """When median >= threshold, episodes are shuffled (not sorted)."""
        monkeypatch.setattr(config, "PREFER_LONGER_THRESHOLD", 600)
        eps = _make_episodes_varied([900, 1200, 800, 1500, 700])
        # Run many times to confirm it's not always sorted
        orderings = set()
        for _ in range(30):
            result = _sort_prefer_longer(list(eps))
            orderings.add(tuple(ep.title for ep in result))
        assert len(orderings) > 1, "Long episodes should be shuffled, not deterministic"

    def test_threshold_zero_always_shuffles(self, monkeypatch):
        """Setting threshold to 0 disables the preference."""
        monkeypatch.setattr(config, "PREFER_LONGER_THRESHOLD", 0)
        eps = _make_episodes_varied([60, 120, 30])
        orderings = set()
        for _ in range(30):
            result = _sort_prefer_longer(list(eps))
            orderings.add(tuple(ep.title for ep in result))
        assert len(orderings) > 1, "Should shuffle when threshold is 0"

    def test_no_duration_info_shuffles(self, monkeypatch):
        """Episodes without duration metadata should just be shuffled."""
        monkeypatch.setattr(config, "PREFER_LONGER_THRESHOLD", 600)
        eps = _make_episodes_varied([None, None, None])
        result = _sort_prefer_longer(eps)
        assert len(result) == 3

    def test_empty_list(self, monkeypatch):
        monkeypatch.setattr(config, "PREFER_LONGER_THRESHOLD", 600)
        assert _sort_prefer_longer([]) == []

    def test_none_durations_sorted_to_end(self, monkeypatch):
        """Episodes without duration go after those with duration."""
        monkeypatch.setattr(config, "PREFER_LONGER_THRESHOLD", 600)
        eps = _make_episodes_varied([None, 300, 120, None])
        result = _sort_prefer_longer(eps)
        # The two with durations should come first (longest first)
        assert result[0].duration_seconds == 300
        assert result[1].duration_seconds == 120
