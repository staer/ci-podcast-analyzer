"""Feed parsing and episode audio downloading."""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import requests

import config
from src.models import Episode, PodcastFeed

logger = logging.getLogger(__name__)


def parse_feed(feed_url: str) -> PodcastFeed:
    """Parse an RSS feed and return structured podcast metadata.

    Collects all eligible episodes from the feed, then randomly samples
    enough to reach TARGET_AUDIO_MINUTES (with a minimum of MIN_EPISODES).

    Args:
        feed_url: URL of the podcast RSS feed.

    Returns:
        A PodcastFeed with a randomly sampled set of episodes.
    """
    logger.info("Parsing feed: %s", feed_url)
    parsed = feedparser.parse(feed_url)

    if parsed.bozo and not parsed.entries:
        raise ValueError(f"Failed to parse feed: {parsed.bozo_exception}")

    feed_title = parsed.feed.get("title", "Unknown Podcast")
    feed_desc = parsed.feed.get("summary", parsed.feed.get("subtitle", ""))
    feed_lang = parsed.feed.get("language", "")

    # Collect ALL eligible episodes (up to hard cap)
    candidates: list[Episode] = []
    for entry in parsed.entries[: config.MAX_EPISODES_PER_FEED]:
        audio_url = _extract_audio_url(entry)
        if not audio_url:
            logger.debug("Skipping entry without audio enclosure: %s", entry.get("title"))
            continue

        duration = _parse_duration(entry)

        # Skip episodes that are too long
        if duration and duration > config.MAX_AUDIO_DURATION_MINUTES * 60:
            logger.info(
                "Skipping long episode (%d min): %s",
                duration // 60,
                entry.get("title"),
            )
            continue

        candidates.append(
            Episode(
                title=entry.get("title", "Untitled"),
                url=audio_url,
                published=entry.get("published", None),
                duration_seconds=duration,
            )
        )

    logger.info("Found %d eligible episodes in '%s'", len(candidates), feed_title)

    # Randomly sample episodes to reach the target duration
    episodes = _sample_episodes(candidates)

    return PodcastFeed(
        title=feed_title,
        feed_url=feed_url,
        description=feed_desc,
        language=feed_lang,
        episodes=episodes,
    )


def _sample_episodes(candidates: list[Episode]) -> list[Episode]:
    """Sample episodes to reach TARGET_AUDIO_MINUTES, preferring cached ones.

    Strategy:
    - Partition candidates into those with existing cached transcriptions
      and those without
    - Pick cached episodes first (shuffled), then new episodes (shuffled)
    - Pick episodes until we reach the target duration
    - Always pick at least MIN_EPISODES (even if that exceeds the target)
    - If no duration metadata is available, fall back to MIN_EPISODES
    """
    if not candidates:
        return []

    from src.cache import _transcription_path, current_whisper_params
    params = current_whisper_params()

    # Partition: cached first, then fresh (both shuffled internally)
    cached = [ep for ep in candidates if _transcription_path(ep.url, params).exists()]
    fresh = [ep for ep in candidates if ep not in cached]
    random.shuffle(cached)
    random.shuffle(fresh)
    ordered = cached + fresh

    if cached:
        logger.info(
            "%d of %d candidates have cached transcriptions — preferring those",
            len(cached), len(candidates),
        )

    target_seconds = config.TARGET_AUDIO_MINUTES * 60
    min_eps = config.MIN_EPISODES

    selected: list[Episode] = []
    total_duration = 0.0
    has_duration_info = any(ep.duration_seconds for ep in ordered)

    for ep in ordered:
        selected.append(ep)
        ep_dur = ep.duration_seconds or 0.0
        total_duration += ep_dur

        # Stop once we have enough duration AND enough episodes
        if has_duration_info and len(selected) >= min_eps and total_duration >= target_seconds:
            break
        # If no duration info at all, just pick min_eps
        if not has_duration_info and len(selected) >= min_eps:
            break

    logger.info(
        "Sampled %d episodes (%.0f min) from %d candidates (target: %d min, min eps: %d)",
        len(selected),
        total_duration / 60,
        len(candidates),
        config.TARGET_AUDIO_MINUTES,
        min_eps,
    )
    return selected


def download_episode(episode: Episode, output_dir: Path | None = None) -> Episode:
    """Download the audio file for an episode.

    Args:
        episode: Episode with a valid audio URL.
        output_dir: Directory to save audio.  Defaults to config.AUDIO_DIR.

    Returns:
        A copy of the Episode with audio_path set.
    """
    output_dir = output_dir or config.AUDIO_DIR

    # Build a safe filename from the episode title
    safe_title = re.sub(r"[^\w\s-]", "", episode.title)[:80].strip()
    safe_title = re.sub(r"\s+", "_", safe_title)

    # Determine extension from URL
    parsed_url = urlparse(episode.url)
    ext = Path(parsed_url.path).suffix or ".mp3"
    filename = f"{safe_title}{ext}"
    dest = output_dir / filename

    if dest.exists():
        logger.info("Audio already downloaded: %s", dest)
        return episode.model_copy(update={"audio_path": str(dest)})

    logger.info("Downloading: %s → %s", episode.url, dest)
    response = requests.get(
        episode.url,
        stream=True,
        timeout=config.DOWNLOAD_TIMEOUT_SECONDS,
        headers={"User-Agent": "PodcastDifficultyAnalyzer/1.0"},
    )
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            f.write(chunk)

    logger.info("Downloaded %.1f MB", dest.stat().st_size / (1024 * 1024))
    return episode.model_copy(update={"audio_path": str(dest)})


def download_episodes(feed: PodcastFeed) -> list[Episode]:
    """Download audio for all episodes in a feed.

    Returns:
        List of episodes with audio_path populated.
    """
    downloaded: list[Episode] = []
    for ep in feed.episodes:
        try:
            ep_with_audio = download_episode(ep)
            downloaded.append(ep_with_audio)
        except requests.RequestException as exc:
            logger.warning("Failed to download '%s': %s", ep.title, exc)
    return downloaded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_audio_url(entry: dict) -> str | None:
    """Pull the audio enclosure URL from a feed entry."""
    # Standard RSS enclosures
    for link in entry.get("links", []):
        href = link.get("href", "")
        link_type = link.get("type", "")
        if "audio" in link_type or any(
            href.lower().endswith(ext) for ext in (".mp3", ".m4a", ".ogg", ".wav")
        ):
            return href

    # feedparser puts enclosures here too
    for enc in entry.get("enclosures", []):
        url = enc.get("url") or enc.get("href", "")
        if url:
            return url

    return None


def _parse_duration(entry: dict) -> float | None:
    """Try to extract episode duration in seconds from feed metadata."""
    raw = entry.get("itunes_duration", None)
    if not raw:
        return None

    # Could be seconds as a plain number, or HH:MM:SS / MM:SS
    raw = str(raw).strip()
    if raw.isdigit():
        return float(raw)

    parts = raw.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        pass
    return None
