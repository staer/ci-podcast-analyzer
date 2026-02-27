#!/usr/bin/env python3
"""Spanish Podcast Difficulty Analyzer – CLI entry point.

Usage examples:
    # Analyze a single podcast feed
    python main.py https://feeds.example.com/spanish-podcast.xml

    # Analyze multiple feeds and compare
    python main.py feed1.xml feed2.xml feed3.xml

    # Skip LLM analysis (default - offline / free mode)
    python main.py https://feeds.example.com/podcast.xml

    # Enable LLM analysis (requires OPENAI_API_KEY)
    python main.py --use-llm https://feeds.example.com/podcast.xml

    # Analyze 45 minutes of audio, at least 3 episodes
    python main.py --duration 45 --min-episodes 3 https://feeds.example.com/podcast.xml

    # Use a different Whisper model size
    python main.py --whisper-model large-v3 https://feeds.example.com/podcast.xml

    # Limit to N episodes
    python main.py --episodes 5 https://feeds.example.com/podcast.xml

    # Save results to JSON
    python main.py --output results.json https://feeds.example.com/podcast.xml

    # Use Ollama (free local LLM) instead of OpenAI
    python main.py --local-llm https://feeds.example.com/podcast.xml

    # Use a specific Ollama model
    python main.py --ollama-model mistral https://feeds.example.com/podcast.xml

    # Run from a feeds file (repeatable batch mode)
    python main.py --feeds-file feeds.json

    # Feeds file with CLI overrides
    python main.py --feeds-file feeds.json --duration 60 --output results.json
"""

from __future__ import annotations

import sys

# ---- Python version guard ------------------------------------------------
# spaCy 3.x requires Python >=3.9, <3.13.  We target 3.10+ for modern syntax.
_v = sys.version_info
if _v < (3, 10) or _v >= (3, 13):
    sys.exit(
        f"[ERROR] Python 3.10–3.12 is required (detected {_v.major}.{_v.minor}).\n"
        "        spaCy 3.x does not yet support Python 3.13+.\n"
        "        Install a compatible version from https://www.python.org/downloads/"
    )
# --------------------------------------------------------------------------

import argparse
import json
import logging

import config
from src.cache import (
    load_cached_transcription,
    save_cached_transcription,
    load_cached_analysis,
    save_cached_analysis,
    load_cached_llm_analysis,
    save_cached_llm_analysis,
    scan_cached_analyses,
    scan_cached_llm_analyses,
)
from src.feed import download_episodes, parse_feed
from src.transcribe import transcribe_episode, save_transcript
from src.analyze import analyze_structure
from src.llm_analyze import analyze_with_llm
from src.models import DifficultyScore, EpisodeAnalysis
from src.scoring import compute_podcast_score, format_ranking, format_report


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Ensure stdout/stderr can handle Unicode on Windows
    import io
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def _load_feeds_file(path: str) -> tuple[list[str], dict]:
    """Load feed URLs and settings from a JSON file.

    Expected format:
    {
      "settings": { ... },
      "feeds": [
        { "name": "...", "url": "..." },
        ...
      ]
    }

    Returns:
        Tuple of (feed_urls, settings_dict).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    settings = data.get("settings", {})
    feeds = data.get("feeds", [])
    urls = [entry["url"] for entry in feeds if "url" in entry]

    if not urls:
        raise ValueError(f"No feeds found in {path}")

    feed_names = [entry.get("name", entry["url"]) for entry in feeds if "url" in entry]
    logging.info("Loaded %d feeds from %s: %s", len(urls), path, ", ".join(feed_names))
    return urls, settings


def rescore_from_cache() -> list[DifficultyScore]:
    """Rescore every podcast in the cache using current config weights.

    No network access, no transcription — reads cached an_*.json and
    (optionally) llm_*.json files and recomputes the composite score.

    Returns:
        List of DifficultyScore objects, one per unique feed_url in the cache.
    """
    grouped = scan_cached_analyses()
    if not grouped:
        logging.warning("No cached analyses found in %s", config.CACHE_DIR)
        return []

    # Also load any cached LLM results, keyed by episode_url
    llm_by_episode = scan_cached_llm_analyses()

    results: list[DifficultyScore] = []
    for feed_url, cached_items in grouped.items():
        # Determine podcast title from the first item that has one
        podcast_title = next(
            (c.podcast_title for c in cached_items if c.podcast_title), feed_url or "Unknown"
        )

        # Build stub EpisodeAnalysis objects from cached data
        episode_analyses: list[EpisodeAnalysis] = []
        for ca in cached_items:
            from src.models import Episode

            ea = EpisodeAnalysis(
                episode=Episode(
                    title=ca.episode_title or ca.episode_url,
                    url=ca.episode_url,
                ),
                structural_metrics=ca.structural_metrics,
            )
            # Attach cached LLM analysis if available
            if ca.episode_url in llm_by_episode:
                ea.llm_analysis = llm_by_episode[ca.episode_url].llm_analysis
            episode_analyses.append(ea)

        score = compute_podcast_score(podcast_title, feed_url, episode_analyses)
        results.append(score)

    return results


def reanalyze_from_cache() -> list[DifficultyScore]:
    """Re-run structural NLP analysis from cached transcriptions, then rescore.

    This recomputes all StructuralMetrics (including punctuation_density)
    from the cached transcription text without re-downloading or re-
    transcribing audio.  Updated metrics are written back to the analysis
    cache so that future ``--rescore`` runs use the new values.

    Returns:
        List of DifficultyScore objects, one per unique feed_url.
    """
    grouped = scan_cached_analyses()
    if not grouped:
        logging.warning("No cached analyses found in %s", config.CACHE_DIR)
        return []

    llm_by_episode = scan_cached_llm_analyses()

    total_reanalyzed = 0
    total_skipped = 0
    results: list[DifficultyScore] = []

    for feed_url, cached_items in grouped.items():
        podcast_title = next(
            (c.podcast_title for c in cached_items if c.podcast_title), feed_url or "Unknown"
        )

        episode_analyses: list[EpisodeAnalysis] = []
        for ca in cached_items:
            from src.models import Episode

            ep = Episode(
                title=ca.episode_title or ca.episode_url,
                url=ca.episode_url,
            )
            ea = EpisodeAnalysis(episode=ep)

            # Try to load the cached transcription for this episode
            transcript = load_cached_transcription(ca.episode_url, ca.whisper_params)
            if transcript is not None:
                try:
                    metrics = analyze_structure(transcript)
                    # Update the analysis cache with fresh metrics
                    save_cached_analysis(
                        ca.episode_url, metrics,
                        params=ca.whisper_params,
                        episode_title=ca.episode_title,
                        podcast_title=ca.podcast_title,
                        feed_url=ca.feed_url,
                    )
                    ea.structural_metrics = metrics
                    total_reanalyzed += 1
                except Exception as exc:
                    logging.error(
                        "Re-analysis failed for '%s': %s — keeping old metrics",
                        ca.episode_title, exc,
                    )
                    ea.structural_metrics = ca.structural_metrics
                    total_skipped += 1
            else:
                logging.warning(
                    "No cached transcription for '%s' — keeping old metrics",
                    ca.episode_title,
                )
                ea.structural_metrics = ca.structural_metrics
                total_skipped += 1

            if ca.episode_url in llm_by_episode:
                ea.llm_analysis = llm_by_episode[ca.episode_url].llm_analysis

            episode_analyses.append(ea)

        score = compute_podcast_score(podcast_title, feed_url, episode_analyses)
        results.append(score)

    logging.info(
        "Re-analysis complete: %d episodes reanalyzed, %d kept old metrics",
        total_reanalyzed, total_skipped,
    )
    return results


def analyze_feed(feed_url: str, *, use_llm: bool = True, use_cache: bool = True) -> DifficultyScore:
    """Full pipeline: parse → download → transcribe → analyze → score."""

    # 1. Parse the RSS feed
    feed = parse_feed(feed_url)
    if not feed.episodes:
        logging.warning("No downloadable episodes found in '%s'", feed.title)
        return DifficultyScore(podcast_title=feed.title, feed_url=feed_url)

    # 2. Download audio files
    episodes = download_episodes(feed)
    if not episodes:
        logging.error("All downloads failed for '%s'", feed.title)
        return DifficultyScore(podcast_title=feed.title, feed_url=feed_url)

    # 3. Process each episode
    episode_analyses: list[EpisodeAnalysis] = []
    for ep in episodes:
        logging.info("=" * 50)
        logging.info("Processing: %s", ep.title)

        ea = EpisodeAnalysis(episode=ep)

        # 3a. Transcribe (check cache first)
        transcript = None
        if use_cache:
            transcript = load_cached_transcription(ep.url)

        if transcript is None:
            try:
                transcript = transcribe_episode(ep)
                save_transcript(transcript)
                if use_cache:
                    save_cached_transcription(ep.url, transcript)
            except Exception as exc:
                logging.error("Transcription failed for '%s': %s", ep.title, exc)
                episode_analyses.append(ea)
                continue
        else:
            logging.info(">> Using cached transcription (skipping Whisper)")

        ea.transcription = transcript

        # 3b. Structural NLP analysis (check cache first)
        metrics = None
        if use_cache:
            metrics = load_cached_analysis(ep.url)

        if metrics is None:
            try:
                metrics = analyze_structure(transcript)
                if use_cache:
                    save_cached_analysis(
                        ep.url, metrics,
                        episode_title=ep.title,
                        podcast_title=feed.title,
                        feed_url=feed_url,
                    )
            except Exception as exc:
                logging.error("Structural analysis failed for '%s': %s", ep.title, exc)
        else:
            logging.info(">> Using cached structural analysis")

        ea.structural_metrics = metrics

        # 3c. LLM qualitative analysis
        if use_llm:
            llm_result = None
            if use_cache:
                llm_result = load_cached_llm_analysis(ep.url)
            if llm_result is None:
                try:
                    llm_result = analyze_with_llm(transcript)
                    if use_cache and llm_result:
                        save_cached_llm_analysis(
                            ep.url, llm_result,
                            episode_title=ep.title,
                            podcast_title=feed.title,
                            feed_url=feed_url,
                        )
                except Exception as exc:
                    logging.warning("LLM analysis failed for '%s': %s", ep.title, exc)
            else:
                logging.info(">> Using cached LLM analysis")
            ea.llm_analysis = llm_result

        episode_analyses.append(ea)

    # 4. Compute composite score
    return compute_podcast_score(feed.title, feed_url, episode_analyses)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Spanish podcast difficulty from RSS feeds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "feeds",
        nargs="*",
        default=[],
        help="One or more podcast RSS feed URLs to analyze.",
    )
    parser.add_argument(
        "--feeds-file", "-f",
        default=None,
        help="Path to a JSON file with feed URLs and settings (see feeds.json).",
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=None,
        help="Hard cap on episodes to consider from feed (default: %(default)s).",
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=None,
        help="Target audio duration in minutes to sample per feed (default: 30).",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=None,
        help="Minimum episodes to analyze per feed, even if duration target is met (default: 2).",
    )
    parser.add_argument(
        "--whisper-model", "-w",
        default=None,
        help="Whisper model size: tiny, base, small, medium, large-v3.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size for Whisper decoding. 1=greedy (fast), 5=beam (accurate). Default: 1.",
    )
    parser.add_argument(
        "--max-transcribe-minutes",
        type=int,
        default=None,
        help="Only transcribe the first N minutes of each episode (0=full). Default: 10.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM analysis (requires OpenAI API key or --local-llm).",
    )
    parser.add_argument(
        "--local-llm",
        action="store_true",
        help="Use Ollama (local) instead of OpenAI. Free, no API key needed.",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Ollama model to use (default: llama3). Implies --local-llm.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Save results as JSON to this file.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-transcription, ignoring cached results.",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Rescore all cached episodes using current weights. No downloads or "
             "transcription — instant results from the cache database.",
    )
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="Re-run structural NLP analysis from cached transcriptions then rescore. "
             "No downloads or transcription — recomputes metrics like punctuation_density.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # --reanalyze: re-run NLP from cached transcriptions, then rescore
    # Checked first because it's a superset of --rescore.
    if args.reanalyze:
        print("\n[*] Re-analyzing all cached transcriptions and rescoring...\n")
        results = reanalyze_from_cache()
        if not results:
            print("  No cached data found. Run the analyzer first to build the cache.")
            sys.exit(1)

        for score in results:
            print(format_report(score))

        if len(results) > 1:
            print(format_ranking(results))

        if args.output:
            output_data = [r.model_dump() for r in results]
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n[*] Results saved to {args.output}")

        return

    # --rescore: instant rescore from cache, no network needed
    if args.rescore:
        print("\n[*] Rescoring all cached episodes with current weights...\n")
        results = rescore_from_cache()
        if not results:
            print("  No cached data found. Run the analyzer first to build the cache.")
            sys.exit(1)

        for score in results:
            print(format_report(score))

        if len(results) > 1:
            print(format_ranking(results))

        if args.output:
            output_data = [r.model_dump() for r in results]
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n[*] Results saved to {args.output}")

        return

    # Load feeds from JSON file if provided
    feed_urls: list[str] = list(args.feeds)
    if args.feeds_file:
        feed_urls, file_settings = _load_feeds_file(args.feeds_file)
        # JSON file settings are defaults; CLI flags override them
        if not args.episodes and "max_episodes" in file_settings:
            config.MAX_EPISODES_PER_FEED = file_settings["max_episodes"]
        if not args.duration and "target_audio_minutes" in file_settings:
            config.TARGET_AUDIO_MINUTES = file_settings["target_audio_minutes"]
        if not args.min_episodes and "min_episodes" in file_settings:
            config.MIN_EPISODES = file_settings["min_episodes"]
        if not args.whisper_model and "whisper_model" in file_settings:
            config.WHISPER_MODEL_SIZE = file_settings["whisper_model"]
        if "skip_intro_seconds" in file_settings:
            config.SKIP_INTRO_SECONDS = file_settings["skip_intro_seconds"]
        if not args.beam_size and "beam_size" in file_settings:
            config.WHISPER_BEAM_SIZE = file_settings["beam_size"]
        if not args.max_transcribe_minutes and "max_transcribe_minutes" in file_settings:
            config.MAX_TRANSCRIBE_MINUTES = file_settings["max_transcribe_minutes"]
        if file_settings.get("use_llm"):
            args.use_llm = True
        if file_settings.get("local_llm"):
            config.USE_LOCAL_LLM = True
        if "ollama_model" in file_settings:
            config.USE_LOCAL_LLM = True
            config.OLLAMA_MODEL = file_settings["ollama_model"]

    if not feed_urls and not args.rescore and not args.reanalyze:
        parser.error("Provide feed URLs as arguments or via --feeds-file.")

    # Apply CLI overrides to config (CLI wins over file)
    if args.episodes:
        config.MAX_EPISODES_PER_FEED = args.episodes
    if args.duration:
        config.TARGET_AUDIO_MINUTES = args.duration
    if args.min_episodes:
        config.MIN_EPISODES = args.min_episodes
    if args.whisper_model:
        config.WHISPER_MODEL_SIZE = args.whisper_model
    if args.beam_size is not None:
        config.WHISPER_BEAM_SIZE = args.beam_size
    if args.max_transcribe_minutes is not None:
        config.MAX_TRANSCRIBE_MINUTES = args.max_transcribe_minutes
    if args.local_llm or args.ollama_model:
        config.USE_LOCAL_LLM = True
    if args.ollama_model:
        config.OLLAMA_MODEL = args.ollama_model

    # Run analysis for each feed
    results: list[DifficultyScore] = []
    for feed_url in feed_urls:
        print(f"\n[*] Analyzing: {feed_url}\n")
        score = analyze_feed(feed_url, use_llm=args.use_llm, use_cache=not args.no_cache)
        results.append(score)
        print(format_report(score))

    # If multiple feeds, print a comparison ranking
    if len(results) > 1:
        print(format_ranking(results))

    # Save JSON output
    if args.output:
        output_data = [r.model_dump() for r in results]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n[*] Results saved to {args.output}")


if __name__ == "__main__":
    main()
