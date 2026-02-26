"""Composite scoring: normalise individual metrics and combine into a
single difficulty score mapped to the CEFR scale."""

from __future__ import annotations

import logging

import numpy as np

import config
from src.models import (
    DifficultyScore,
    EpisodeAnalysis,
    LLMAnalysis,
    StructuralMetrics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalisation ranges – based on empirical observations of Spanish podcasts.
# Each tuple is (min_value, max_value).  Values outside the range are clamped.
# ---------------------------------------------------------------------------
NORM_RANGES: dict[str, tuple[float, float]] = {
    # 40 wpm = extremely slow / beginner, 220 wpm = rapid native speech
    "speech_rate": (40.0, 220.0),
    # Fraction of words outside top-5k frequency list
    "vocabulary_level": (0.0, 0.45),
    # MATTR-based lexical diversity (more stable than raw TTR across
    # different transcript lengths).
    "lexical_diversity": (0.15, 0.55),
    # Average words per sentence (wider: Whisper punctuation is imperfect)
    "sentence_length": (2.0, 30.0),
    # Average parse-tree depth (wider: spoken Spanish parse trees are deep)
    "grammar_complexity": (1.0, 10.0),
    # LLM slang score (already 0-1)
    "slang_score": (0.0, 1.0),
    # LLM topic complexity (already 0-1)
    "topic_complexity": (0.0, 1.0),
    # Whisper avg_log_prob: more negative = less clear.
    # We invert this so that *lower* clarity → higher difficulty.
    "clarity": (-1.2, -0.1),
}

CEFR_THRESHOLDS = [
    (0.15, "A1"),
    (0.30, "A2"),
    (0.50, "B1"),
    (0.70, "B2"),
    (0.85, "C1"),
    (1.01, "C2"),
]


def _normalise(value: float, lo: float, hi: float) -> float:
    """Clamp *value* into [lo, hi] and scale to 0-1."""
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def score_episode(episode_analysis: EpisodeAnalysis) -> dict[str, float]:
    """Compute per-component normalised scores for one episode.

    Returns:
        dict mapping component name → normalised 0-1 score.
    """
    sm = episode_analysis.structural_metrics or StructuralMetrics()
    llm = episode_analysis.llm_analysis or LLMAnalysis()

    logger.debug(
        "Raw metrics: wpm=%.1f, outside_5k=%.3f, lexdiv=%.3f, "
        "sent_len=%.1f, parse_depth=%.1f, confidence=%.3f",
        sm.words_per_minute,
        sm.pct_outside_top_5k,
        sm.lexical_diversity,
        sm.avg_sentence_length,
        sm.avg_parse_depth,
        sm.avg_segment_confidence,
    )

    components: dict[str, float] = {}

    # Speech rate
    components["speech_rate"] = _normalise(
        sm.words_per_minute, *NORM_RANGES["speech_rate"]
    )

    # Vocabulary level (% outside top 5k)
    components["vocabulary_level"] = _normalise(
        sm.pct_outside_top_5k, *NORM_RANGES["vocabulary_level"]
    )

    # Lexical diversity
    components["lexical_diversity"] = _normalise(
        sm.lexical_diversity, *NORM_RANGES["lexical_diversity"]
    )

    # Sentence length
    components["sentence_length"] = _normalise(
        sm.avg_sentence_length, *NORM_RANGES["sentence_length"]
    )

    # Grammar complexity (parse depth)
    components["grammar_complexity"] = _normalise(
        sm.avg_parse_depth, *NORM_RANGES["grammar_complexity"]
    )

    # Slang (from LLM)
    components["slang_score"] = _normalise(
        llm.slang_score, *NORM_RANGES["slang_score"]
    )

    # Topic complexity (from LLM)
    components["topic_complexity"] = _normalise(
        llm.topic_complexity, *NORM_RANGES["topic_complexity"]
    )

    # Clarity (inverted: worse clarity = harder)
    # avg_log_prob is negative; more negative = harder to understand
    clarity_raw = sm.avg_segment_confidence if sm.avg_segment_confidence != 0 else -0.5
    clarity_norm = _normalise(clarity_raw, *NORM_RANGES["clarity"])
    # Invert: 0 (clear) → hard-to-hear = high difficulty
    components["clarity"] = 1.0 - clarity_norm

    return components


def _cefr_from_score(score: float) -> str:
    for threshold, level in CEFR_THRESHOLDS:
        if score < threshold:
            return level
    return "C2"


def compute_podcast_score(
    podcast_title: str,
    feed_url: str,
    episode_analyses: list[EpisodeAnalysis],
) -> DifficultyScore:
    """Aggregate episode-level analyses into a single podcast difficulty score."""

    if not episode_analyses:
        return DifficultyScore(
            podcast_title=podcast_title,
            feed_url=feed_url,
        )

    weights = dict(config.SCORING_WEIGHTS)

    # If no episodes have LLM analysis, redistribute LLM weights
    # proportionally across the structural components so the score
    # still uses the full 0-1 range.
    has_llm = any(ea.llm_analysis is not None for ea in episode_analyses)
    llm_keys = {"slang_score", "topic_complexity"}
    if not has_llm:
        llm_weight = sum(weights[k] for k in llm_keys)
        structural_keys = [k for k in weights if k not in llm_keys]
        structural_total = sum(weights[k] for k in structural_keys)
        for k in structural_keys:
            # Scale each structural weight proportionally to absorb the LLM share
            weights[k] = weights[k] / structural_total * (structural_total + llm_weight)
        for k in llm_keys:
            weights[k] = 0.0
        logger.info(
            "No LLM analysis — redistributed weights: %s",
            {k: round(v, 3) for k, v in weights.items() if v > 0},
        )

    # Score each episode
    all_components: list[dict[str, float]] = []
    for ea in episode_analyses:
        comps = score_episode(ea)
        all_components.append(comps)

    # Outlier trimming: drop the N highest-scoring episodes to prevent
    # a single atypical episode from skewing the podcast average.
    trimmed_indices: list[int] = []
    trim_count = getattr(config, "OUTLIER_TRIM_COUNT", 1)
    trim_min = getattr(config, "OUTLIER_TRIM_MIN_EPISODES", 4)
    if trim_count > 0 and len(all_components) >= trim_min:
        # Compute a quick weighted score per episode for ranking
        episode_scores = []
        for i, comps in enumerate(all_components):
            s = sum(comps.get(k, 0.0) * weights[k] for k in weights)
            episode_scores.append((s, i))
        episode_scores.sort(reverse=True)
        # Trim the top N outliers
        for _, idx in episode_scores[:trim_count]:
            trimmed_indices.append(idx)
            title = episode_analyses[idx].episode.title
            logger.info(
                "Trimmed outlier episode: '%s' (score=%.3f)",
                title, episode_scores[0][0],
            )

    # Build the kept list
    kept_components = [
        c for i, c in enumerate(all_components) if i not in trimmed_indices
    ]
    kept_analyses = [
        ea for i, ea in enumerate(episode_analyses) if i not in trimmed_indices
    ]

    # Average each component across kept episodes
    avg_components: dict[str, float] = {}
    for key in weights:
        if weights[key] == 0.0:
            continue
        vals = [c.get(key, 0.0) for c in kept_components]
        avg_components[key] = round(float(np.mean(vals)), 4)

    # Weighted sum
    overall = sum(avg_components.get(k, 0.0) * weights[k] for k in weights)
    overall = round(float(np.clip(overall, 0.0, 1.0)), 4)
    cefr = _cefr_from_score(overall)

    # Also incorporate LLM CEFR estimates if available
    llm_cefrs = [
        ea.llm_analysis.estimated_cefr
        for ea in episode_analyses
        if ea.llm_analysis and ea.llm_analysis.estimated_cefr
    ]
    if llm_cefrs:
        logger.info("LLM CEFR estimates: %s → algorithmic CEFR: %s", llm_cefrs, cefr)

    trimmed_titles = [episode_analyses[i].episode.title for i in trimmed_indices]

    result = DifficultyScore(
        podcast_title=podcast_title,
        feed_url=feed_url,
        overall_score=overall,
        cefr_estimate=cefr,
        component_scores=avg_components,
        episodes_analyzed=len(kept_analyses),
        episode_results=episode_analyses,  # keep all for the report
        trimmed_episodes=trimmed_titles,
    )

    trimmed_msg = ""
    if trimmed_titles:
        trimmed_msg = f" (trimmed {len(trimmed_titles)} outlier: {', '.join(trimmed_titles)})"
    logger.info(
        "Podcast '%s' → score=%.3f (%s), %d episodes%s",
        podcast_title,
        overall,
        cefr,
        len(kept_analyses),
        trimmed_msg,
    )
    return result


def format_report(score: DifficultyScore) -> str:
    """Return a human-readable text report for a podcast score."""
    # Detect whether LLM was used
    has_llm = any(
        ea.llm_analysis is not None
        for ea in score.episode_results
    )
    mode = "Structural + LLM" if has_llm else "Structural only (no LLM)"

    ep_label = str(score.episodes_analyzed)
    if score.trimmed_episodes:
        ep_label += f" (+ {len(score.trimmed_episodes)} outlier trimmed)"

    lines = [
        "=" * 60,
        f"  Podcast Difficulty Report",
        "=" * 60,
        f"  Title:    {score.podcast_title}",
        f"  Feed:     {score.feed_url}",
        f"  Episodes: {ep_label}",
        f"  Mode:     {mode}",
        "",
        f"  Overall Score:  {score.overall_score:.3f} / 1.000",
        f"  CEFR Estimate:  {score.cefr_estimate}",
        "",
        "  Component Scores:",
    ]

    # LLM-dependent components to label
    llm_components = {"slang_score", "topic_complexity"}

    for comp, val in sorted(score.component_scores.items()):
        bar = "#" * int(val * 20)
        suffix = "  (LLM)" if comp in llm_components and has_llm else ""
        lines.append(f"    {comp:<25s} {val:.3f}  {bar}{suffix}")

    if not has_llm:
        lines.append("")
        lines.append("  * Slang and topic scores excluded (LLM disabled).")
        lines.append("    Use --use-llm or --local-llm for full analysis.")

    lines.append("")

    # Per-episode summaries
    trimmed_set = set(score.trimmed_episodes)
    for i, ea in enumerate(score.episode_results, 1):
        trimmed_tag = "  [TRIMMED]" if ea.episode.title in trimmed_set else ""
        lines.append(f"  Episode {i}: {ea.episode.title}{trimmed_tag}")
        if ea.structural_metrics:
            sm = ea.structural_metrics
            lines.append(
                f"    Words: {sm.total_words}  |  WPM: {sm.words_per_minute}  |  "
                f"Lex div: {sm.lexical_diversity:.3f}  |  Outside 5k: {sm.pct_outside_top_5k:.1%}"
            )
        if ea.llm_analysis and ea.llm_analysis.estimated_cefr:
            la = ea.llm_analysis
            lines.append(
                f"    LLM CEFR: {la.estimated_cefr}  |  Slang: {la.slang_score:.2f}  |  "
                f"Topic: {la.topic_complexity:.2f}"
            )
            if la.slang_examples:
                lines.append(f"    Slang examples: {', '.join(la.slang_examples[:5])}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_ranking(results: list[DifficultyScore]) -> str:
    """Return a concise comparative ranking table.

    Shows all podcasts (easiest → hardest) in a single table with one row
    per podcast and short column headers for each scoring component,
    followed by a key that explains each column and its weight.
    """
    ranked = sorted(results, key=lambda r: r.overall_score)
    weights = dict(config.SCORING_WEIGHTS)

    # Detect whether any result has LLM data
    has_llm = any(
        any(ea.llm_analysis is not None for ea in r.episode_results)
        for r in ranked
    )
    if not has_llm:
        # Mirror the redistribution done in compute_podcast_score
        llm_keys = {"slang_score", "topic_complexity"}
        llm_weight = sum(weights[k] for k in llm_keys)
        structural_keys = [k for k in weights if k not in llm_keys]
        structural_total = sum(weights[k] for k in structural_keys)
        for k in structural_keys:
            weights[k] = weights[k] / structural_total * (structural_total + llm_weight)
        for k in llm_keys:
            weights[k] = 0.0

    # Column definitions: (key, short_header, full_description)
    columns: list[tuple[str, str, str]] = [
        ("speech_rate",       "SPD",  "Speech rate (words per minute)"),
        ("vocabulary_level",  "VOC",  "Vocabulary level (% outside top-5k)"),
        ("lexical_diversity", "LEX",  "Lexical diversity (MATTR)"),
        ("sentence_length",   "SEN",  "Sentence length (avg words/sentence)"),
        ("grammar_complexity","GRM",  "Grammar complexity (parse depth)"),
        ("slang_score",       "SLG",  "Slang usage (LLM)"),
        ("topic_complexity",  "TOP",  "Topic complexity (LLM)"),
        ("clarity",           "CLR",  "Clarity (Whisper confidence)"),
    ]

    # Filter to active columns (non-zero weight)
    active = [(k, hdr, desc) for k, hdr, desc in columns if weights.get(k, 0) > 0]

    # Find the longest podcast title for the name column
    max_title = max((len(r.podcast_title) for r in ranked), default=10)
    name_w = min(max(max_title, 10), 40)  # clamp 10–40

    # Build header
    hdr_parts = [f"{'#':>3s}  {'Podcast':<{name_w}s}  {'Score':>5s}  {'CEFR':>4s}  {'Ep':>2s}"]
    for _, hdr, _ in active:
        hdr_parts.append(f"  {hdr:>5s}")
    header = "".join(hdr_parts)
    sep = "─" * len(header)

    lines: list[str] = [
        "",
        sep,
        "  RANKING  (easiest → hardest)",
        sep,
        header,
        sep,
    ]

    for rank, r in enumerate(ranked, 1):
        cefr = r.cefr_estimate or "??"
        title = r.podcast_title[:name_w]
        row = [f"{rank:>3d}  {title:<{name_w}s}  {r.overall_score:5.3f}  {cefr:>4s}  {r.episodes_analyzed:>2d}"]
        for key, _, _ in active:
            val = r.component_scores.get(key, 0.0)
            row.append(f"  {val:5.3f}")
        lines.append("".join(row))
        if r.trimmed_episodes:
            lines.append(f"{'':>{3 + 2 + name_w}}  ↳ trimmed: {', '.join(r.trimmed_episodes)}")

    lines.append(sep)

    # Weight key
    lines.append("")
    lines.append("  WEIGHTS KEY  (component scores are 0–1; higher = harder)")
    lines.append(f"  {'Col':>5s}  {'Wt':>5s}  Description")
    lines.append(f"  {'───':>5s}  {'───':>5s}  {'─' * 40}")
    for key, hdr, desc in active:
        w = weights[key]
        lines.append(f"  {hdr:>5s}  {w:5.2f}  {desc}")
    lines.append(f"  {'':>5s}  {'─────':>5s}")
    lines.append(f"  {'':>5s}  {sum(weights[k] for k, _, _ in active):5.2f}  TOTAL")

    if not has_llm:
        lines.append("")
        lines.append("  * LLM components excluded — weights redistributed to structural metrics.")
        lines.append("    Use --use-llm or --local-llm for full analysis.")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)
