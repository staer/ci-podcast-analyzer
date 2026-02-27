"""Composite scoring: normalise individual metrics and combine into a
single difficulty score mapped to the CEFR scale."""

from __future__ import annotations

import json
import logging
from pathlib import Path

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
# Trainable settings – loaded from settings.json so the training program
# can optimise them without touching source code.
# ---------------------------------------------------------------------------
_SETTINGS_PATH = Path(__file__).parent.parent / "settings.json"


def _load_settings() -> dict:
    """Load scoring settings from settings.json."""
    if _SETTINGS_PATH.exists():
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.warning("settings.json not found — using built-in defaults")
    return {}


def _get_norm_ranges() -> dict[str, tuple[float, float]]:
    """Return normalisation ranges from settings.json (or defaults)."""
    settings = _load_settings()
    raw = settings.get("norm_ranges", {})
    defaults = {
        "speech_rate": (60.0, 220.0),
        "vocabulary_level": (0.02, 0.35),
        "lexical_diversity": (0.40, 0.65),
        "sentence_length": (2.0, 30.0),
        "grammar_complexity": (3.0, 6.5),
        "tense_complexity": (0.05, 0.60),
        "slang_score": (0.0, 1.0),
        "topic_complexity": (0.0, 1.0),
        "clarity": (0.005, 0.12),
    }
    result = {}
    for key, default in defaults.items():
        if key in raw and len(raw[key]) == 2:
            result[key] = (float(raw[key][0]), float(raw[key][1]))
        else:
            result[key] = default
    return result


def _get_weights() -> dict[str, float]:
    """Return scoring weights from settings.json (or config defaults)."""
    settings = _load_settings()
    raw = settings.get("weights", {})
    if raw:
        return {k: float(v) for k, v in raw.items()}
    return dict(config.SCORING_WEIGHTS)


def _get_cefr_thresholds() -> list[tuple[float, str]]:
    """Return CEFR thresholds from settings.json (or defaults)."""
    settings = _load_settings()
    raw = settings.get("cefr_thresholds", [])
    if raw:
        return [(float(pair[0]), str(pair[1])) for pair in raw]
    return [
        (0.20, "A1"),
        (0.35, "A2"),
        (0.50, "B1"),
        (0.65, "B2"),
        (0.80, "C1"),
        (1.01, "C2"),
    ]


# Module-level references — reloaded each time scoring runs so that
# the training program can update settings.json between iterations.
NORM_RANGES = _get_norm_ranges()
CEFR_THRESHOLDS = _get_cefr_thresholds()


def _normalise(value: float, lo: float, hi: float) -> float:
    """Clamp *value* into [lo, hi] and scale to 0-1."""
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def reload_settings() -> None:
    """Reload NORM_RANGES and CEFR_THRESHOLDS from settings.json.

    Call this after the training program writes updated settings so that
    subsequent score_episode / compute_podcast_score calls use the new values.
    """
    global NORM_RANGES, CEFR_THRESHOLDS
    NORM_RANGES = _get_norm_ranges()
    CEFR_THRESHOLDS = _get_cefr_thresholds()


_MAX_WEIGHT = 0.50  # No single component may exceed 50%


def _redistribute_llm_weights(
    weights: dict[str, float],
    llm_keys: set[str],
) -> None:
    """Redistribute LLM weights to structural keys *in-place*.

    After proportional redistribution, any component exceeding
    ``_MAX_WEIGHT`` is capped and the excess is spread evenly
    across the remaining uncapped components (iteratively).
    """
    llm_total = sum(weights[k] for k in llm_keys)
    if llm_total == 0:
        return

    structural_keys = [k for k in weights if k not in llm_keys]
    structural_total = sum(weights[k] for k in structural_keys)

    # Step 1: proportional scale-up
    for k in structural_keys:
        weights[k] = weights[k] / structural_total * (structural_total + llm_total)
    for k in llm_keys:
        weights[k] = 0.0

    # Step 2: iterative cap enforcement
    for _ in range(10):  # converges in 2-3 rounds
        capped = {k for k in structural_keys if weights[k] > _MAX_WEIGHT}
        if not capped:
            break
        excess = sum(weights[k] - _MAX_WEIGHT for k in capped)
        for k in capped:
            weights[k] = _MAX_WEIGHT
        uncapped = [k for k in structural_keys if k not in capped and weights[k] > 0]
        if not uncapped:
            break
        uncapped_total = sum(weights[k] for k in uncapped)
        for k in uncapped:
            weights[k] += excess * (weights[k] / uncapped_total)


def score_episode(
    episode_analysis: EpisodeAnalysis,
    *,
    norm_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Compute per-component normalised scores for one episode.

    Returns:
        dict mapping component name → normalised 0-1 score.
        Also includes the special key ``_sentence_confidence`` (0-1)
        which indicates how reliable the sentence-length score is based
        on punctuation density.  A value near 1 means punctuation was
        plentiful; near 0 means Whisper likely produced run-on text.
    """
    sm = episode_analysis.structural_metrics or StructuralMetrics()
    llm = episode_analysis.llm_analysis or LLMAnalysis()

    ranges = norm_ranges or NORM_RANGES

    logger.debug(
        "Raw metrics: wpm=%.1f, vocab_score=%.3f, lexdiv=%.3f, "
        "sent_len=%.1f, parse_depth=%.1f, confidence=%.3f",
        sm.words_per_minute,
        sm.vocab_score,
        sm.lexical_diversity,
        sm.avg_sentence_length,
        sm.avg_parse_depth,
        sm.avg_segment_confidence,
    )

    components: dict[str, float] = {}

    # Speech rate
    components["speech_rate"] = _normalise(
        sm.words_per_minute, *ranges["speech_rate"]
    )

    # Vocabulary level — use bucketed vocab_score when available,
    # fall back to pct_outside_top_5k for old cached data.
    vocab_raw = sm.vocab_score if sm.vocab_score > 0 else sm.pct_outside_top_5k
    components["vocabulary_level"] = _normalise(
        vocab_raw, *ranges["vocabulary_level"]
    )

    # Lexical diversity
    components["lexical_diversity"] = _normalise(
        sm.lexical_diversity, *ranges["lexical_diversity"]
    )

    # --- Short-transcript lexical diversity dampener ---
    # MATTR is less reliable on very short texts (total_words close to
    # the MATTR window size), producing inflated diversity values.
    # Blend toward a neutral midpoint when the transcript is short.
    _LEXDIV_RELIABLE_WORDS = 1000  # ~5× MATTR window → stable estimate
    _LEXDIV_MIN_WORDS = 200        # MATTR window size (minimum for any value)
    _NEUTRAL = 0.5
    if sm.total_words < _LEXDIV_RELIABLE_WORDS:
        lexdiv_confidence = max(0.0, min(1.0,
            (sm.total_words - _LEXDIV_MIN_WORDS)
            / (_LEXDIV_RELIABLE_WORDS - _LEXDIV_MIN_WORDS)
        ))
        original_lex = components["lexical_diversity"]
        components["lexical_diversity"] = round(
            lexdiv_confidence * original_lex
            + (1.0 - lexdiv_confidence) * _NEUTRAL,
            4,
        )
        logger.debug(
            "Short-text LEX dampener: words=%d, confidence=%.2f, "
            "LEX %.3f → %.3f",
            sm.total_words, lexdiv_confidence, original_lex,
            components["lexical_diversity"],
        )

    # Sentence length
    components["sentence_length"] = _normalise(
        sm.avg_sentence_length, *ranges["sentence_length"]
    )

    # Grammar complexity (parse depth)
    components["grammar_complexity"] = _normalise(
        sm.avg_parse_depth, *ranges["grammar_complexity"]
    )

    # Tense complexity (weighted difficulty of verb tenses used)
    components["tense_complexity"] = _normalise(
        sm.tense_complexity, *ranges["tense_complexity"]
    )

    # --- Run-on sentence detection ---
    # When Whisper omits punctuation the transcript becomes one giant
    # "sentence", inflating both avg_sentence_length AND avg_parse_depth
    # (spaCy's tree depth explodes without sentence breaks).  We compute
    # a confidence factor (0-1) indicating whether sentence boundaries
    # are trustworthy, using two complementary signals:
    #
    #   1. punctuation_density (sentence-ending-punct / total-words).
    #      Well-punctuated speech has density >= 0.03.
    #   2. avg_sentence_length itself — when density is unavailable
    #      (old cached data defaults to 0.0) we use the sentence length
    #      as a fallback: lengths <= 30 are plausible, >= 80 are clearly
    #      run-on text.
    #
    # The confidence factor is used by compute_podcast_score to dampen
    # the weights of both sentence_length and grammar_complexity.
    _PUNCT_DENSITY_OK = 0.03   # density at or above this → full confidence
    _PUNCT_DENSITY_MIN = 0.005 # density at or below this → check fallback
    _SENT_LEN_OK = 30.0        # avg words/sentence considered plausible
    _SENT_LEN_RUNON = 80.0     # avg words/sentence clearly run-on

    if sm.punctuation_density >= _PUNCT_DENSITY_OK:
        sentence_confidence = 1.0
    elif sm.punctuation_density > _PUNCT_DENSITY_MIN:
        # Some punctuation, but sparse — interpolate
        sentence_confidence = (
            (sm.punctuation_density - _PUNCT_DENSITY_MIN)
            / (_PUNCT_DENSITY_OK - _PUNCT_DENSITY_MIN)
        )
    else:
        # punctuation_density <= 0.005 — either truly unpunctuated or
        # old cached data that never computed the field (defaults to 0).
        # Fall back to avg_sentence_length as a secondary heuristic.
        if sm.avg_sentence_length <= _SENT_LEN_OK:
            sentence_confidence = 1.0
        elif sm.avg_sentence_length >= _SENT_LEN_RUNON:
            sentence_confidence = 0.0
        else:
            sentence_confidence = 1.0 - (
                (sm.avg_sentence_length - _SENT_LEN_OK)
                / (_SENT_LEN_RUNON - _SENT_LEN_OK)
            )

    components["_sentence_confidence"] = round(sentence_confidence, 4)

    # When confidence is low, blend sentence_length and grammar_complexity
    # toward a neutral midpoint (0.5).  This prevents inflated run-on
    # values from polluting the podcast-level average.  At confidence=0
    # the component becomes exactly 0.5 ("unknown"); at confidence=1 the
    # original value is kept.
    if sentence_confidence < 1.0:
        for key in ("sentence_length", "grammar_complexity"):
            original = components[key]
            components[key] = round(
                sentence_confidence * original + (1.0 - sentence_confidence) * _NEUTRAL,
                4,
            )
        logger.info(
            "Run-on detected (punct_density=%.4f, avg_sent_len=%.1f) "
            "– sentence boundary confidence reduced to %.2f",
            sm.punctuation_density,
            sm.avg_sentence_length,
            sentence_confidence,
        )

    # Slang (from LLM)
    components["slang_score"] = _normalise(
        llm.slang_score, *ranges["slang_score"]
    )

    # Topic complexity (from LLM)
    components["topic_complexity"] = _normalise(
        llm.topic_complexity, *ranges["topic_complexity"]
    )

    # Clarity (word-level composite: higher = harder to understand)
    # Use clarity_score when available, fall back to inverted avg_segment_confidence
    if sm.clarity_score > 0:
        clarity_raw = sm.clarity_score
    elif sm.avg_segment_confidence != 0:
        # Legacy fallback: invert so higher = worse clarity
        # Map old range roughly: -0.05 (clear) → ~0.01, -0.30 (bad) → ~0.10
        clarity_raw = max(0.0, -sm.avg_segment_confidence - 0.04) * 0.4
    else:
        clarity_raw = 0.05  # neutral default
    components["clarity"] = _normalise(clarity_raw, *ranges["clarity"])

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
    *,
    norm_ranges: dict[str, tuple[float, float]] | None = None,
    weights_override: dict[str, float] | None = None,
) -> DifficultyScore:
    """Aggregate episode-level analyses into a single podcast difficulty score."""

    if not episode_analyses:
        return DifficultyScore(
            podcast_title=podcast_title,
            feed_url=feed_url,
        )

    # --- Deduplicate episodes ---
    # The same episode URL can appear more than once when the feed has
    # different entries that resolve to the same audio file, or when
    # the same episode was downloaded at different lengths.  Keeping
    # duplicates would bias the average.  We keep the *longest*
    # transcript per URL so the most reliable metrics win.
    seen_urls: dict[str, int] = {}  # url → index of best candidate
    for i, ea in enumerate(episode_analyses):
        url = ea.episode.url
        total_words = (
            ea.structural_metrics.total_words
            if ea.structural_metrics
            else 0
        )
        if url not in seen_urls:
            seen_urls[url] = i
        else:
            prev = episode_analyses[seen_urls[url]]
            prev_words = (
                prev.structural_metrics.total_words
                if prev.structural_metrics
                else 0
            )
            if total_words > prev_words:
                seen_urls[url] = i

    dedup_indices = set(seen_urls.values())
    if len(dedup_indices) < len(episode_analyses):
        dropped = len(episode_analyses) - len(dedup_indices)
        logger.info(
            "Deduplicated %d duplicate episode(s) for '%s' (%d → %d)",
            dropped, podcast_title,
            len(episode_analyses), len(dedup_indices),
        )
    unique_analyses = [
        ea for i, ea in enumerate(episode_analyses) if i in dedup_indices
    ]

    weights = dict(weights_override or _get_weights())

    # If no episodes have LLM analysis, redistribute LLM weights
    # proportionally across the structural components so the score
    # still uses the full 0-1 range.
    has_llm = any(ea.llm_analysis is not None for ea in unique_analyses)
    llm_keys = {"slang_score", "topic_complexity"}
    if not has_llm:
        _redistribute_llm_weights(weights, llm_keys)
        logger.info(
            "No LLM analysis — redistributed weights: %s",
            {k: round(v, 3) for k, v in weights.items() if v > 0},
        )

    # Score each episode
    all_components: list[dict[str, float]] = []
    for ea in unique_analyses:
        comps = score_episode(ea, norm_ranges=norm_ranges)
        all_components.append(comps)

    # --- Exclude severe run-on episodes ---
    # Episodes with _sentence_confidence == 0 (Whisper produced no
    # punctuation at all) have completely unreliable sentence length
    # and grammar metrics.  Their speech-rate and vocabulary are still
    # valid, but the mixed-reliability components pollute the average.
    # When we have enough clean episodes, drop the fully-unpunctuated
    # ones entirely rather than blending their values to 0.5.
    _MIN_CLEAN_EPISODES = 3
    excluded_runon_indices: set[int] = set()
    clean_count = sum(
        1 for c in all_components if c.get("_sentence_confidence", 1.0) > 0
    )
    if clean_count >= _MIN_CLEAN_EPISODES:
        for i, comps in enumerate(all_components):
            if comps.get("_sentence_confidence", 1.0) == 0.0:
                excluded_runon_indices.add(i)
                logger.info(
                    "Excluded run-on episode: '%s' (sentence_confidence=0)",
                    unique_analyses[i].episode.title,
                )
    scoring_components = [
        c for i, c in enumerate(all_components) if i not in excluded_runon_indices
    ]
    scoring_analyses = [
        ea for i, ea in enumerate(unique_analyses) if i not in excluded_runon_indices
    ]

    # Outlier trimming: drop the N highest-scoring episodes to prevent
    # a single atypical episode from skewing the podcast average.
    trimmed_indices: list[int] = []
    trim_count = getattr(config, "OUTLIER_TRIM_COUNT", 1)
    trim_min = getattr(config, "OUTLIER_TRIM_MIN_EPISODES", 4)
    if trim_count > 0 and len(scoring_components) >= trim_min:
        # Compute a quick weighted score per episode for ranking
        episode_scores = []
        for i, comps in enumerate(scoring_components):
            s = sum(comps.get(k, 0.0) * weights[k] for k in weights)
            episode_scores.append((s, i))
        episode_scores.sort(reverse=True)
        # Trim the top N outliers
        for _, idx in episode_scores[:trim_count]:
            trimmed_indices.append(idx)
            title = scoring_analyses[idx].episode.title
            logger.info(
                "Trimmed outlier episode: '%s' (score=%.3f)",
                title, episode_scores[0][0],
            )

    # Build the kept list
    kept_components = [
        c for i, c in enumerate(scoring_components) if i not in trimmed_indices
    ]
    kept_analyses = [
        ea for i, ea in enumerate(scoring_analyses) if i not in trimmed_indices
    ]

    # Average each component across kept episodes
    avg_components: dict[str, float] = {}
    for key in weights:
        if weights[key] == 0.0:
            continue
        vals = [c.get(key, 0.0) for c in kept_components]
        avg_components[key] = round(float(np.mean(vals)), 4)

    # --- Sentence-boundary dampening ---
    # When punctuation is unreliable (low _sentence_confidence), reduce
    # the effective weight of sentence_length AND grammar_complexity
    # (parse-tree depth is also distorted by run-on text) and
    # redistribute to the remaining components.
    sent_confs = [c.get("_sentence_confidence", 1.0) for c in kept_components]
    avg_sent_conf = float(np.mean(sent_confs)) if sent_confs else 1.0
    dampened_keys = ["sentence_length", "grammar_complexity"]
    if avg_sent_conf < 1.0:
        total_freed = 0.0
        for dk in dampened_keys:
            orig_w = weights.get(dk, 0)
            if orig_w <= 0:
                continue
            new_w = orig_w * avg_sent_conf
            total_freed += orig_w - new_w
            weights[dk] = new_w

        # Redistribute freed weight proportionally to other active components
        other_keys = [
            k for k in weights
            if k not in dampened_keys and weights[k] > 0
        ]
        other_total = sum(weights[k] for k in other_keys)
        if other_total > 0 and total_freed > 0:
            for k in other_keys:
                weights[k] += total_freed * (weights[k] / other_total)

        logger.info(
            "Sentence-boundary weights dampened (confidence %.2f): %s",
            avg_sent_conf,
            {k: round(weights[k], 4) for k in dampened_keys},
        )

    # Weighted sum
    overall = sum(avg_components.get(k, 0.0) * weights[k] for k in weights)
    overall = round(float(np.clip(overall, 0.0, 1.0)), 4)
    cefr = _cefr_from_score(overall)

    # Also incorporate LLM CEFR estimates if available
    llm_cefrs = [
        ea.llm_analysis.estimated_cefr
        for ea in unique_analyses
        if ea.llm_analysis and ea.llm_analysis.estimated_cefr
    ]
    if llm_cefrs:
        logger.info("LLM CEFR estimates: %s → algorithmic CEFR: %s", llm_cefrs, cefr)

    trimmed_titles = [scoring_analyses[i].episode.title for i in trimmed_indices]
    excluded_titles = [
        unique_analyses[i].episode.title for i in sorted(excluded_runon_indices)
    ]

    result = DifficultyScore(
        podcast_title=podcast_title,
        feed_url=feed_url,
        overall_score=overall,
        cefr_estimate=cefr,
        component_scores=avg_components,
        episodes_analyzed=len(kept_analyses),
        episode_results=episode_analyses,  # keep all (incl. dupes) for the report
        trimmed_episodes=trimmed_titles + excluded_titles,
    )

    trimmed_msg = ""
    info_parts = []
    if trimmed_titles:
        info_parts.append(f"trimmed {len(trimmed_titles)} outlier")
    if excluded_titles:
        info_parts.append(f"excluded {len(excluded_titles)} run-on")
    dup_dropped = len(episode_analyses) - len(unique_analyses)
    if dup_dropped > 0:
        info_parts.append(f"deduped {dup_dropped}")
    if info_parts:
        trimmed_msg = f" ({'; '.join(info_parts)})"
    logger.info(
        "Podcast '%s' → score=%.3f (%s), %d episodes%s",
        podcast_title,
        overall,
        cefr,
        len(kept_analyses),
        trimmed_msg,
    )
    return result


WIDTH = 100
"""Fixed output width shared by format_report and format_ranking."""


def _wrap(text: str, indent: int, width: int = WIDTH) -> list[str]:
    """Word-wrap *text* to *width*, indenting continuation lines."""
    if len(text) <= width:
        return [text]
    prefix = " " * indent
    result: list[str] = []
    while text:
        if len(text) <= width:
            result.append(text)
            break
        # Find last space within width
        cut = text.rfind(" ", 0, width + 1)
        if cut <= indent:
            cut = text.find(" ", indent)
            if cut == -1:
                result.append(text)
                break
        result.append(text[:cut])
        text = prefix + text[cut + 1:]
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

    sep = "=" * WIDTH

    lines = [
        sep,
        f"  Podcast Difficulty Report",
        sep,
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
        title_line = f"  Episode {i}: {ea.episode.title}{trimmed_tag}"
        lines.extend(_wrap(title_line, indent=4))
        if ea.structural_metrics:
            sm = ea.structural_metrics
            vocab_label = f"Vocab: {sm.vocab_score:.3f}" if sm.vocab_score > 0 else f"Outside 5k: {sm.pct_outside_top_5k:.1%}"
            lines.append(
                f"    Words: {sm.total_words}  |  WPM: {sm.words_per_minute}  |  "
                f"Lex div: {sm.lexical_diversity:.3f}  |  {vocab_label}"
            )
        if ea.llm_analysis and ea.llm_analysis.estimated_cefr:
            la = ea.llm_analysis
            lines.append(
                f"    LLM CEFR: {la.estimated_cefr}  |  Slang: {la.slang_score:.2f}  |  "
                f"Topic: {la.topic_complexity:.2f}"
            )
            if la.slang_examples:
                slang_line = f"    Slang examples: {', '.join(la.slang_examples[:5])}"
                lines.extend(_wrap(slang_line, indent=6))
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


def format_ranking(results: list[DifficultyScore]) -> str:
    """Return a concise comparative ranking table.

    Fixed-width (120 cols).  Long podcast titles wrap to a continuation
    line indented under the name column rather than being truncated.
    A weights key is appended below the table.
    """
    ranked = sorted(results, key=lambda r: r.overall_score)
    weights = _get_weights()

    # Detect whether any result has LLM data
    has_llm = any(
        any(ea.llm_analysis is not None for ea in r.episode_results)
        for r in ranked
    )
    if not has_llm:
        # Mirror the redistribution done in compute_podcast_score
        llm_keys = {"slang_score", "topic_complexity"}
        _redistribute_llm_weights(weights, llm_keys)

    # Column definitions: (key, short_header, full_description)
    columns: list[tuple[str, str, str]] = [
        ("speech_rate",       "SPD",  "Speech rate (words per minute)"),
        ("vocabulary_level",  "VOC",  "Vocabulary level (bucketed frequency score)"),
        ("lexical_diversity", "LEX",  "Lexical diversity (MATTR)"),
        ("sentence_length",   "SEN",  "Sentence length (avg words/sentence)"),
        ("grammar_complexity","GRM",  "Grammar complexity (parse depth)"),
        ("tense_complexity", "TNS",  "Tense complexity (verb tense difficulty)"),
        ("slang_score",       "SLG",  "Slang usage (LLM)"),
        ("topic_complexity",  "TOP",  "Topic complexity (LLM)"),
        ("clarity",           "CLR",  "Clarity (word-level Whisper confidence)"),
    ]

    # Filter to active columns (non-zero weight)
    active = [(k, hdr, desc) for k, hdr, desc in columns if weights.get(k, 0) > 0]

    # Fixed layout: #(3) + gap(2) + Name(variable) + gap(2) + Score(5) + gap(2) + CEFR(4) + gap(2) + Ep(2)
    # then each component col = gap(1) + val(5)  →  6 per component
    meta_w = 3 + 2 + 2 + 5 + 2 + 4 + 2 + 2  # 22 chars for #, Score, CEFR, Ep + gaps
    comp_w = len(active) * 6                   # 6 chars per component column
    name_w = WIDTH - meta_w - comp_w           # remainder goes to the name column

    # Build header
    hdr = f"{'#':>3s}  {'Podcast':<{name_w}s}  {'Score':>5s}  {'CEFR':>4s}  {'Ep':>2s}"
    for _, h, _ in active:
        hdr += f" {h:>5s}"
    sep = "-" * WIDTH

    lines: list[str] = [
        "",
        sep,
        "  RANKING  (easiest -> hardest)",
        sep,
        hdr,
        sep,
    ]

    indent = 3 + 2  # rank + gap – continuation lines start under the name

    for rank, r in enumerate(ranked, 1):
        cefr = r.cefr_estimate or "??"
        title = r.podcast_title

        # Build the data suffix (Score + CEFR + Ep + components)
        suffix = f"  {r.overall_score:5.3f}  {cefr:>4s}  {r.episodes_analyzed:>2d}"
        for key, _, _ in active:
            val = r.component_scores.get(key, 0.0)
            suffix += f" {val:5.3f}"

        if len(title) <= name_w:
            # Fits on one line
            lines.append(f"{rank:>3d}  {title:<{name_w}s}{suffix}")
        else:
            # Word-wrap the title within name_w, then attach the data suffix
            # to the first chunk and indent continuations under the name column.
            chunks: list[str] = []
            remaining = title
            while remaining:
                if len(remaining) <= name_w:
                    chunks.append(remaining)
                    break
                cut = remaining.rfind(" ", 0, name_w + 1)
                if cut <= 0:
                    cut = remaining.find(" ")
                    if cut == -1:
                        chunks.append(remaining)
                        break
                chunks.append(remaining[:cut])
                remaining = remaining[cut + 1:]

            lines.append(f"{rank:>3d}  {chunks[0]:<{name_w}s}{suffix}")
            for chunk in chunks[1:]:
                lines.append(f"{'':<{indent}s}{chunk}")

    lines.append(sep)

    # Weight key
    lines.append("")
    lines.append("  WEIGHTS KEY  (component scores are 0-1; higher = harder)")
    lines.append(f"  {'Col':>5s}  {'Wt':>5s}  Description")
    lines.append(f"  {'---':>5s}  {'---':>5s}  {'-' * 40}")
    for key, hdr, desc in active:
        w = weights[key]
        lines.append(f"  {hdr:>5s}  {w:5.2f}  {desc}")
    lines.append(f"  {'':>5s}  {'-----':>5s}")
    lines.append(f"  {'':>5s}  {sum(weights[k] for k, _, _ in active):5.2f}  TOTAL")

    if not has_llm:
        lines.append("")
        lines.append("  * LLM components excluded -- weights redistributed to structural metrics.")
        lines.append("    Use --use-llm or --local-llm for full analysis.")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)
