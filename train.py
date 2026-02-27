#!/usr/bin/env python
"""train.py — Optimise scoring parameters for podcast difficulty ranking.

Reads cached episode analyses and a labelled set of training feeds
(training-feeds.json) to find norm_ranges and weights that produce
correct ordering with good separation between difficulty groups.

Usage:
    python train.py                   # full optimisation
    python train.py --dry-run         # show current loss without changing anything
    python train.py --iterations 500  # custom iteration budget

After optimisation the best parameters are written to settings.json.
Run ``python main.py --rescore`` to verify the new ranking.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

# ---------------------------------------------------------------------------
# Bootstrap project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from src.cache import scan_cached_analyses, scan_cached_llm_analyses  # noqa: E402
from src.models import Episode, EpisodeAnalysis  # noqa: E402
from src.scoring import compute_podcast_score, reload_settings  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_SETTINGS_PATH = _ROOT / "settings.json"
_TRAINING_PATH = _ROOT / "training-feeds.json"

# ---------------------------------------------------------------------------
# Component keys (order matters — matches parameter vector layout)
# ---------------------------------------------------------------------------
COMPONENTS = [
    "speech_rate",
    "vocabulary_level",
    "lexical_diversity",
    "sentence_length",
    "grammar_complexity",
    "tense_complexity",
    "slang_score",
    "topic_complexity",
    "clarity",
]

# Structural components — the only ones with data when LLM is off.
# These are the weights we actually optimise; LLM weights stay at their
# settings.json values (typically 0.10 each, redistributed at scoring time).
STRUCTURAL = [
    "speech_rate",
    "vocabulary_level",
    "lexical_diversity",
    "sentence_length",
    "grammar_complexity",
    "tense_complexity",
    "clarity",
]

LLM_KEYS = {"slang_score", "topic_complexity"}

# Components with domain-known norm ranges that should NOT be optimised.
# These have well-understood real-world bounds:
#   speech_rate: ~60 WPM (very slow learner content) to ~220 WPM (fast native)
#   clarity:     ~0.005 (studio-quality) to ~0.12 (noisy/overlapping speakers)
FIXED_RANGES: dict[str, tuple[float, float]] = {
    "speech_rate": (60.0, 220.0),
    "clarity": (0.005, 0.12),
}

# Components whose norm-ranges ARE part of the optimised parameter vector
OPTIMIZED_COMPONENTS = [k for k in COMPONENTS if k not in FIXED_RANGES]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_config() -> dict:
    """Load training-feeds.json."""
    with open(_TRAINING_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_settings() -> dict:
    """Load current settings.json."""
    with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_settings(settings: dict) -> None:
    """Write settings.json."""
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_episode_analyses(
    grouped: dict[str, list],
    llm_by_ep: dict[str, object],
) -> dict[str, list[EpisodeAnalysis]]:
    """Build EpisodeAnalysis lists from cached data, keyed by feed_url."""
    result: dict[str, list[EpisodeAnalysis]] = {}
    for feed_url, cached_items in grouped.items():
        analyses = []
        for ca in cached_items:
            ea = EpisodeAnalysis(
                episode=Episode(
                    title=ca.episode_title or ca.episode_url,
                    url=ca.episode_url,
                ),
                structural_metrics=ca.structural_metrics,
            )
            if ca.episode_url in llm_by_ep:
                ea.llm_analysis = llm_by_ep[ca.episode_url].llm_analysis
            analyses.append(ea)
        result[feed_url] = analyses
    return result


# ---------------------------------------------------------------------------
# Parameter vector <-> dicts
#
# Layout (N_OPT*2 + N_STRUCT values):
#   [0..2*N_OPT-1]  norm_ranges for OPTIMIZED_COMPONENTS × (lo, hi)
#                    Components in FIXED_RANGES are excluded — their
#                    ranges are injected from domain constants.
#   [2*N_OPT .. ]    raw structural weights — each in [0, 1], normalised
#                    to sum to 1.0 during decoding.  A floor of MIN_WEIGHT
#                    is enforced per component.
#
# LLM weights (slang_score, topic_complexity) are NOT optimised.
# They are read from settings.json and passed through unchanged.
# When LLM data is absent, compute_podcast_score redistributes
# them proportionally to structural weights automatically.
# ---------------------------------------------------------------------------

MIN_WEIGHT = 0.03  # each structural component contributes ≥ 3%
MAX_WEIGHT = 0.50  # no single component may exceed 50%


def params_to_dicts(
    x: np.ndarray,
    llm_weights: dict[str, float] | None = None,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Decode a flat parameter vector into (norm_ranges, weights)."""
    norm_ranges: dict[str, tuple[float, float]] = {}
    n_opt = len(OPTIMIZED_COMPONENTS)
    for i, key in enumerate(OPTIMIZED_COMPONENTS):
        lo = x[2 * i]
        hi = x[2 * i + 1]
        if lo > hi:
            lo, hi = hi, lo
        if hi - lo < 0.01:
            hi = lo + 0.01
        norm_ranges[key] = (float(lo), float(hi))
    # Inject domain-fixed ranges (not optimised)
    norm_ranges.update(FIXED_RANGES)

    # Determine the weight budget left for structural components
    llm_total = 0.0
    if llm_weights:
        llm_total = sum(llm_weights.get(k, 0.0) for k in LLM_KEYS)
    structural_budget = 1.0 - llm_total

    # Structural weights: raw values in [0, 1] → clamp, enforce max share,
    # then normalise to sum to structural_budget so total weights = 1.0
    n_struct = len(STRUCTURAL)
    raw_w = x[2 * n_opt : 2 * n_opt + n_struct]
    clamped = np.clip(raw_w, MIN_WEIGHT, MAX_WEIGHT)
    # Normalise to structural_budget (leaves room for LLM weights)
    sw = clamped / clamped.sum() * structural_budget
    # Enforce per-component cap (50% of total = 0.50) after normalisation
    max_abs = MAX_WEIGHT
    capped = False
    for _ in range(5):  # iterate to redistribute overflow
        over = sw > max_abs
        if not over.any():
            break
        excess = (sw[over] - max_abs).sum()
        sw[over] = max_abs
        under = ~over
        if under.any() and sw[under].sum() > 0:
            sw[under] += excess * (sw[under] / sw[under].sum())
        capped = True
    weights: dict[str, float] = {}
    for i, key in enumerate(STRUCTURAL):
        weights[key] = float(sw[i])
    # LLM weights fixed
    if llm_weights:
        for key in LLM_KEYS:
            weights[key] = llm_weights.get(key, 0.0)
    else:
        for key in LLM_KEYS:
            weights[key] = 0.0

    return norm_ranges, weights


def dicts_to_params(
    norm_ranges: dict[str, tuple[float, float]],
    weights: dict[str, float],
) -> np.ndarray:
    """Encode current dicts into a flat parameter vector."""
    n_opt = len(OPTIMIZED_COMPONENTS)
    n_struct = len(STRUCTURAL)
    x = np.zeros(2 * n_opt + n_struct)
    for i, key in enumerate(OPTIMIZED_COMPONENTS):
        lo, hi = norm_ranges.get(key, (0.0, 1.0))
        x[2 * i] = lo
        x[2 * i + 1] = hi

    # Structural weights: store raw (un-normalised) values
    for i, key in enumerate(STRUCTURAL):
        x[2 * n_opt + i] = np.clip(weights.get(key, 1.0 / n_struct),
                             MIN_WEIGHT, MAX_WEIGHT)
    return x


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def compute_loss(
    x: np.ndarray,
    *,
    training_cfg: dict,
    feed_analyses: dict[str, list[EpisodeAnalysis]],
    llm_weights: dict[str, float] | None = None,
) -> float:
    """Return a scalar loss for the parameter vector *x*.

    Loss components:
        1. **Ordering violations** — pairwise penalty when easier ≥ harder.
        2. **Target-range deviation** — podcasts should land in their group's
           target score range.
        3. **Separation bonus** — reward gaps between group averages.
        4. **Within-group spread** — penalise large spread within a group.
        5. **Weight concentration** — prevent single-weight dominance.
        6. **Minimum weight** — each structural component ≥ 2%.
        7. **Narrow range** — prevent overfitting via collapsed ranges.
    """
    norm_ranges, weights = params_to_dicts(x, llm_weights=llm_weights)

    group_order = training_cfg["group_order"]
    target_ranges = training_cfg["target_ranges"]
    feeds_by_group: dict[str, list[dict]] = {}
    for feed in training_cfg["training_feeds"]:
        feeds_by_group.setdefault(feed["group"], []).append(feed)

    # Score each training podcast
    scores_by_group: dict[str, list[float]] = {}
    for group in group_order:
        group_scores = []
        for feed in feeds_by_group.get(group, []):
            url = feed["url"]
            analyses = feed_analyses.get(url, [])
            if not analyses:
                continue
            title = feed["name"]
            result = compute_podcast_score(
                title, url, analyses,
                norm_ranges=norm_ranges,
                weights_override=weights,
            )
            group_scores.append(result.overall_score)
        scores_by_group[group] = group_scores

    loss = 0.0

    # --- 1. Ordering violations ---
    # Every pair of groups in order should have all podcasts in the
    # easier group scoring below all podcasts in the harder group.
    ORDER_PENALTY = 50.0
    for i in range(len(group_order)):
        for j in range(i + 1, len(group_order)):
            easy_group = group_order[i]
            hard_group = group_order[j]
            for se in scores_by_group.get(easy_group, []):
                for sh in scores_by_group.get(hard_group, []):
                    if se >= sh:
                        # Violation: easy scored as hard (or equal)
                        loss += ORDER_PENALTY * (1.0 + se - sh)

    # --- 2. Target-range deviation ---
    RANGE_PENALTY = 10.0
    for group in group_order:
        lo_target, hi_target = target_ranges[group]
        for s in scores_by_group.get(group, []):
            if s < lo_target:
                loss += RANGE_PENALTY * (lo_target - s) ** 2
            elif s > hi_target:
                loss += RANGE_PENALTY * (s - hi_target) ** 2

    # --- 3. Separation bonus ---
    # Reward distance between consecutive group means.
    SEPARATION_WEIGHT = 5.0
    group_means = []
    for group in group_order:
        gs = scores_by_group.get(group, [])
        if gs:
            group_means.append(np.mean(gs))
        else:
            group_means.append(0.5)
    for k in range(len(group_means) - 1):
        gap = group_means[k + 1] - group_means[k]
        if gap > 0:
            # Reward larger gaps (negative loss contribution)
            loss -= SEPARATION_WEIGHT * gap
        else:
            # Gap is negative — ordering violation already penalised above,
            # but add a small extra nudge.
            loss += SEPARATION_WEIGHT * abs(gap)

    # --- 4. Within-group spread penalty ---
    # Penalise large spread within a group (podcasts in the same group
    # should score similarly).
    SPREAD_PENALTY = 3.0
    for group in group_order:
        gs = scores_by_group.get(group, [])
        if len(gs) >= 2:
            loss += SPREAD_PENALTY * (max(gs) - min(gs)) ** 2

    # --- 5. Weight concentration penalty ---
    # Entropy bonus — prefer more balanced structural weights.
    sw_arr = np.array([weights[k] for k in STRUCTURAL])
    entropy = -np.sum(sw_arr * np.log(sw_arr + 1e-10))
    max_entropy = np.log(len(STRUCTURAL))
    loss -= 1.0 * (entropy / max_entropy)  # reward uniformity

    # --- 7. Norm-range width penalty ---
    # Prevent the optimizer from collapsing ranges to near-zero width,
    # which overfits to the training set.  Each range should maintain
    # a minimum relative width compared to the component_bounds span.
    MIN_RANGE_FRAC = 0.30  # range must be at least 30% of the feasible span
    component_spans = {
        "vocabulary_level":  0.40,
        "lexical_diversity": 0.75,
        "sentence_length":   59.0,
        "grammar_complexity": 11.0,
        "tense_complexity":  0.60,
        "slang_score":       1.0,
        "topic_complexity":  1.0,
    }
    NARROW_PENALTY = 12.0
    for key in OPTIMIZED_COMPONENTS:
        lo, hi = norm_ranges.get(key, (0, 1))
        span = hi - lo
        min_span = component_spans.get(key, 1.0) * MIN_RANGE_FRAC
        if span < min_span:
            loss += NARROW_PENALTY * ((min_span - span) / min_span) ** 2

    # --- 8. Floor-too-high penalty ---
    # Prevent the optimizer from pushing the lower bound so high that
    # most podcasts clamp to 0.  The lo value should stay in the lower
    # half of the feasible range for each component.
    component_bounds_map = {
        "vocabulary_level":  (0.0, 0.40),
        "lexical_diversity": (0.15, 0.90),
        "sentence_length":   (1.0, 60.0),
        "grammar_complexity": (1.0, 12.0),
        "tense_complexity":  (0.0, 0.60),
        "slang_score":       (0.0, 1.0),
        "topic_complexity":  (0.0, 1.0),
    }
    FLOOR_PENALTY = 6.0
    for key in OPTIMIZED_COMPONENTS:
        lo, hi = norm_ranges.get(key, (0, 1))
        cb_lo, cb_hi = component_bounds_map.get(key, (0, 1))
        midpoint = (cb_lo + cb_hi) / 2.0
        if lo > midpoint:
            # Penalise how far the floor has drifted above the midpoint
            overshoot = (lo - midpoint) / (cb_hi - cb_lo)
            loss += FLOOR_PENALTY * overshoot ** 2

    return float(loss)


def compute_loss_breakdown(
    x: np.ndarray,
    *,
    training_cfg: dict,
    feed_analyses: dict[str, list],
    llm_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return individual loss components for diagnostics.

    Same logic as compute_loss but tracks each penalty separately.
    """
    norm_ranges, weights = params_to_dicts(x, llm_weights=llm_weights)

    group_order = training_cfg["group_order"]
    target_ranges = training_cfg["target_ranges"]
    feeds_by_group: dict[str, list[dict]] = {}
    for feed in training_cfg["training_feeds"]:
        feeds_by_group.setdefault(feed["group"], []).append(feed)

    scores_by_group: dict[str, list[float]] = {}
    for group in group_order:
        group_scores = []
        for feed in feeds_by_group.get(group, []):
            url = feed["url"]
            analyses = feed_analyses.get(url, [])
            if not analyses:
                continue
            result = compute_podcast_score(
                feed["name"], url, analyses,
                norm_ranges=norm_ranges, weights_override=weights,
            )
            group_scores.append(result.overall_score)
        scores_by_group[group] = group_scores

    parts: dict[str, float] = {}

    # 1. Ordering violations
    ORDER_PENALTY = 50.0
    ordering = 0.0
    for i in range(len(group_order)):
        for j in range(i + 1, len(group_order)):
            for se in scores_by_group.get(group_order[i], []):
                for sh in scores_by_group.get(group_order[j], []):
                    if se >= sh:
                        ordering += ORDER_PENALTY * (1.0 + se - sh)
    parts["ordering_violations"] = ordering

    # 2. Target-range deviation
    RANGE_PENALTY = 10.0
    range_dev = 0.0
    for group in group_order:
        lo_t, hi_t = target_ranges[group]
        for s in scores_by_group.get(group, []):
            if s < lo_t:
                range_dev += RANGE_PENALTY * (lo_t - s) ** 2
            elif s > hi_t:
                range_dev += RANGE_PENALTY * (s - hi_t) ** 2
    parts["target_range_deviation"] = range_dev

    # 3. Separation bonus (negative = good)
    SEPARATION_WEIGHT = 5.0
    sep = 0.0
    group_means = []
    for group in group_order:
        gs = scores_by_group.get(group, [])
        group_means.append(np.mean(gs) if gs else 0.5)
    for k in range(len(group_means) - 1):
        gap = group_means[k + 1] - group_means[k]
        if gap > 0:
            sep -= SEPARATION_WEIGHT * gap
        else:
            sep += SEPARATION_WEIGHT * abs(gap)
    parts["separation_bonus"] = sep

    # 4. Within-group spread
    SPREAD_PENALTY = 3.0
    spread = 0.0
    for group in group_order:
        gs = scores_by_group.get(group, [])
        if len(gs) >= 2:
            spread += SPREAD_PENALTY * (max(gs) - min(gs)) ** 2
    parts["within_group_spread"] = spread

    # 5. Weight entropy bonus (negative = good)
    sw_arr = np.array([weights[k] for k in STRUCTURAL])
    entropy = -np.sum(sw_arr * np.log(sw_arr + 1e-10))
    max_entropy = np.log(len(STRUCTURAL))
    parts["weight_entropy_bonus"] = -1.0 * (entropy / max_entropy)

    # 7. Narrow range penalty
    MIN_RANGE_FRAC = 0.30
    component_spans = {
        "vocabulary_level": 0.40,
        "lexical_diversity": 0.75, "sentence_length": 59.0,
        "grammar_complexity": 11.0, "tense_complexity": 0.60,
        "slang_score": 1.0, "topic_complexity": 1.0,
    }
    narrow = 0.0
    for key in OPTIMIZED_COMPONENTS:
        lo, hi = norm_ranges.get(key, (0, 1))
        span = hi - lo
        min_span = component_spans.get(key, 1.0) * MIN_RANGE_FRAC
        if span < min_span:
            narrow += 12.0 * ((min_span - span) / min_span) ** 2
    parts["narrow_range_penalty"] = narrow

    # 8. Floor-too-high penalty
    component_bounds_map = {
        "vocabulary_level": (0.0, 0.40),
        "lexical_diversity": (0.15, 0.90), "sentence_length": (1.0, 60.0),
        "grammar_complexity": (1.0, 12.0), "tense_complexity": (0.0, 0.60),
        "slang_score": (0.0, 1.0), "topic_complexity": (0.0, 1.0),
    }
    floor_pen = 0.0
    for key in OPTIMIZED_COMPONENTS:
        lo, hi = norm_ranges.get(key, (0, 1))
        cb_lo, cb_hi = component_bounds_map.get(key, (0, 1))
        midpoint = (cb_lo + cb_hi) / 2.0
        if lo > midpoint:
            overshoot = (lo - midpoint) / (cb_hi - cb_lo)
            floor_pen += 6.0 * overshoot ** 2
    parts["floor_too_high_penalty"] = floor_pen

    parts["total"] = sum(parts.values())
    return parts


# ---------------------------------------------------------------------------
# Bounds for differential_evolution
# ---------------------------------------------------------------------------

def get_bounds(current_ranges: dict[str, tuple[float, float]]) -> list[tuple[float, float]]:
    """Build bounds for the parameter vector.

    Norm-range values are bounded around physically plausible values.
    Components in FIXED_RANGES are excluded from the vector.
    Structural weight softmax inputs are bounded by MIN/MAX.
    """
    bounds = []

    component_bounds = {
        "vocabulary_level":  (0.0, 0.40),
        "lexical_diversity": (0.15, 0.90),
        "sentence_length":   (1.0, 60.0),
        "grammar_complexity": (1.0, 12.0),
        "tense_complexity":  (0.0, 0.60),
        "slang_score":       (0.0, 1.0),
        "topic_complexity":  (0.0, 1.0),
    }

    for key in OPTIMIZED_COMPONENTS:
        lo_bound, hi_bound = component_bounds[key]
        bounds.append((lo_bound, hi_bound))  # norm_range lo
        bounds.append((lo_bound, hi_bound))  # norm_range hi

    # Raw structural weight: bounded by MIN/MAX
    for _ in STRUCTURAL:
        bounds.append((MIN_WEIGHT, MAX_WEIGHT))

    return bounds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_params(
    label: str,
    norm_ranges: dict[str, tuple[float, float]],
    weights: dict[str, float],
):
    """Pretty-print norm_ranges and weights."""
    print(f"\n  {label}")
    print(f"  {'Component':<25s}  {'Range':>18s}  {'Weight':>7s}")
    print(f"  {'-'*25}  {'-'*18}  {'-'*7}")
    for key in COMPONENTS:
        lo, hi = norm_ranges.get(key, (0, 1))
        w = weights.get(key, 0)
        tag = "  FIXED" if key in FIXED_RANGES else ""
        print(f"  {key:<25s}  ({lo:7.3f}, {hi:7.3f})  {w:7.4f}{tag}")
    print(f"  {'':25s}  {'':18s}  {sum(weights.values()):7.4f}  TOTAL")


def score_training_feeds(
    training_cfg: dict,
    feed_analyses: dict[str, list[EpisodeAnalysis]],
    norm_ranges: dict[str, tuple[float, float]],
    weights: dict[str, float],
) -> list[tuple[str, str, float, str]]:
    """Score all training feeds, returning (name, group, score, cefr)."""
    results = []
    for feed in training_cfg["training_feeds"]:
        url = feed["url"]
        analyses = feed_analyses.get(url, [])
        if not analyses:
            results.append((feed["name"], feed["group"], 0.0, "??"))
            continue
        score = compute_podcast_score(
            feed["name"], url, analyses,
            norm_ranges=norm_ranges,
            weights_override=weights,
        )
        results.append((feed["name"], feed["group"], score.overall_score, score.cefr_estimate))
    return sorted(results, key=lambda r: r[2])


def main():
    parser = argparse.ArgumentParser(
        description="Optimise scoring parameters for podcast difficulty ranking."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show current loss and scores without optimising.",
    )
    parser.add_argument(
        "--iterations", type=int, default=300,
        help="Max iterations for the optimizer (default: 300).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--popsize", type=int, default=20,
        help="Population size multiplier for differential evolution (default: 20).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Load data
    print("\n[*] Loading training configuration...")
    training_cfg = load_training_config()
    print(f"    {len(training_cfg['training_feeds'])} training feeds across "
          f"{len(training_cfg['group_order'])} groups")

    print("[*] Scanning cached analyses...")
    grouped = scan_cached_analyses()
    llm_by_ep = scan_cached_llm_analyses()
    feed_analyses = build_episode_analyses(grouped, llm_by_ep)

    # Check training feeds are in cache
    training_urls = {f["url"] for f in training_cfg["training_feeds"]}
    cached_urls = set(feed_analyses.keys())
    missing = training_urls - cached_urls
    if missing:
        print(f"\n  WARNING: {len(missing)} training feed(s) not found in cache:")
        for url in missing:
            name = next(
                (f["name"] for f in training_cfg["training_feeds"] if f["url"] == url),
                url,
            )
            print(f"    - {name} ({url})")
        print("  Run the full analyzer first to populate the cache.\n")
        if len(missing) == len(training_urls):
            sys.exit(1)

    # Load current settings
    settings = load_settings()
    current_ranges = {
        k: tuple(v) for k, v in settings.get("norm_ranges", {}).items()
    }
    current_weights = settings.get("weights", {})
    # LLM weights are fixed — not optimised
    llm_weights = {k: current_weights.get(k, 0.0) for k in LLM_KEYS}

    # Detect whether any training feed has LLM data
    has_llm = any(
        ea.llm_analysis is not None
        for url in training_urls
        for ea in feed_analyses.get(url, [])
    )
    if not has_llm:
        print("    No LLM data in training feeds — LLM weights fixed, "
              "optimising structural weights only.")

    # Show current state
    print_params("CURRENT PARAMETERS", current_ranges, current_weights)

    x0 = dicts_to_params(current_ranges, current_weights)
    current_loss = compute_loss(
        x0,
        training_cfg=training_cfg,
        feed_analyses=feed_analyses,
        llm_weights=llm_weights,
    )
    print(f"\n  Current loss: {current_loss:.4f}")

    print("\n  Current training-feed scores:")
    current_scores = score_training_feeds(
        training_cfg, feed_analyses, current_ranges, current_weights,
    )
    for name, group, score, cefr in current_scores:
        print(f"    {name:40s}  {score:.4f}  {cefr}  [{group}]")

    if args.dry_run:
        print("\n[*] Dry run — no optimisation performed.\n")
        return

    # --- Run optimisation ---
    print(f"\n[*] Starting optimisation (maxiter={args.iterations}, "
          f"popsize={args.popsize}, seed={args.seed})...")

    print("""
    METRICS GUIDE
    ─────────────
    loss        A single number the optimizer tries to MINIMIZE (more negative
                = better).  It combines: ordering penalties (easier podcasts
                must score below harder ones), target-range fit (each group's
                score should land in its target band), group separation
                (reward gaps between tiers), weight-balance incentives, and
                norm-range health checks.  Typical good values are –2 to –4.

    convergence Fraction (0→1) indicating how similar all candidate solutions
                in the population have become.  Starts near 0 (diverse
                population) and approaches 1 as the population converges to
                a single optimum.  It stays at 0.00 for a long time because
                differential_evolution maintains a large diverse population
                (popsize×dimensions candidates).  This is normal — progress
                is tracked via 'loss' improvements instead.
    """)

    bounds = get_bounds(current_ranges)

    # Track progress
    best_loss = [current_loss]
    eval_count = [0]

    def objective(x):
        return compute_loss(
            x, training_cfg=training_cfg, feed_analyses=feed_analyses,
            llm_weights=llm_weights,
        )

    def callback(xk, convergence=0):
        eval_count[0] += 1
        loss = objective(xk)
        if loss < best_loss[0]:
            best_loss[0] = loss
            print(f"    iter {eval_count[0]:>4d}  loss={loss:.4f}  "
                  f"(improvement, convergence={convergence:.4f})")
        elif eval_count[0] % 50 == 0:
            print(f"    iter {eval_count[0]:>4d}  loss={loss:.4f}  "
                  f"convergence={convergence:.4f}")
        return False  # don't stop early

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=args.iterations,
        popsize=args.popsize,
        seed=args.seed,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.8,
        x0=x0,
        callback=callback,
        disp=False,
        polish=True,
    )

    # Decode best parameters
    best_ranges, best_weights = params_to_dicts(result.x, llm_weights=llm_weights)

    print(f"\n[*] Optimisation complete.")
    print(f"    Iterations: {result.nit}")
    print(f"    Final loss: {result.fun:.4f}  (was {current_loss:.4f})")

    # Loss breakdown
    breakdown = compute_loss_breakdown(
        result.x, training_cfg=training_cfg,
        feed_analyses=feed_analyses, llm_weights=llm_weights,
    )
    print("""
  LOSS BREAKDOWN  (positive = penalties, negative = bonuses)
  ─────────────────────────────────────────────────────────""")
    labels = {
        "ordering_violations":   "Ordering violations    (easier must score < harder)",
        "target_range_deviation":"Target-range deviation  (scores within group bands)",
        "separation_bonus":      "Separation bonus        (reward gaps between groups)",
        "within_group_spread":   "Within-group spread     (same-group consistency)",
        "weight_entropy_bonus":  "Weight entropy bonus    (prefer balanced weights)",
        "narrow_range_penalty":  "Narrow-range penalty    (prevent collapsed ranges)",
        "floor_too_high_penalty":"Floor-too-high penalty  (keep lo below midpoint)",
    }
    for key, label in labels.items():
        val = breakdown.get(key, 0.0)
        sign = "+" if val >= 0 else ""
        print(f"    {label}  {sign}{val:>8.4f}")
    print(f"    {'':─<55s}  ────────")
    print(f"    {'TOTAL':<55s}  {breakdown['total']:>8.4f}")

    print_params("OPTIMISED PARAMETERS", best_ranges, best_weights)

    print("\n  Optimised training-feed scores:")
    opt_scores = score_training_feeds(
        training_cfg, feed_analyses, best_ranges, best_weights,
    )
    for name, group, score, cefr in opt_scores:
        print(f"    {name:40s}  {score:.4f}  {cefr}  [{group}]")

    # --- Compare ---
    print("\n  Comparison (current → optimised):")
    current_dict = {name: (score, cefr) for name, _, score, cefr in current_scores}
    for name, group, new_score, new_cefr in opt_scores:
        old_score, old_cefr = current_dict.get(name, (0.0, "??"))
        delta = new_score - old_score
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"    {name:40s}  {old_score:.4f} → {new_score:.4f}  "
              f"({arrow}{abs(delta):.4f})  [{group}]")

    if result.fun >= current_loss:
        print("\n  No improvement found — keeping current settings.\n")
        return

    # Save
    new_settings = copy.deepcopy(settings)
    new_settings["norm_ranges"] = {
        k: [round(lo, 4), round(hi, 4)] for k, (lo, hi) in best_ranges.items()
    }
    new_settings["weights"] = {
        k: round(v, 6) for k, v in best_weights.items()
    }
    save_settings(new_settings)
    print(f"\n[*] Saved optimised parameters to {_SETTINGS_PATH.name}")
    print("    Run  python main.py --rescore  to see the full ranking.\n")


if __name__ == "__main__":
    main()
