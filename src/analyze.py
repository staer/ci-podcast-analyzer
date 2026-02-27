"""Structural NLP analysis: speech rate, vocabulary, grammar complexity."""

from __future__ import annotations

import logging
import re
from collections import Counter

import spacy
from wordfreq import top_n_list

import config
from src.models import StructuralMetrics, Transcription

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded resources
# ---------------------------------------------------------------------------
_nlp: spacy.language.Language | None = None
_freq_lists: dict[int, set[str]] | None = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy model '%s'...", config.SPACY_MODEL)
        _nlp = spacy.load(config.SPACY_MODEL)
    return _nlp


def _get_frequency_lists() -> dict[int, set[str]]:
    """Build sets of the top 1k, 5k, and 10k Spanish words."""
    global _freq_lists
    if _freq_lists is None:
        logger.info("Building Spanish word frequency lists...")
        _freq_lists = {
            1_000: set(top_n_list("es", 1_000)),
            5_000: set(top_n_list("es", 5_000)),
            10_000: set(top_n_list("es", 10_000)),
        }
    return _freq_lists


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_structure(transcript: Transcription) -> StructuralMetrics:
    """Run structural / NLP analysis on a transcript.

    Computes: speech rate, vocabulary level, lexical diversity,
    sentence length, grammar complexity, and clarity proxy.
    """
    text = transcript.full_text
    if not text:
        logger.warning("Empty transcript for '%s'", transcript.episode_title)
        return StructuralMetrics()

    nlp = _get_nlp()
    # Process in chunks if the text is very long (spaCy has a max_length)
    if len(text) > nlp.max_length:
        text = text[: nlp.max_length]

    doc = nlp(text)

    # --- Basic counts ---
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    words = [t.text.lower() for t in tokens]
    total_words = len(words)

    if total_words < config.MIN_TRANSCRIPT_WORDS:
        logger.warning(
            "Transcript too short (%d words) for reliable analysis", total_words
        )

    # Lexical diversity: use Moving-Average TTR (MATTR) to avoid
    # inflated scores on short transcripts.  MATTR computes TTR over a
    # sliding window of fixed size and averages the results, making the
    # metric comparable across transcripts of different lengths.
    common_words = [t.text.lower() for t in tokens if t.pos_ != "PROPN"]
    lexical_diversity = _mattr(common_words, window=config.MATTR_WINDOW_SIZE)
    unique_words = len(set(words))

    # --- Speech rate ---
    duration_min = transcript.duration_seconds / 60.0 if transcript.duration_seconds else 1.0
    words_per_minute = total_words / duration_min

    # --- Sentence-level ---
    sentences = list(doc.sents)
    sent_lengths = [
        len([t for t in sent if not t.is_punct and not t.is_space])
        for sent in sentences
    ]
    avg_sentence_length = (
        sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0.0
    )

    # Punctuation density: ratio of sentence-ending punctuation marks
    # (. ? ! … ;) to total words.  A well-punctuated transcript has
    # roughly one terminator per 8-20 words (~0.05-0.12).  When Whisper
    # omits punctuation the ratio drops near zero, making sentence-
    # boundary metrics unreliable.
    _sent_end_punct = {".", "?", "!", "…", ";", "。", "？", "！"}
    punct_count = sum(1 for t in doc if t.text in _sent_end_punct)
    punctuation_density = punct_count / total_words if total_words else 0.0

    # Average word length (characters)
    avg_word_length = (
        sum(len(w) for w in words) / total_words if total_words else 0.0
    )

    # --- Vocabulary level ---
    # Exclude proper nouns (PROPN): names, places, brands inflate rare-word
    # counts without reflecting actual podcast difficulty for learners.
    freq = _get_frequency_lists()
    vocab_words = [t.text.lower() for t in tokens if t.pos_ != "PROPN" and t.text.isalpha()]
    total_vocab = len(vocab_words) or 1
    outside_1k = sum(1 for w in vocab_words if w not in freq[1_000])
    outside_5k = sum(1 for w in vocab_words if w not in freq[5_000])
    outside_10k = sum(1 for w in vocab_words if w not in freq[10_000])

    # --- Grammar complexity ---
    parse_depths = [_tree_depth(sent.root) for sent in sentences]
    avg_parse_depth = (
        sum(parse_depths) / len(parse_depths) if parse_depths else 0.0
    )

    # Subjunctive detection: spaCy Spanish models tag mood in morph
    verb_tokens = [t for t in doc if t.pos_ == "VERB"]
    subjunctive_count = sum(
        1 for t in verb_tokens if "Sub" in t.morph.get("Mood", [])
    )
    subjunctive_ratio = (
        subjunctive_count / len(verb_tokens) if verb_tokens else 0.0
    )

    # Subordinate clause ratio (dependency labels: advcl, acl, ccomp, xcomp, csubj)
    subordinating_deps = {"advcl", "acl", "ccomp", "xcomp", "csubj"}
    sub_clause_count = sum(1 for t in doc if t.dep_ in subordinating_deps)
    subordinate_clause_ratio = (
        sub_clause_count / len(sentences) if sentences else 0.0
    )

    # --- Clarity proxy from Whisper segment confidence ---
    confidences = [
        seg.avg_log_prob
        for seg in transcript.segments
        if seg.avg_log_prob is not None
    ]
    avg_segment_confidence = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )

    metrics = StructuralMetrics(
        words_per_minute=round(words_per_minute, 1),
        total_words=total_words,
        unique_words=unique_words,
        lexical_diversity=round(lexical_diversity, 4),
        avg_sentence_length=round(avg_sentence_length, 1),
        avg_word_length=round(avg_word_length, 2),
        pct_outside_top_1k=round(outside_1k / total_vocab, 4),
        pct_outside_top_5k=round(outside_5k / total_vocab, 4),
        pct_outside_top_10k=round(outside_10k / total_vocab, 4),
        avg_parse_depth=round(avg_parse_depth, 2),
        subjunctive_ratio=round(subjunctive_ratio, 4),
        subordinate_clause_ratio=round(subordinate_clause_ratio, 4),
        punctuation_density=round(punctuation_density, 4),
        avg_segment_confidence=round(avg_segment_confidence, 4),
    )

    logger.info(
        "Structural analysis: %d words, %.0f wpm, lexdiv=%.3f, "
        "outside_5k=%.1f%%, parse_depth=%.1f",
        total_words,
        words_per_minute,
        lexical_diversity,
        metrics.pct_outside_top_5k * 100,
        avg_parse_depth,
    )
    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mattr(words: list[str], window: int = 200) -> float:
    """Compute Moving-Average Type-Token Ratio.

    Slides a window of *window* words across *words*, computes TTR for
    each position, and returns the mean.  This removes the bias that
    makes plain TTR artificially high on short texts.

    If the text is shorter than *window*, falls back to plain TTR.
    """
    n = len(words)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(words)) / n

    ttr_sum = 0.0
    count = 0
    for i in range(n - window + 1):
        segment = words[i : i + window]
        ttr_sum += len(set(segment)) / window
        count += 1
    return ttr_sum / count


def _tree_depth(token) -> int:
    """Compute the depth of a dependency parse tree rooted at *token*."""
    if not list(token.children):
        return 1
    return 1 + max(_tree_depth(c) for c in token.children)
