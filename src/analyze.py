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
# Tense difficulty tiers – linguistic difficulty weights (0 = easiest)
# ---------------------------------------------------------------------------
TENSE_DIFFICULTY: dict[str, float] = {
    # A1 – present indicative
    "presente_indicativo": 0.05,
    # A2 – basic past tenses & imperative
    "preterito_indefinido": 0.20,
    "preterito_imperfecto": 0.25,
    "imperativo": 0.20,
    # B1 – future, present perfect, conditional
    "futuro_simple": 0.40,
    "preterito_perfecto": 0.45,
    "condicional": 0.50,
    # B2 – subjunctive present, pluperfect
    "presente_subjuntivo": 0.65,
    "pluscuamperfecto": 0.70,
    # C1 – advanced compound & subjunctive
    "imperfecto_subjuntivo": 0.85,
    "futuro_perfecto": 0.80,
    "condicional_perfecto": 0.85,
    "perfecto_subjuntivo": 0.80,
    "pluscuamperfecto_subjuntivo": 0.95,
    # Catch-all
    "other": 0.30,
}

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
    """Build sets of the top 1k, 2k, 3k, 4k, 5k, and 10k Spanish words."""
    global _freq_lists
    if _freq_lists is None:
        logger.info("Building Spanish word frequency lists...")
        _freq_lists = {
            1_000: set(top_n_list("es", 1_000)),
            2_000: set(top_n_list("es", 2_000)),
            3_000: set(top_n_list("es", 3_000)),
            4_000: set(top_n_list("es", 4_000)),
            5_000: set(top_n_list("es", 5_000)),
            10_000: set(top_n_list("es", 10_000)),
        }
    return _freq_lists


# ---------------------------------------------------------------------------
# Vocabulary difficulty weights per frequency bucket
# ---------------------------------------------------------------------------
VOCAB_BUCKET_DIFFICULTY: dict[str, float] = {
    "top_1k": 0.00,    # Most common words — trivial for any learner
    "1k_2k":  0.10,    # Still very common
    "2k_3k":  0.25,    # Conversational vocabulary
    "3k_4k":  0.45,    # Intermediate vocabulary
    "4k_5k":  0.65,    # Upper-intermediate vocabulary
    "5k_plus": 1.00,   # Rare / advanced vocabulary
}


def _compute_vocab_score(vocab_words: list[str], freq: dict[int, set[str]]) -> tuple[dict[str, float], float]:
    """Bucket words by frequency and compute a weighted difficulty score.

    Returns:
        (vocab_distribution, vocab_score) where distribution maps bucket
        names to their fraction, and vocab_score is a 0–1 composite.
    """
    total = len(vocab_words)
    if total == 0:
        return {}, 0.0

    # Count words in each bucket (mutually exclusive, highest band first)
    buckets: dict[str, int] = {
        "top_1k": 0,
        "1k_2k": 0,
        "2k_3k": 0,
        "3k_4k": 0,
        "4k_5k": 0,
        "5k_plus": 0,
    }
    for w in vocab_words:
        if w in freq[1_000]:
            buckets["top_1k"] += 1
        elif w in freq[2_000]:
            buckets["1k_2k"] += 1
        elif w in freq[3_000]:
            buckets["2k_3k"] += 1
        elif w in freq[4_000]:
            buckets["3k_4k"] += 1
        elif w in freq[5_000]:
            buckets["4k_5k"] += 1
        else:
            buckets["5k_plus"] += 1

    distribution = {k: round(v / total, 4) for k, v in buckets.items()}

    # Weighted average of bucket difficulties
    score = sum(
        (count / total) * VOCAB_BUCKET_DIFFICULTY[bucket]
        for bucket, count in buckets.items()
    )
    return distribution, round(score, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _classify_verb_tense(token) -> str | None:
    """Classify a finite verb token into a Spanish tense category.

    Returns ``None`` for non-finite forms (infinitive, gerund,
    participle) that don't contribute to tense complexity.
    """
    verb_form = token.morph.get("VerbForm", [])
    # Skip non-finite forms
    if "Inf" in verb_form or "Ger" in verb_form or "Part" in verb_form:
        return None

    mood = token.morph.get("Mood", [])
    tense = token.morph.get("Tense", [])

    if "Ind" in mood:
        if "Pres" in tense:
            return "presente_indicativo"
        if "Imp" in tense:
            return "preterito_imperfecto"
        if "Past" in tense:
            return "preterito_indefinido"
        if "Fut" in tense:
            return "futuro_simple"
        return "other"
    if "Sub" in mood:
        if "Pres" in tense:
            return "presente_subjuntivo"
        if "Imp" in tense:
            return "imperfecto_subjuntivo"
        return "other"
    if "Cnd" in mood:
        return "condicional"
    if "Imp" in mood:
        return "imperativo"

    return "other"


def _detect_compound_tenses(doc) -> Counter:
    """Detect compound tenses (haber + past participle).

    Returns a Counter mapping compound-tense names to their counts.
    """
    compounds: Counter = Counter()
    for token in doc:
        # Look for auxiliary "haber" whose head is a past participle
        if token.lemma_ != "haber" or token.pos_ != "AUX":
            continue
        head = token.head
        if "Part" not in head.morph.get("VerbForm", []):
            continue

        mood = token.morph.get("Mood", [])
        tense = token.morph.get("Tense", [])

        if "Ind" in mood:
            if "Pres" in tense:
                compounds["preterito_perfecto"] += 1
            elif "Imp" in tense:
                compounds["pluscuamperfecto"] += 1
            elif "Fut" in tense:
                compounds["futuro_perfecto"] += 1
        elif "Cnd" in mood:
            compounds["condicional_perfecto"] += 1
        elif "Sub" in mood:
            if "Pres" in tense:
                compounds["perfecto_subjuntivo"] += 1
            elif "Imp" in tense:
                compounds["pluscuamperfecto_subjuntivo"] += 1

    return compounds


def _compute_tense_complexity(tense_counts: Counter) -> tuple[dict[str, float], float]:
    """Compute tense distribution and weighted complexity score.

    Returns:
        (tense_distribution, tense_complexity) where distribution maps
        tense names to their fraction of total, and complexity is a
        0-1 weighted difficulty score.
    """
    total = sum(tense_counts.values())
    if total == 0:
        return {}, 0.0

    distribution = {k: round(v / total, 4) for k, v in tense_counts.most_common()}

    # Weighted average of difficulty scores
    complexity = sum(
        (count / total) * TENSE_DIFFICULTY.get(tense, TENSE_DIFFICULTY["other"])
        for tense, count in tense_counts.items()
    )
    return distribution, round(complexity, 4)


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

    # Bucketed vocabulary score — composite difficulty from word frequency tiers
    vocab_distribution, vocab_score = _compute_vocab_score(vocab_words, freq)

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

    # --- Tense analysis ---
    # Classify each finite verb into a tense category, then detect
    # compound tenses (haber + participle) separately.
    simple_tense_counts: Counter = Counter()
    for t in verb_tokens:
        tense_label = _classify_verb_tense(t)
        if tense_label is not None:
            simple_tense_counts[tense_label] += 1

    compound_tense_counts = _detect_compound_tenses(doc)

    # Merge: compound tenses override the simple classification of the
    # auxiliary "haber" (which would appear as presente_indicativo etc.).
    # Subtract the auxiliary counts that were reclassified as compound.
    all_tense_counts = Counter(simple_tense_counts)
    for compound_tense, count in compound_tense_counts.items():
        all_tense_counts[compound_tense] += count
        # The haber auxiliary was already counted as a simple tense
        # (e.g. "he" → presente_indicativo).  Remove those to avoid
        # double-counting.  Determine which simple tense haber would
        # have been classified as from the compound name.
        _COMPOUND_AUX_MAP = {
            "preterito_perfecto": "presente_indicativo",
            "pluscuamperfecto": "preterito_imperfecto",
            "futuro_perfecto": "futuro_simple",
            "condicional_perfecto": "condicional",
            "perfecto_subjuntivo": "presente_subjuntivo",
            "pluscuamperfecto_subjuntivo": "imperfecto_subjuntivo",
        }
        aux_tense = _COMPOUND_AUX_MAP.get(compound_tense)
        if aux_tense and all_tense_counts[aux_tense] >= count:
            all_tense_counts[aux_tense] -= count

    # Remove zero-count entries
    all_tense_counts = Counter({k: v for k, v in all_tense_counts.items() if v > 0})

    tense_distribution, tense_complexity = _compute_tense_complexity(all_tense_counts)

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
        vocab_distribution=vocab_distribution,
        vocab_score=vocab_score,
        avg_parse_depth=round(avg_parse_depth, 2),
        subjunctive_ratio=round(subjunctive_ratio, 4),
        subordinate_clause_ratio=round(subordinate_clause_ratio, 4),
        tense_distribution=tense_distribution,
        tense_complexity=tense_complexity,
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
