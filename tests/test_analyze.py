"""Tests for src/analyze.py – structural NLP metrics.

These tests load the real spaCy model (es_core_news_lg) so they require
it to be installed.  Mark the module as slow / integration if needed.
"""

from __future__ import annotations

import pytest

from src.analyze import (
    TENSE_DIFFICULTY,
    VOCAB_BUCKET_DIFFICULTY,
    _classify_verb_tense,
    _compute_tense_complexity,
    _compute_vocab_score,
    _detect_compound_tenses,
    _mattr,
    _tree_depth,
    analyze_structure,
)
from src.models import Transcription, TranscriptionSegment, WordTimestamp


# ===================================================================
# Helpers
# ===================================================================

def _make_transcription(
    text: str,
    duration_seconds: float = 60.0,
    segments: list[TranscriptionSegment] | None = None,
) -> Transcription:
    """Build a minimal Transcription for testing."""
    if segments is None:
        segments = [
            TranscriptionSegment(text=text, start=0.0, end=duration_seconds)
        ]
    return Transcription(
        episode_title="Test",
        full_text=text,
        segments=segments,
        duration_seconds=duration_seconds,
    )


# ===================================================================
# Speech rate
# ===================================================================

class TestSpeechRate:
    """Words-per-minute calculation."""

    def test_basic_wpm(self):
        # 100 words in 60 seconds → 100 wpm
        words = " ".join(["hola"] * 100)
        t = _make_transcription(words, duration_seconds=60.0)
        m = analyze_structure(t)
        assert m.words_per_minute == pytest.approx(100.0, rel=0.05)

    def test_faster_speech(self):
        # 200 words in 60 seconds → 200 wpm
        words = " ".join(["bueno"] * 200)
        t = _make_transcription(words, duration_seconds=60.0)
        m = analyze_structure(t)
        assert m.words_per_minute == pytest.approx(200.0, rel=0.05)

    def test_half_speed(self):
        # 100 words in 120 seconds → 50 wpm
        words = " ".join(["gracias"] * 100)
        t = _make_transcription(words, duration_seconds=120.0)
        m = analyze_structure(t)
        assert m.words_per_minute == pytest.approx(50.0, rel=0.05)


# ===================================================================
# Lexical diversity
# ===================================================================

class TestLexicalDiversity:
    """MATTR-based lexical diversity."""

    def test_all_same_word(self):
        """Repeating the same word → low diversity."""
        words = " ".join(["casa"] * 200)
        t = _make_transcription(words)
        m = analyze_structure(t)
        assert m.lexical_diversity < 0.05

    def test_all_unique_words(self):
        """Many unique words → high diversity."""
        # Use simple Spanish words that spaCy won't merge
        unique_words = [
            "casa", "perro", "gato", "mesa", "silla", "agua", "sol", "luna",
            "libro", "coche", "tren", "barco", "flor", "arbol", "piedra",
            "fuego", "viento", "nube", "rio", "campo", "playa", "montaña",
            "ciudad", "pueblo", "camino", "puerta", "ventana", "pared",
            "techo", "suelo", "mano", "pie", "ojo", "boca", "nariz",
        ]
        text = " ".join(unique_words)
        t = _make_transcription(text)
        m = analyze_structure(t)
        assert m.lexical_diversity > 0.5

    def test_short_vs_long_not_dramatically_different(self):
        """MATTR should produce comparable values for short and long texts
        with the same vocabulary mix (unlike plain TTR)."""
        # Build a repeating block of mixed words
        block = [
            "el", "perro", "grande", "come", "la", "comida", "en",
            "una", "casa", "bonita", "cerca", "del", "parque",
        ]
        short_text = " ".join(block * 25)   # ~325 words
        long_text = " ".join(block * 150)   # ~1950 words
        t_short = _make_transcription(short_text)
        t_long = _make_transcription(long_text)
        m_short = analyze_structure(t_short)
        m_long = analyze_structure(t_long)
        # With MATTR these should be close; with plain TTR they'd diverge.
        assert abs(m_short.lexical_diversity - m_long.lexical_diversity) < 0.15


# ===================================================================
# Vocabulary level
# ===================================================================

class TestVocabularyLevel:
    """Fraction of words outside frequency lists."""

    def test_common_words_mostly_in_top_1k(self):
        """Very common words should have low pct_outside_top_1k."""
        text = "yo tengo una casa grande y bonita en la ciudad"
        text = " ".join([text] * 20)  # Repeat for enough words
        t = _make_transcription(text)
        m = analyze_structure(t)
        # Most of these should be in the top 1k
        assert m.pct_outside_top_1k < 0.4

    def test_rare_words_outside_lists(self):
        """Rare/technical words should have high pct_outside."""
        rare = "electroencefalografista otorrinolaringologista"
        text = " ".join([rare] * 50)
        t = _make_transcription(text)
        m = analyze_structure(t)
        assert m.pct_outside_top_5k > 0.5

    def test_ordering_1k_5k_10k(self):
        """Outside-1k >= outside-5k >= outside-10k always."""
        text = "el presidente anuncio una nueva politica economica para el desarrollo sostenible del pais"
        text = " ".join([text] * 15)
        t = _make_transcription(text)
        m = analyze_structure(t)
        assert m.pct_outside_top_1k >= m.pct_outside_top_5k
        assert m.pct_outside_top_5k >= m.pct_outside_top_10k


# ===================================================================
# Grammar complexity (parse depth)
# ===================================================================

class TestGrammarComplexity:
    """Parse depth and clause detection."""

    def test_simple_sentence_shallow(self):
        """Short simple sentences → lower parse depth."""
        text = "Yo como pan. El gato duerme. Ella canta bien."
        text = " ".join([text] * 30)
        t = _make_transcription(text)
        m = analyze_structure(t)
        assert m.avg_parse_depth < 5.0

    def test_complex_sentence_deeper(self):
        """Complex nested sentence → higher parse depth."""
        text = (
            "El hombre que conocio a la mujer que trabajaba en la empresa "
            "que estaba cerca del parque que visitamos ayer dijo que vendria "
            "cuando terminara el proyecto que estaba desarrollando."
        )
        text = " ".join([text] * 10)
        t = _make_transcription(text)
        m = analyze_structure(t)
        # Should be deeper than simple sentences
        assert m.avg_parse_depth > 3.0

    def test_parse_depth_positive(self):
        """Parse depth should always be positive when there's text."""
        t = _make_transcription("Hola mundo. Buenos dias.")
        m = analyze_structure(t)
        assert m.avg_parse_depth > 0.0


# ===================================================================
# Sentence length
# ===================================================================

class TestSentenceLength:
    """Average sentence length in words."""

    def test_short_sentences(self):
        text = "Hola. Si. No. Bien. Gracias."
        text = " ".join([text] * 30)
        t = _make_transcription(text)
        m = analyze_structure(t)
        assert m.avg_sentence_length < 5.0

    def test_long_sentences(self):
        text = (
            "Esta es una oracion bastante larga que contiene muchas palabras "
            "diferentes para probar que el sistema puede detectar las oraciones "
            "largas correctamente en el texto."
        )
        text = " ".join([text] * 10)
        t = _make_transcription(text)
        m = analyze_structure(t)
        assert m.avg_sentence_length > 10.0


# ===================================================================
# Segment confidence (clarity proxy)
# ===================================================================

class TestSegmentConfidence:
    """avg_segment_confidence from whisper log probs."""

    def test_confidence_averaged(self):
        segs = [
            TranscriptionSegment(text="uno", start=0, end=1, avg_log_prob=-0.2),
            TranscriptionSegment(text="dos", start=1, end=2, avg_log_prob=-0.8),
        ]
        t = _make_transcription(
            "uno dos",
            duration_seconds=2.0,
            segments=segs,
        )
        m = analyze_structure(t)
        assert m.avg_segment_confidence == pytest.approx(-0.5, abs=0.01)

    def test_no_confidence_defaults_zero(self):
        segs = [
            TranscriptionSegment(text="hola", start=0, end=1, avg_log_prob=None),
        ]
        t = _make_transcription("hola", segments=segs)
        m = analyze_structure(t)
        assert m.avg_segment_confidence == 0.0


# ===================================================================
# Word-level clarity composite
# ===================================================================

class TestClarityComposite:
    """Tests for the word-level clarity_score composite metric."""

    def _make_word(self, word: str, prob: float) -> WordTimestamp:
        return WordTimestamp(word=word, start=0.0, end=0.1, probability=prob)

    def test_clarity_score_all_high_prob(self):
        """All words with high probability → low clarity_score."""
        words = [self._make_word("hola", 0.95), self._make_word("mundo", 0.90)]
        segs = [
            TranscriptionSegment(
                text="hola mundo", start=0, end=1,
                avg_log_prob=-0.1, words=words,
            )
        ]
        t = _make_transcription("hola mundo", duration_seconds=2.0, segments=segs)
        m = analyze_structure(t)
        # low_conf_segment_pct = 0 (avg_log_prob > -0.5)
        # uncertain_word_pct = 0 (all probs > 0.5)
        # mean_word_prob ~ 0.925
        # clarity_score = 0.5*0 + 0.3*0 + 0.2*(1-0.925) = 0.015
        assert m.clarity_score == pytest.approx(0.015, abs=0.005)
        assert m.low_conf_segment_pct == 0.0
        assert m.uncertain_word_pct == 0.0

    def test_clarity_score_all_low_prob(self):
        """All words with low probability → high clarity_score."""
        words = [self._make_word("uh", 0.2), self._make_word("hmm", 0.3)]
        segs = [
            TranscriptionSegment(
                text="uh hmm", start=0, end=1,
                avg_log_prob=-0.8, words=words,
            )
        ]
        t = _make_transcription("uh hmm", duration_seconds=2.0, segments=segs)
        m = analyze_structure(t)
        # low_conf_segment_pct = 1.0 (avg_log_prob < -0.5)
        # uncertain_word_pct = 1.0 (all probs < 0.5)
        # mean_word_prob = 0.25
        # clarity_score = 0.5*1 + 0.3*1 + 0.2*0.75 = 0.95
        assert m.clarity_score == pytest.approx(0.95, abs=0.05)
        assert m.low_conf_segment_pct == 1.0
        assert m.uncertain_word_pct == 1.0

    def test_clarity_score_no_words_returns_zero(self):
        """Segments without word-level data → clarity_score = 0."""
        segs = [
            TranscriptionSegment(text="hola", start=0, end=1, avg_log_prob=-0.3)
        ]
        t = _make_transcription("hola", duration_seconds=2.0, segments=segs)
        m = analyze_structure(t)
        assert m.clarity_score == 0.0
        assert m.mean_word_prob == 0.0

    def test_clarity_score_mixed_confidence(self):
        """Mix of high and low confidence words."""
        words = [
            self._make_word("hola", 0.95),
            self._make_word("pues", 0.40),  # uncertain
        ]
        segs = [
            TranscriptionSegment(
                text="hola pues", start=0, end=1,
                avg_log_prob=-0.3, words=words,
            )
        ]
        t = _make_transcription("hola pues", duration_seconds=2.0, segments=segs)
        m = analyze_structure(t)
        # uncertain_word_pct = 0.5, low_conf_segment_pct = 0, mean_word_prob = 0.675
        assert 0.0 < m.clarity_score < 0.5
        assert m.uncertain_word_pct == pytest.approx(0.5)


# ===================================================================
# MATTR (Moving-Average TTR)
# ===================================================================

class TestMATTR:
    """Unit tests for the _mattr helper."""

    def test_empty_list(self):
        assert _mattr([], window=50) == 0.0

    def test_single_word(self):
        assert _mattr(["hola"], window=50) == pytest.approx(1.0)

    def test_all_same_word(self):
        words = ["casa"] * 300
        result = _mattr(words, window=100)
        assert result == pytest.approx(1 / 100)  # 1 unique in each window

    def test_all_unique(self):
        words = [f"word{i}" for i in range(300)]
        result = _mattr(words, window=100)
        assert result == pytest.approx(1.0)  # every window has 100 unique

    def test_shorter_than_window_falls_back_to_ttr(self):
        words = ["hola", "mundo", "hola"]
        result = _mattr(words, window=100)
        # Fallback to plain TTR: 2 unique / 3 total
        assert result == pytest.approx(2 / 3)

    def test_stable_across_lengths(self):
        """Same vocabulary pattern at different lengths should give similar MATTR."""
        pattern = ["el", "perro", "come", "la", "comida"] * 10  # 50 words, repeated
        short = pattern * 5   # 250 words
        long = pattern * 40   # 2000 words
        mattr_short = _mattr(short, window=50)
        mattr_long = _mattr(long, window=50)
        assert abs(mattr_short - mattr_long) < 0.05


# ===================================================================
# _tree_depth
# ===================================================================

class TestTreeDepth:
    """Unit test for the recursive tree depth function."""

    def test_with_spacy_doc(self):
        """Ensure _tree_depth works on a real spaCy parse."""
        from src.analyze import _get_nlp
        nlp = _get_nlp()
        doc = nlp("El gato negro duerme en la casa.")
        for sent in doc.sents:
            depth = _tree_depth(sent.root)
            assert depth >= 1
            assert isinstance(depth, int)


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Edge cases and empty input handling."""

    def test_empty_text_returns_defaults(self):
        t = _make_transcription("")
        m = analyze_structure(t)
        assert m.total_words == 0
        assert m.words_per_minute == 0.0

    def test_single_word(self):
        t = _make_transcription("hola")
        m = analyze_structure(t)
        assert m.total_words >= 1

    def test_word_counts_are_consistent(self):
        text = "el perro come la comida en la casa grande de la ciudad"
        text = " ".join([text] * 15)
        t = _make_transcription(text)
        m = analyze_structure(t)
        assert m.unique_words <= m.total_words
        # Lexical diversity excludes proper nouns, so it won't equal
        # unique_words/total_words exactly, but must be in (0, 1].
        assert 0.0 < m.lexical_diversity <= 1.0

    def test_proper_nouns_excluded_from_vocabulary(self):
        """Proper nouns (PROPN) like names/places should not inflate vocabulary
        difficulty — they're not 'hard words' for a learner."""
        # Text with common words + proper noun repeated
        text_with_names = "Xoloitzcuintle vive en la ciudad de Tenochtitlan. " * 10
        text_plain = "el perro vive en la ciudad de México. " * 10
        t_names = _make_transcription(text_with_names)
        t_plain = _make_transcription(text_plain)
        m_names = analyze_structure(t_names)
        m_plain = analyze_structure(t_plain)
        # With PROPN filtering, adding exotic proper nouns should not
        # dramatically increase the pct_outside_top_5k score.
        # Allow some tolerance since spaCy may not tag everything perfectly.
        assert m_names.pct_outside_top_5k < m_plain.pct_outside_top_5k + 0.15


# ===================================================================
# Tense detection
# ===================================================================

class TestTenseDetection:
    """Tests for verb tense classification and complexity scoring."""

    def test_present_indicative_detected(self):
        """Simple present-tense text should detect presente_indicativo."""
        text = "Yo como pan. Ella habla español. Nosotros vivimos aquí."
        text = " ".join([text] * 20)
        t = _make_transcription(text)
        m = analyze_structure(t)
        assert "presente_indicativo" in m.tense_distribution
        # Mostly present tense → low complexity
        assert m.tense_complexity < 0.30

    def test_past_tenses_detected(self):
        """Text with past tenses should detect preterito forms."""
        text = (
            "Ayer fui al mercado. Compré muchas frutas. "
            "Cuando era niño jugaba en el parque. Mi madre cocinaba bien."
        )
        text = " ".join([text] * 15)
        t = _make_transcription(text)
        m = analyze_structure(t)
        # Should have some past tense forms
        past_tenses = {"preterito_indefinido", "preterito_imperfecto"}
        detected_tenses = set(m.tense_distribution.keys())
        assert detected_tenses & past_tenses, (
            f"Expected past tenses, got: {detected_tenses}"
        )

    def test_subjunctive_raises_complexity(self):
        """Text with subjunctive should have higher complexity than indicative."""
        simple_text = "Yo como pan. Yo vivo aquí. Yo hablo español. " * 20
        advanced_text = (
            "Quiero que vengas. Espero que puedas. "
            "Dudo que sepa. Es posible que tenga razón. "
        ) * 15
        t_simple = _make_transcription(simple_text)
        t_advanced = _make_transcription(advanced_text)
        m_simple = analyze_structure(t_simple)
        m_advanced = analyze_structure(t_advanced)
        assert m_advanced.tense_complexity > m_simple.tense_complexity

    def test_tense_distribution_sums_to_one(self):
        """Tense distribution fractions should sum to approximately 1.0."""
        text = (
            "Yo como pan. Ayer fui al mercado. Mañana iré al cine. "
            "Quiero que vengas mañana."
        )
        text = " ".join([text] * 15)
        t = _make_transcription(text)
        m = analyze_structure(t)
        if m.tense_distribution:
            total = sum(m.tense_distribution.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_tense_complexity_in_valid_range(self):
        """Tense complexity should be between 0 and 1."""
        text = "Hola amigos. Hoy vamos a hablar sobre la vida cotidiana."
        text = " ".join([text] * 20)
        t = _make_transcription(text)
        m = analyze_structure(t)
        assert 0.0 <= m.tense_complexity <= 1.0

    def test_empty_text_has_no_tenses(self):
        """Empty transcript should have empty tense distribution."""
        t = _make_transcription("")
        m = analyze_structure(t)
        assert m.tense_distribution == {}
        assert m.tense_complexity == 0.0


class TestComputeTenseComplexity:
    """Unit tests for the _compute_tense_complexity helper."""

    def test_empty_counts(self):
        from collections import Counter
        dist, score = _compute_tense_complexity(Counter())
        assert dist == {}
        assert score == 0.0

    def test_all_present_indicative(self):
        from collections import Counter
        counts = Counter({"presente_indicativo": 100})
        dist, score = _compute_tense_complexity(counts)
        assert dist == {"presente_indicativo": 1.0}
        assert score == pytest.approx(TENSE_DIFFICULTY["presente_indicativo"])

    def test_mixed_tenses(self):
        from collections import Counter
        counts = Counter({
            "presente_indicativo": 50,
            "imperfecto_subjuntivo": 50,
        })
        dist, score = _compute_tense_complexity(counts)
        expected = 0.5 * TENSE_DIFFICULTY["presente_indicativo"] + \
                   0.5 * TENSE_DIFFICULTY["imperfecto_subjuntivo"]
        assert score == pytest.approx(expected, abs=0.01)

    def test_distribution_fractions(self):
        from collections import Counter
        counts = Counter({
            "presente_indicativo": 75,
            "futuro_simple": 25,
        })
        dist, score = _compute_tense_complexity(counts)
        assert dist["presente_indicativo"] == pytest.approx(0.75, abs=0.01)
        assert dist["futuro_simple"] == pytest.approx(0.25, abs=0.01)


# ===================================================================
# Vocabulary bucketed scoring
# ===================================================================

class TestComputeVocabScore:
    """Unit tests for _compute_vocab_score."""

    def test_empty_list(self):
        freq = {n: set() for n in (1_000, 2_000, 3_000, 4_000, 5_000, 10_000)}
        dist, score = _compute_vocab_score([], freq)
        assert dist == {}
        assert score == 0.0

    def test_all_top_1k(self):
        words = ["w1", "w2", "w3"]
        freq = {
            1_000: {"w1", "w2", "w3"},
            2_000: {"w1", "w2", "w3"},
            3_000: {"w1", "w2", "w3"},
            4_000: {"w1", "w2", "w3"},
            5_000: {"w1", "w2", "w3"},
            10_000: {"w1", "w2", "w3"},
        }
        dist, score = _compute_vocab_score(words, freq)
        assert dist["top_1k"] == pytest.approx(1.0)
        assert score == pytest.approx(0.0)  # VOCAB_BUCKET_DIFFICULTY["top_1k"] == 0

    def test_all_5k_plus(self):
        words = ["rare1", "rare2"]
        freq = {n: set() for n in (1_000, 2_000, 3_000, 4_000, 5_000, 10_000)}
        dist, score = _compute_vocab_score(words, freq)
        assert dist["5k_plus"] == pytest.approx(1.0)
        assert score == pytest.approx(1.0)  # VOCAB_BUCKET_DIFFICULTY["5k_plus"] == 1.0

    def test_mixed_buckets(self):
        words = ["easy", "mid", "hard", "rare"]
        freq = {
            1_000: {"easy"},
            2_000: {"easy", "mid"},
            3_000: {"easy", "mid"},
            4_000: {"easy", "mid"},
            5_000: {"easy", "mid", "hard"},
            10_000: {"easy", "mid", "hard"},
        }
        # easy → top_1k, mid → 1k_2k, hard → 4k_5k, rare → 5k_plus
        dist, score = _compute_vocab_score(words, freq)
        assert dist["top_1k"] == pytest.approx(0.25)
        assert dist["1k_2k"] == pytest.approx(0.25)
        assert dist["4k_5k"] == pytest.approx(0.25)
        assert dist["5k_plus"] == pytest.approx(0.25)
        expected = 0.25 * (0.00 + 0.10 + 0.65 + 1.00)
        assert score == pytest.approx(expected, abs=0.01)

    def test_distribution_sums_to_one(self):
        words = ["a", "b", "c", "d", "e"]
        freq = {
            1_000: {"a"},
            2_000: {"a", "b"},
            3_000: {"a", "b", "c"},
            4_000: {"a", "b", "c"},
            5_000: {"a", "b", "c", "d"},
            10_000: {"a", "b", "c", "d"},
        }
        dist, score = _compute_vocab_score(words, freq)
        assert sum(dist.values()) == pytest.approx(1.0, abs=0.01)
        assert 0.0 <= score <= 1.0
