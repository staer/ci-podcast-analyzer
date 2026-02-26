"""Tests for src/analyze.py – structural NLP metrics.

These tests load the real spaCy model (es_core_news_lg) so they require
it to be installed.  Mark the module as slow / integration if needed.
"""

from __future__ import annotations

import pytest

from src.analyze import _tree_depth, analyze_structure
from src.models import Transcription, TranscriptionSegment


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
    """Type-token ratio."""

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
