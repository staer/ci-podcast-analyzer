"""Data models for the podcast analyzer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Episode(BaseModel):
    """Represents a single podcast episode."""

    title: str
    url: str  # Direct audio URL
    published: str | None = None
    duration_seconds: float | None = None
    audio_path: str | None = None  # Local path after download


class PodcastFeed(BaseModel):
    """Represents a parsed podcast RSS feed."""

    title: str
    feed_url: str
    description: str | None = None
    language: str | None = None
    episodes: list[Episode] = Field(default_factory=list)


class TranscriptionSegment(BaseModel):
    """A single segment from Whisper transcription."""

    text: str
    start: float  # Start time in seconds
    end: float    # End time in seconds
    words: list[WordTimestamp] = Field(default_factory=list)
    avg_log_prob: float | None = None  # Whisper confidence proxy


class WordTimestamp(BaseModel):
    """Word-level timing from Whisper."""

    word: str
    start: float
    end: float
    probability: float | None = None


class Transcription(BaseModel):
    """Full transcription of an episode."""

    episode_title: str
    full_text: str
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    duration_seconds: float = 0.0
    language: str = "es"
    language_probability: float | None = None


class WhisperParams(BaseModel):
    """Whisper parameters used to produce a transcription (for cache validity)."""

    model_size: str = "small"
    beam_size: int = 1
    language: str = "es"
    skip_intro_seconds: int = 45
    max_transcribe_minutes: int = 10
    first_half_only: bool = False


class CachedTranscription(BaseModel):
    """A transcription bundled with the parameters that produced it."""

    episode_title: str = ""
    whisper_params: WhisperParams
    episode_url: str
    transcription: Transcription
    cached_at: str = ""  # ISO timestamp


class CachedAnalysis(BaseModel):
    """Cached structural metrics for an episode."""

    episode_title: str = ""
    podcast_title: str = ""  # feed-level title, for rescore grouping
    feed_url: str = ""       # feed URL, for rescore grouping
    episode_url: str
    whisper_params: WhisperParams  # metrics depend on the transcription
    structural_metrics: StructuralMetrics
    cached_at: str = ""


class CachedLLMAnalysis(BaseModel):
    """Cached LLM qualitative analysis for an episode."""

    episode_title: str = ""
    podcast_title: str = ""
    feed_url: str = ""
    episode_url: str
    llm_analysis: LLMAnalysis
    cached_at: str = ""


class StructuralMetrics(BaseModel):
    """Metrics derived from NLP / structural analysis."""

    words_per_minute: float = 0.0
    total_words: int = 0
    unique_words: int = 0
    lexical_diversity: float = 0.0  # type-token ratio
    avg_sentence_length: float = 0.0
    avg_word_length: float = 0.0
    # Vocabulary level: fraction of words NOT in the top-N frequency lists
    pct_outside_top_1k: float = 0.0
    pct_outside_top_5k: float = 0.0
    pct_outside_top_10k: float = 0.0
    # Bucketed vocabulary: distribution across frequency tiers and composite score
    vocab_distribution: dict[str, float] = Field(default_factory=dict)
    vocab_score: float = 0.0  # weighted difficulty from frequency buckets (0-1)
    # Grammar complexity from spaCy
    avg_parse_depth: float = 0.0
    subjunctive_ratio: float = 0.0  # ratio of subjunctive verb forms
    subordinate_clause_ratio: float = 0.0
    # Tense analysis: distribution of verb tenses and weighted complexity
    tense_distribution: dict[str, float] = Field(default_factory=dict)
    tense_complexity: float = 0.0  # weighted difficulty score (0-1)
    # Punctuation density: ratio of sentence-ending punct to total words.
    # Low values indicate Whisper did not reliably insert punctuation,
    # making sentence-boundary metrics (avg_sentence_length) unreliable.
    punctuation_density: float = 0.0
    # Clarity proxy from Whisper confidence
    avg_segment_confidence: float = 0.0
    # Word-level clarity metrics (higher = harder to understand)
    clarity_score: float = 0.0        # composite: 50% low_conf_seg + 30% uncertain_words + 20% (1-mean_word_prob)
    low_conf_segment_pct: float = 0.0  # fraction of segments with avg_log_prob < -0.5
    uncertain_word_pct: float = 0.0    # fraction of words with probability < 0.5
    mean_word_prob: float = 0.0        # average word-level probability (0-1)


class LLMAnalysis(BaseModel):
    """Qualitative analysis from an LLM."""

    slang_score: float = 0.0          # 0 (no slang) to 1 (heavy slang)
    slang_examples: list[str] = Field(default_factory=list)
    topic_complexity: float = 0.0     # 0 (simple daily) to 1 (highly technical)
    topic_summary: str = ""
    estimated_cefr: str = ""          # A1, A2, B1, B2, C1, C2
    explanation: str = ""
    idiom_count: int = 0
    formality_score: float = 0.0      # 0 (very informal) to 1 (very formal)


class EpisodeAnalysis(BaseModel):
    """Complete analysis for one episode."""

    episode: Episode
    transcription: Transcription | None = None
    structural_metrics: StructuralMetrics | None = None
    llm_analysis: LLMAnalysis | None = None


class DifficultyScore(BaseModel):
    """Final composite difficulty score for a podcast."""

    podcast_title: str
    feed_url: str
    overall_score: float = 0.0        # 0 (easiest) to 1 (hardest)
    cefr_estimate: str = ""           # Mapped from overall_score
    component_scores: dict[str, float] = Field(default_factory=dict)
    episodes_analyzed: int = 0        # episodes kept after outlier removal
    total_episodes: int = 0           # episodes considered before outlier removal
    episode_results: list[EpisodeAnalysis] = Field(default_factory=list)
    trimmed_episodes: list[str] = Field(default_factory=list)  # titles of outlier-trimmed episodes


# --- Update forward refs (needed because WordTimestamp is referenced before
#     its definition in TranscriptionSegment) ---
TranscriptionSegment.model_rebuild()
CachedAnalysis.model_rebuild()
CachedLLMAnalysis.model_rebuild()
