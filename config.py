"""Configuration for the Spanish Podcast Difficulty Analyzer."""

from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = DATA_DIR / "results"

# Create directories on import
for d in [AUDIO_DIR, TRANSCRIPTS_DIR, CACHE_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Whisper settings ---
WHISPER_MODEL_SIZE = "small"    # tiny, base, small, medium, large-v3
WHISPER_DEVICE = "auto"        # "auto", "cpu", or "cuda"
WHISPER_LANGUAGE = "es"
WHISPER_BEAM_SIZE = 1           # 1 = greedy (fast), 5 = beam search (accurate)
WHISPER_COMPUTE_TYPE = "int8"   # int8 (fastest on CPU), float16 (GPU), auto
WHISPER_CPU_THREADS = 0         # 0 = auto-detect, or set specific core count
MAX_TRANSCRIBE_MINUTES = 20    # Only transcribe first N minutes per episode (0 = full)
SKIP_INTRO_SECONDS = 45         # Skip this many seconds at the start (ads/intros)
FIRST_HALF_ONLY = False         # Only transcribe the first half of each episode

# --- Feed settings ---
MAX_EPISODES_PER_FEED = 20      # Hard cap on episodes to consider from the feed
MIN_EPISODES = 5                # Always analyze at least this many episodes
TARGET_AUDIO_MINUTES = 60       # Download enough episodes to reach this duration
DOWNLOAD_TIMEOUT_SECONDS = 300
# When episodes are shorter than this threshold (seconds), prefer
# longer episodes over shorter ones during sampling.  Longer episodes
# produce more reliable structural metrics (especially lexical diversity).
# Set to 0 to disable the preference.
PREFER_LONGER_THRESHOLD = 600   # 10 minutes

# --- Scoring settings ---
# When we have enough episodes, drop the single highest-scoring outlier
# before averaging.  This prevents one atypical episode from skewing the
# podcast score.  Set to 0 to disable.
OUTLIER_TRIM_COUNT = 1
OUTLIER_TRIM_MIN_EPISODES = 4   # Only trim when we have at least this many

# --- Analysis settings ---
SPACY_MODEL = "es_core_news_lg"
# Minimum number of words in a transcript to consider it valid
MIN_TRANSCRIPT_WORDS = 100
# MATTR window size for lexical diversity.  Larger = more stable but
# requires longer transcripts.  200 is a good balance: even the shortest
# CI podcast episodes (~280 words) produce multiple windows.
MATTR_WINDOW_SIZE = 200

# --- LLM settings ---
OPENAI_MODEL = "gpt-4o"
# Maximum transcript characters to send to the LLM (to control cost)
LLM_MAX_TRANSCRIPT_CHARS = 8000

# --- Local LLM (Ollama) settings ---
USE_LOCAL_LLM = False
OLLAMA_BASE_URL = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint
OLLAMA_MODEL = "llama3"  # Model to use (llama3, mistral, gemma2, etc.)

# --- Scoring weights (must sum to 1.0) ---
# Speech rate is the single strongest signal: learner podcasts speak slowly
# (60-100 wpm) while native ones are 160-200+.  Vocabulary is the next most
# reliable discriminator (though biased upward for vocab-teaching podcasts
# that deliberately introduce rare words).  Lexical diversity is very noisy
# on spoken transcripts — MATTR is length-dependent and inflated on short
# episodes; it gets a minimal weight.  Clarity (Whisper confidence) directly
# reflects how easy the audio is to follow — clear studio recordings vs.
# noisy conversation — and gets a meaningful weight.  Tense complexity
# captures how advanced the grammar is: podcasts heavy on present indicative
# are beginner-friendly; those with subjunctive / compound tenses are harder.
SCORING_WEIGHTS = {
    "speech_rate": 0.30,
    "vocabulary_level": 0.18,
    "lexical_diversity": 0.02,
    "sentence_length": 0.05,
    "grammar_complexity": 0.02,
    "tense_complexity": 0.10,
    "subjunctive_ratio": 0.05,
    "subordinate_clause_ratio": 0.03,
    "slang_score": 0.10,
    "topic_complexity": 0.10,
    "clarity": 0.05,
}
