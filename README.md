# Spanish Podcast Difficulty Analyzer

Automatically analyzes Spanish-language podcasts and rates them by difficulty level (CEFR A1–C2). Point it at one or more RSS feeds and it will download episodes, transcribe the audio, run NLP analysis, and produce a composite difficulty score so you can find podcasts that match your level.

## How It Works

The analyzer pipeline has four stages:

1. **Feed parsing & download** — Reads the RSS feed, samples episodes to reach a target duration (preferring previously cached episodes), and downloads the audio files. Long episodes (>20 min) are downloaded but only the first 20 minutes of audio are decoded for transcription.
2. **Transcription** — Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2-based Whisper) to transcribe Spanish audio with word-level timestamps. Audio is decoded via PyAV and trimmed to the configured limit before transcription, so long episodes don't waste GPU/CPU time. The first 45 seconds of each episode are skipped to avoid ads/intros.
3. **Structural NLP analysis** — Uses [spaCy](https://spacy.io/) and [wordfreq](https://github.com/rspeer/wordfreq) to compute metrics like speech rate, vocabulary level, grammar complexity, and lexical diversity. Proper nouns (names, places) are excluded so exotic topics don't inflate difficulty.
4. **Scoring** — Normalises the metrics, trims the highest-scoring outlier episode, and produces a composite 0–1 difficulty score mapped to CEFR levels.

An optional LLM stage can assess slang usage, topic complexity, and formality via OpenAI or a local [Ollama](https://ollama.com/) model.

### Metrics & Scoring Weights

| Metric | Source | What it measures |
|---|---|---|
| Speech rate (WPM) | Whisper timestamps | How fast the speaker talks |
| Vocabulary level | wordfreq top-1k/5k/10k | % of words outside common frequency lists (proper nouns excluded) |
| Lexical diversity | spaCy | Type-token ratio (proper nouns excluded) |
| Sentence length | spaCy | Average words per sentence |
| Grammar complexity | spaCy parse tree | Average dependency parse depth |
| Subjunctive ratio | spaCy morphology | How often subjunctive mood is used |
| Clarity | Whisper confidence | Average segment log-probability |
| Slang score | LLM (optional) | Amount of colloquial/slang language |
| Topic complexity | LLM (optional) | Subject matter difficulty |

#### Scoring Weights

| Component | Weight | Notes |
|---|---|---|
| Vocabulary level | 25% | Strongest signal for learner content |
| Speech rate | 20% | Slow = easier |
| Slang score | 15% | LLM-only; redistributed when LLM disabled |
| Grammar complexity | 10% | Parse tree depth |
| Topic complexity | 10% | LLM-only; redistributed when LLM disabled |
| Clarity | 10% | Whisper confidence proxy |
| Lexical diversity | 5% | Downweighted — noisy on short transcripts |
| Sentence length | 5% | Downweighted — Whisper punctuation is imperfect |

When LLM is disabled (default), the slang and topic weights are redistributed proportionally across the structural components.

## Quick Start

### Prerequisites

- Python 3.10–3.12 (spaCy 3.x does not yet support 3.13+)
- ~2 GB disk space for models (downloaded automatically on first run)
- **Optional:** NVIDIA GPU with CUDA for faster transcription (see [GPU Setup](#gpu-setup))

### Option A: Using the wrapper script (recommended)

The PowerShell script creates a virtual environment, installs all dependencies, and runs the analyzer:

```powershell
# Analyze a single feed
.\analyze.ps1 https://anchor.fm/s/4baec630/podcast/rss

# Analyze all feeds in the included feeds file
.\analyze.ps1 -f feeds.json

# See all options
.\analyze.ps1 --help
```

On Linux/macOS, use `./analyze.sh` instead.

### Option B: Manual setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download es_core_news_lg

python main.py https://anchor.fm/s/4baec630/podcast/rss
```

## Usage

```
python main.py [OPTIONS] [FEED_URL ...]
```

### Common Options

| Flag | Description | Default |
|---|---|---|
| `--feeds-file`, `-f` | JSON file with feed URLs and settings | — |
| `--duration`, `-d` | Target audio minutes to sample per feed | 60 |
| `--episodes`, `-n` | Hard cap on episodes per feed | 20 |
| `--min-episodes` | Minimum episodes even if duration target is met | 5 |
| `--whisper-model`, `-w` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` | `small` |
| `--beam-size` | Beam width for decoding (1 = fast greedy, 5 = accurate) | 1 |
| `--max-transcribe-minutes` | Only transcribe first N minutes per episode (0 = full) | 20 |
| `--output`, `-o` | Save results as JSON to a file | — |
| `--no-cache` | Force re-transcription, ignoring cached results | off |
| `--rescore` | Rescore all cached episodes using current weights (no downloads) | off |
| `--verbose`, `-v` | Enable debug logging | off |

### LLM Options

LLM analysis is **off by default** (no API key needed). Enable it for slang/topic scoring:

| Flag | Description |
|---|---|
| `--use-llm` | Enable LLM analysis (requires `OPENAI_API_KEY` env var) |
| `--local-llm` | Use Ollama instead of OpenAI (free, runs locally) |
| `--ollama-model` | Ollama model name (default: `llama3`). Implies `--local-llm` |

### Examples

```powershell
# Compare multiple feeds side-by-side
python main.py feed1_url feed2_url feed3_url

# Higher quality transcription (slower)
python main.py --whisper-model medium --beam-size 5 --max-transcribe-minutes 0 feed_url

# Batch mode from feeds.json with JSON output
python main.py -f feeds.json -o results.json

# Use Ollama for free local LLM scoring
python main.py --use-llm --local-llm feed_url

# Instantly rescore all cached podcasts with current weights
python main.py --rescore

# Rescore and save to JSON
python main.py --rescore -o rescored.json
```

## Feeds File

The included `feeds.json` has 7 pre-configured Spanish podcast feeds. You can edit it or create your own:

```json
{
  "settings": {
    "whisper_model": "small",
    "beam_size": 1,
    "max_transcribe_minutes": 20,
    "use_llm": false,
    "target_audio_minutes": 60,
    "min_episodes": 5,
    "max_episodes": 20,
    "skip_intro_seconds": 45
  },
  "feeds": [
    { "name": "My Podcast", "url": "https://example.com/feed.xml" }
  ]
}
```

CLI flags override settings from the feeds file.

## Caching

Transcription is the slowest step. Results are automatically cached in `data/cache/` so re-running the same feeds is nearly instant. Cache files are JSON and include:

- Full transcription with word-level timestamps (`tx_*.json`)
- Structural NLP metrics (`an_*.json`)
- LLM qualitative analysis (`llm_*.json`), when LLM is enabled
- The Whisper parameters used (model, beam size, skip seconds, etc.)
- Feed URL and podcast title for grouping

Changing any Whisper parameter automatically invalidates the cache for affected episodes. Re-runs prefer episodes that are already cached to avoid unnecessary downloads. Use `--no-cache` to force a fresh transcription.

### Rescoring from Cache

Over time you build up a database of transcribed podcasts. When you tweak scoring weights, normalisation ranges, or CEFR thresholds in `config.py`, you can instantly rescore everything without any downloads or transcription:

```powershell
# Rescore all cached podcasts with current weights
python main.py --rescore

# Rescore and save updated results
python main.py --rescore -o results.json
```

The rescore workflow:
1. Scans `data/cache/` for all cached analysis files
2. Groups episodes by podcast feed
3. Applies the current scoring weights from `config.py`
4. Prints updated reports and comparative ranking

This makes it easy to iterate on the scoring model: run the full pipeline once per podcast, then rescore instantly as many times as you want.

## Outlier Trimming

When 4 or more episodes are analyzed, the single highest-scoring (hardest) episode is automatically excluded from the average. This prevents one atypical episode — a guest speaker, a special topic, unusually fast speech — from skewing the whole podcast score.

Trimmed episodes are still shown in the report, marked with `[TRIMMED]`. The feature is configurable in `config.py`:

```python
OUTLIER_TRIM_COUNT = 1          # Number of outliers to drop (0 = disable)
OUTLIER_TRIM_MIN_EPISODES = 4   # Only trim when we have at least this many
```

## Output

The analyzer prints a human-readable report for each feed:

```
============================================================
  Podcast Difficulty Report
============================================================
  Title:    Chill Spanish Listening Practice
  Feed:     https://anchor.fm/s/51d84fcc/podcast/rss
  Episodes: 5 (+ 1 outlier trimmed)
  Mode:     Structural only (no LLM)

  Overall Score:  0.248 / 1.000
  CEFR Estimate:  A2

  Component Scores:
    clarity                   0.075  #
    grammar_complexity        0.380  #######
    lexical_diversity         0.620  ############
    sentence_length           0.510  ##########
    speech_rate               0.205  ####
    vocabulary_level          0.190  ###

  Episode 1: #314 "Los chistes malos"
    Words: 479  |  WPM: 73.8  |  Lex div: 0.480  |  Outside 5k: 14.1%
  ...
  Episode 6: #312 "Difícil tema"  [TRIMMED]
    Words: 521  |  WPM: 120.3  |  Lex div: 0.510  |  Outside 5k: 22.0%

============================================================
```

When analyzing multiple feeds, a detailed comparative ranking is printed at the end (easiest → hardest), showing exactly how each component contributed to the final score:

```
======================================================================
  COMPARATIVE RANKING  (easiest → hardest)
======================================================================

  1. Chill Spanish  —  0.248  [A2]  (5 episodes)
     Component                      Avg  x     Wt  =  Contrib
     ───────────────────────────────────────────────────────
     Clarity (Whisper conf.)      0.075  x   0.13  =   0.0100
     Grammar complexity           0.380  x   0.13  =   0.0507
     Lexical diversity (TTR)      0.620  x   0.07  =   0.0413
     Sentence length              0.510  x   0.07  =   0.0340
     Speech rate (WPM)            0.205  x   0.27  =   0.0547
     Vocabulary (outside 5k)      0.190  x   0.33  =   0.0633
     ───────────────────────────────────────────────────────
     TOTAL                                            0.2540

  2. Blood and Marble  —  0.712  [B2]  (6 episodes)
     Trimmed outliers: The Twelve Tables
     ...
======================================================================
```

With `--output results.json`, the full data (all metrics, per-episode breakdowns) is saved as JSON for further processing.

## Project Structure

```
├── main.py              # CLI entry point and pipeline orchestration
├── config.py            # All tunable settings (weights, ranges, thresholds)
├── feeds.json           # Pre-configured Spanish podcast feeds
├── requirements.txt     # Python dependencies
├── analyze.ps1          # PowerShell wrapper (setup + run)
├── analyze.sh           # Bash wrapper (Linux/macOS)
├── LICENSE              # MIT License
├── NOTICES.txt          # Third-party license attribution
├── src/
│   ├── models.py        # Pydantic data models
│   ├── feed.py          # RSS parsing and audio download
│   ├── transcribe.py    # Whisper transcription + GPU auto-detection
│   ├── analyze.py       # spaCy / wordfreq structural analysis
│   ├── llm_analyze.py   # Optional LLM qualitative analysis
│   ├── scoring.py       # Score normalisation, outlier trimming, CEFR mapping
│   └── cache.py         # Transcription/analysis/LLM caching + rescore scan
└── tests/
    ├── conftest.py      # pytest path setup
    ├── test_scoring.py  # Normalisation, CEFR mapping, weights, outlier trimming
    ├── test_analyze.py  # Speech rate, vocab, parse depth, proper noun filtering
    ├── test_cache.py    # Cache keys, round-trips, LLM cache, scan, feed metadata
    ├── test_models.py   # Pydantic model validation and serialisation
    └── test_feed.py     # Episode sampling, cache preference
```

## Testing

The project includes a comprehensive test suite (139 tests) using pytest.

```powershell
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_scoring.py -v

# Run a single test class
python -m pytest tests/test_scoring.py::TestNormalise -v
```

### What's tested

| Test file | Coverage |
|---|---|
| `test_scoring.py` | Normalisation (bounds, clamping, negative ranges, edge cases), CEFR boundary mapping, per-episode scoring, composite scoring, weight redistribution without LLM, report formatting, outlier trimming (trim/no-trim thresholds, disabled, score reduction, report labels) |
| `test_analyze.py` | Speech rate (WPM), lexical diversity, vocabulary level (1k/5k/10k ordering), parse depth, sentence length, segment confidence, tree depth, proper noun exclusion, edge cases |
| `test_cache.py` | Cache key determinism, parameter change → different key, save/load round-trips, parameter mismatch invalidation, word timestamp preservation, JSON structure, LLM analysis cache, cache scan grouping by feed, legacy file handling, feed metadata |
| `test_models.py` | All Pydantic models: required/optional fields, defaults, validation errors, forward references, serialisation round-trips, CachedLLMAnalysis, feed metadata |
| `test_feed.py` | Episode sampling: min-episodes floor, duration targeting, random shuffling, missing-duration fallback, cached episode preference, edge cases |

> **Note:** `test_analyze.py` loads the real spaCy model (`es_core_news_lg`) so it requires the model to be installed and takes ~5–25 seconds depending on hardware. The other test files run in under a second.

## GPU Setup

The analyzer **auto-detects CUDA GPUs** at startup. When a GPU is found, it automatically upgrades from CPU defaults to GPU-optimised settings:

| Setting | CPU default | GPU auto-upgrade |
|---|---|---|
| Whisper model | `small` | `medium` |
| Compute type | `int8` | `float16` |
| Beam size | `1` (greedy) | `5` (beam search) |

This means you get higher-quality transcription for free — just have a CUDA GPU available.

### Installing CUDA support

faster-whisper uses CTranslate2, which requires CUDA libraries:

1. **Install NVIDIA drivers** for your GPU (if not already installed)
2. **Install CUDA Toolkit 12.x** from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
3. **Install cuDNN** (bundled with recent CUDA Toolkit installers, or download from [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn))
4. **Verify** CUDA is visible:
   ```powershell
   python -c "from ctranslate2 import get_cuda_device_count; print(f'GPUs: {get_cuda_device_count()}')"
   ```

If the check prints `GPUs: 0`, CUDA libraries are not on your PATH. On Windows, the `nvidia-cublas-cu12` pip package is installed automatically (via `requirements.txt`) and the analyzer registers its DLL directory at startup. If you still see errors about `cublas64_12.dll`, ensure `cudnn*.dll` is available (typically in `C:\Program Files\NVIDIA\CUDNN\bin` or via the CUDA Toolkit installer).

### Overriding auto-detection

You can force specific settings regardless of GPU availability:

```powershell
# Force CPU-friendly settings even with a GPU present
python main.py --whisper-model small --beam-size 1 feed_url

# Force large model on GPU
python main.py --whisper-model large-v3 --beam-size 5 feed_url
```

### Speed comparison

| Hardware | Model | Beam | 20 min audio |
|---|---|---|---|
| CPU (22 cores) | small / int8 | 1 | ~8–12 min |
| GPU (RTX 3080) | medium / float16 | 5 | ~40–60 sec |
| GPU (RTX 4090) | large-v3 / float16 | 5 | ~20–30 sec |

Times are approximate and depend on audio content (silence, music, speech density).

## Performance Tips

- **No GPU?** The defaults (`small` model, `int8` compute, `beam_size=1`, 20-min cap) are already optimised for CPU.
- **Have a GPU?** It auto-detects and upgrades settings. Or manually: `--whisper-model large-v3 --beam-size 5`.
- **Long episodes?** Audio is trimmed to 20 minutes at the decoding stage via PyAV, so even hour-long episodes use minimal memory.
- **Second run?** Cached transcriptions are reused — takes seconds instead of minutes.
- **Tuning weights?** Use `--rescore` to instantly re-rank everything from the cache.

## License

This project is licensed under the [MIT License](LICENSE). See [NOTICES.txt](NOTICES.txt) for third-party attribution.
