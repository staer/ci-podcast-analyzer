"""Audio transcription using faster-whisper."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure NVIDIA pip-packaged CUDA libraries (e.g. nvidia-cublas-cu12) are
# visible to the DLL loader.  Without this, CTranslate2 / faster-whisper
# cannot find cublas64_12.dll even though the package is installed.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    _site = Path(sys.prefix, "Lib", "site-packages", "nvidia")
    if _site.is_dir():
        for _sub in _site.iterdir():
            _bin = _sub / "bin"
            if _bin.is_dir():
                os.add_dll_directory(str(_bin))
                if str(_bin) not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
from faster_whisper import WhisperModel

import config
from src.models import Episode, Transcription, TranscriptionSegment, WordTimestamp

logger = logging.getLogger(__name__)

# Whisper expects mono 16 kHz float32 audio
_WHISPER_SAMPLE_RATE = 16_000


def _load_audio_trimmed(path: str, max_minutes: float) -> np.ndarray:
    """Decode audio into a mono 16 kHz float32 ndarray, reading at most *max_minutes*.

    Uses PyAV (already installed as a faster-whisper dependency) so no
    external ffmpeg binary is required.  For episodes shorter than the
    limit the full file is returned.
    """
    import av  # pyav – already a faster-whisper dependency

    max_samples = int(max_minutes * 60 * _WHISPER_SAMPLE_RATE) if max_minutes > 0 else 0

    container = av.open(path)
    stream = container.streams.audio[0]
    # Ask the decoder to output 16 kHz mono s16
    resampler = av.AudioResampler(format="s16", layout="mono", rate=_WHISPER_SAMPLE_RATE)

    chunks: list[np.ndarray] = []
    collected = 0

    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray().flatten().astype(np.float32) / 32768.0
            if max_samples > 0 and collected + len(arr) > max_samples:
                arr = arr[: max_samples - collected]
                chunks.append(arr)
                collected += len(arr)
                break
            chunks.append(arr)
            collected += len(arr)
        if max_samples > 0 and collected >= max_samples:
            break

    container.close()
    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    return audio

# Module-level cache so we only load the model once
_model: WhisperModel | None = None


def _has_cuda() -> bool:
    """Check if a CUDA GPU is available via CTranslate2."""
    try:
        from ctranslate2 import get_cuda_device_count
        return get_cuda_device_count() > 0
    except Exception:
        return False


def _get_model() -> WhisperModel:
    """Lazy-load the Whisper model, auto-tuning for GPU when available."""
    global _model
    if _model is None:
        device = config.WHISPER_DEVICE
        compute_type = config.WHISPER_COMPUTE_TYPE
        model_size = config.WHISPER_MODEL_SIZE
        beam_size = config.WHISPER_BEAM_SIZE

        gpu_available = _has_cuda()

        # Auto-detect: upgrade settings when a GPU is present and the user
        # hasn't explicitly pinned values via CLI / feeds file.
        if gpu_available and device in ("auto", "cuda"):
            device = "cuda"
            # Only upgrade if still at CPU defaults
            if compute_type == "int8":
                compute_type = "float16"
            if model_size == "small":
                model_size = "medium"
            if beam_size == 1:
                beam_size = 5
            logger.info(
                "CUDA GPU detected -- upgrading: model=%s, compute=%s, beam=%d",
                model_size, compute_type, beam_size,
            )
        elif not gpu_available and device == "auto":
            device = "cpu"

        # Store effective values back so the cache key reflects what was used
        config.WHISPER_MODEL_SIZE = model_size
        config.WHISPER_BEAM_SIZE = beam_size
        config.WHISPER_COMPUTE_TYPE = compute_type

        cpu_threads = config.WHISPER_CPU_THREADS
        if cpu_threads == 0:
            import os
            cpu_threads = os.cpu_count() or 4

        logger.info(
            "Loading Whisper model '%s' (device=%s, compute=%s, threads=%d, beam=%d)...",
            model_size, device, compute_type, cpu_threads, beam_size,
        )
        _model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
    return _model


def transcribe_episode(episode: Episode) -> Transcription:
    """Transcribe an episode's audio file.

    Args:
        episode: An Episode whose audio_path points to a local file.

    Returns:
        A Transcription with full text, segments, and word timestamps.

    Raises:
        FileNotFoundError: If audio_path is not set or the file doesn't exist.
    """
    if not episode.audio_path:
        raise FileNotFoundError(f"No audio_path for episode '{episode.title}'")

    audio = Path(episode.audio_path)
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    model = _get_model()
    logger.info("Transcribing: %s", audio.name)

    # For long episodes, decode only the first MAX_TRANSCRIBE_MINUTES of
    # audio so we don't waste time/memory on hours of content.
    max_min = config.MAX_TRANSCRIBE_MINUTES
    if max_min > 0:
        logger.info(
            "Loading first %d minutes of audio (trimming long episodes)...",
            max_min,
        )
        audio_data: str | np.ndarray = _load_audio_trimmed(str(audio), max_min)
        logger.info(
            "Loaded %.1f s of audio (%.1f MB in memory)",
            len(audio_data) / _WHISPER_SAMPLE_RATE,
            audio_data.nbytes / (1024 * 1024),
        )
    else:
        audio_data = str(audio)  # full file path – let Whisper decode it

    segments_iter, info = model.transcribe(
        audio_data,
        language=config.WHISPER_LANGUAGE,
        word_timestamps=True,
        vad_filter=True,  # Filter out silence
        beam_size=config.WHISPER_BEAM_SIZE,
    )

    segments: list[TranscriptionSegment] = []
    full_text_parts: list[str] = []

    skip = config.SKIP_INTRO_SECONDS
    max_end = (
        skip + config.MAX_TRANSCRIBE_MINUTES * 60
        if config.MAX_TRANSCRIBE_MINUTES > 0
        else float("inf")
    )
    skipped_count = 0

    last_progress = -1  # track last printed minute for progress updates

    for seg in segments_iter:
        # Skip segments that start within the intro/ad window
        if seg.start < skip:
            skipped_count += 1
            continue

        # Progress indicator: log every 60 seconds of transcribed audio
        current_minute = int(seg.start // 60)
        if current_minute > last_progress:
            last_progress = current_minute
            if max_end < float("inf"):
                target_min = int(max_end // 60)
                logger.info(
                    "  Transcribing... %d:%02d / ~%d:%02d of audio",
                    current_minute, int(seg.start % 60),
                    target_min, int(max_end % 60),
                )
            else:
                logger.info(
                    "  Transcribing... %d:%02d of audio so far",
                    current_minute, int(seg.start % 60),
                )

        # Stop if we've transcribed enough
        if seg.start >= max_end:
            logger.info(
                "Reached %d-minute transcription limit at %.0fs",
                config.MAX_TRANSCRIBE_MINUTES,
                seg.start,
            )
            break

        words: list[WordTimestamp] = []
        if seg.words:
            words = [
                WordTimestamp(
                    word=w.word.strip(),
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                )
                for w in seg.words
                if w.word.strip()
            ]

        segment = TranscriptionSegment(
            text=seg.text.strip(),
            start=seg.start,
            end=seg.end,
            words=words,
            avg_log_prob=seg.avg_logprob,
        )
        segments.append(segment)
        full_text_parts.append(seg.text.strip())

    if skipped_count:
        logger.info(
            "Skipped %d segments in first %ds (intro/ads)",
            skipped_count,
            skip,
        )

    full_text = " ".join(full_text_parts)
    duration = segments[-1].end if segments else 0.0

    logger.info(
        "Transcription complete: %d segments, %.0f seconds, language=%s (p=%.2f)",
        len(segments),
        duration,
        info.language,
        info.language_probability,
    )

    return Transcription(
        episode_title=episode.title,
        full_text=full_text,
        segments=segments,
        duration_seconds=duration,
        language=info.language,
        language_probability=info.language_probability,
    )


def save_transcript(transcript: Transcription, output_dir: Path | None = None) -> Path:
    """Save the transcript text to a file and return its path."""
    import re
    output_dir = output_dir or config.TRANSCRIPTS_DIR
    # Strip characters illegal in Windows filenames: \ / : * ? " < > |
    # Also replace # and whitespace with underscores for safety.
    safe = re.sub(r'[\\/*?:"<>|#\s]+', "_", transcript.episode_title[:80]).strip("_")
    if not safe:
        safe = "untitled"
    path = output_dir / f"{safe}.txt"
    path.write_text(transcript.full_text, encoding="utf-8")
    logger.info("Saved transcript: %s", path)
    return path
