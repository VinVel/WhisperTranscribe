from __future__ import annotations

import argparse
import functools
import gc
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TARGET_SAMPLE_RATE = 16_000
MIN_LANGUAGE_DETECTION_SECONDS = 0.8
MERGE_GAP_SECONDS = 0.35
MIN_OVERLAP_MARK_SECONDS = 0.2
TIMER_MIN_MS = 10.0
DEFAULT_WHISPER_MODEL_PATH = Path("models") / "faster-whisper-large-v3"
DEFAULT_DIARIZATION_MODEL_PATH = Path("models") / "pyannote-speaker-diarization-community-1"


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    language: str
    text: str
    simultaneous: bool


class TranscriptionError(RuntimeError):
    pass


_TIMER_STACK: list[dict[str, float]] = []


def timer(label: str | None = None, *, enabled: bool = True, min_ms: float = TIMER_MIN_MS):
    def decorator(function):
        if not enabled:
            return function

        timer_label = label or function.__name__

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            started_at = time.perf_counter()
            frame = {"child_ms": 0.0}
            _TIMER_STACK.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                _TIMER_STACK.pop()
                self_ms = max(0.0, elapsed_ms - frame["child_ms"])
                if _TIMER_STACK:
                    _TIMER_STACK[-1]["child_ms"] += elapsed_ms
                if self_ms >= min_ms:
                    print(f"[timer] {timer_label}: {self_ms / 1000:.2f}s")

        return wrapper

    return decorator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an audio/video file to .srt and .txt with local speaker diarization "
            "and per-segment language labels."
        )
    )
    parser.add_argument("media_path", type=Path, help="Path to the input audio or video file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory for the generated .srt and .txt files. Defaults to the input file directory.",
    )
    parser.add_argument(
        "--language",
        help=(
            "Optional Whisper language hint/override such as 'en' or 'de'. "
            "By default, language is detected automatically."
        ),
    )
    parser.add_argument(
        "--whisper-model-path",
        type=Path,
        default=DEFAULT_WHISPER_MODEL_PATH,
        help=(
            "Local path to the faster-whisper large-v3 model directory. "
            f"Defaults to {DEFAULT_WHISPER_MODEL_PATH}."
        ),
    )
    parser.add_argument(
        "--diarization-model-path",
        type=Path,
        default=DEFAULT_DIARIZATION_MODEL_PATH,
        help=(
            "Local path to the pyannote speaker diarization model directory. "
            f"Defaults to {DEFAULT_DIARIZATION_MODEL_PATH}."
        ),
    )
    return parser.parse_args()


def main() -> int:
    total_started_at = time.perf_counter()
    args = parse_args()
    media_path = args.media_path.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else media_path.parent)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        run(
            media_path=media_path,
            output_dir=output_dir,
            language=args.language,
            whisper_model_path=args.whisper_model_path.expanduser().resolve(),
            diarization_model_path=args.diarization_model_path.expanduser().resolve(),
        )
    except TranscriptionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    finally:
        total_elapsed_seconds = time.perf_counter() - total_started_at
        print(f"[timer] total: {total_elapsed_seconds:.2f}s")

    return 0


@timer()
def run(
    media_path: Path,
    output_dir: Path,
    language: str | None,
    whisper_model_path: Path,
    diarization_model_path: Path,
) -> None:
    if not media_path.is_file():
        raise TranscriptionError(f"Input file does not exist: {media_path}")
    if not is_valid_whisper_model_dir(whisper_model_path):
        raise TranscriptionError(
            "Local faster-whisper model directory not found: "
            f"{whisper_model_path}. Run `uv run python .\\setup_models.py` first."
        )
    if not is_valid_pyannote_model_dir(diarization_model_path):
        raise TranscriptionError(
            "Local pyannote diarization checkout is missing or incomplete: "
            f"{diarization_model_path}. Run `uv run python .\\setup_models.py --force` "
            "or pass `--diarization-model-path` to a real local clone."
        )

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise TranscriptionError(
            "ffmpeg is required to normalize audio/video input, but it was not found in PATH."
        )

    print(f"Input: {media_path}")
    print("Preparing audio with ffmpeg...")

    with tempfile.TemporaryDirectory(prefix="whispertransscribe-") as temp_dir:
        normalized_wav = Path(temp_dir) / "normalized.wav"
        normalize_media(ffmpeg_path, media_path, normalized_wav)

        print("Loading normalized waveform...")
        audio = load_wav_mono_float32(normalized_wav)

        device = detect_device()
        compute_type = "float16" if device == "cuda" else "int8"

        speaker_turns, overlap_intervals = run_diarization_stage(
            audio=audio,
            diarization_model_path=diarization_model_path,
            device=device,
        )
        if not speaker_turns:
            raise TranscriptionError("Speaker diarization did not return any speech turns.")

        transcript_segments = run_transcription_stage(
            audio=audio,
            whisper_model_path=whisper_model_path,
            device=device,
            compute_type=compute_type,
            speaker_turns=speaker_turns,
            overlap_intervals=overlap_intervals,
            language_override=language,
        )
        if not transcript_segments:
            raise TranscriptionError("Transcription did not return any text segments.")

        transcript_segments.sort(key=lambda segment: (segment.start, segment.end))
        write_outputs(media_path=media_path, output_dir=output_dir, transcript_segments=transcript_segments)


@timer()
def normalize_media(ffmpeg_path: str, media_path: Path, normalized_wav: Path) -> None:
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        str(normalized_wav),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "ffmpeg failed without stderr output."
        raise TranscriptionError(f"ffmpeg could not normalize the input media.\n{stderr}")


@timer()
def load_wav_mono_float32(wav_path: Path) -> Any:
    import numpy as np

    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw_bytes = wav_file.readframes(frame_count)

    if channels != 1:
        raise TranscriptionError(f"Expected mono WAV after normalization, got {channels} channels.")
    if sample_width != 2:
        raise TranscriptionError(f"Expected 16-bit WAV after normalization, got {sample_width * 8}-bit audio.")
    if sample_rate != TARGET_SAMPLE_RATE:
        raise TranscriptionError(
            f"Expected {TARGET_SAMPLE_RATE} Hz WAV after normalization, got {sample_rate} Hz."
        )

    audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def detect_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@timer()
def load_whisper_model(model_path: Path, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(str(model_path), device=device, compute_type=compute_type)
    except Exception as exc:
        raise TranscriptionError(
            f"Could not load local faster-whisper model from '{model_path}': {exc}"
        ) from exc


def is_valid_whisper_model_dir(model_path: Path) -> bool:
    return model_path.is_dir() and (model_path / "config.json").is_file()


def is_valid_pyannote_model_dir(model_path: Path) -> bool:
    return model_path.is_dir() and (model_path / "config.yaml").is_file()


@timer()
def load_diarization_pipeline(model_path: Path, device: str) -> Any:
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"pyannote\.audio\.core\.io",
        )
        from pyannote.audio import Pipeline

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"pyannote\.audio\.core\.io",
            )
            pipeline = Pipeline.from_pretrained(model_path)
        if pipeline is None:
            raise RuntimeError("Pipeline.from_pretrained returned None.")
        return pipeline.to(torch.device(device))
    except Exception as exc:
        raise TranscriptionError(
            "Could not load the local pyannote diarization pipeline from "
            f"'{model_path}'. Original error: {exc}"
        ) from exc


def configure_diarization_runtime(device: str) -> None:
    import torch

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def release_cuda_resources(device: str) -> None:
    import torch

    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


@timer("run_diarization_stage")
def run_diarization_stage(
    audio: Any,
    diarization_model_path: Path,
    device: str,
) -> tuple[list[SpeakerTurn], list[tuple[float, float]]]:
    configure_diarization_runtime(device)
    print(f"Loading pyannote pipeline from {diarization_model_path} on {device}...")
    diarization_pipeline = load_diarization_pipeline(
        model_path=diarization_model_path,
        device=device,
    )

    try:
        print("Running speaker diarization...")
        return diarize_audio(diarization_pipeline, audio)
    finally:
        del diarization_pipeline
        release_cuda_resources(device)


@timer("run_transcription_stage")
def run_transcription_stage(
    audio: Any,
    whisper_model_path: Path,
    device: str,
    compute_type: str,
    speaker_turns: list[SpeakerTurn],
    overlap_intervals: list[tuple[float, float]],
    language_override: str | None,
) -> list[TranscriptSegment]:
    print(f"Loading faster-whisper model from {whisper_model_path} on {device}...")
    whisper_model = load_whisper_model(
        model_path=whisper_model_path,
        device=device,
        compute_type=compute_type,
    )

    try:
        print("Transcribing speaker turns...")
        return transcribe_speaker_turns(
            whisper_model=whisper_model,
            audio=audio,
            speaker_turns=speaker_turns,
            overlap_intervals=overlap_intervals,
            language_override=language_override,
        )
    finally:
        del whisper_model
        release_cuda_resources(device)


@timer()
def diarize_audio(diarization_pipeline: Any, audio: Any) -> tuple[list[SpeakerTurn], list[tuple[float, float]]]:
    import torch

    waveform = torch.from_numpy(audio).unsqueeze(0)

    try:
        with warnings.catch_warnings(), torch.inference_mode():
            warnings.filterwarnings(
                "ignore",
                message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
                category=Warning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*degrees of freedom is <= 0.*",
                category=UserWarning,
            )
            diarization_output = diarization_pipeline({"waveform": waveform, "sample_rate": TARGET_SAMPLE_RATE})
    except Exception as exc:
        raise TranscriptionError(f"Speaker diarization failed: {exc}") from exc

    diarization = getattr(diarization_output, "exclusive_speaker_diarization", None)
    if diarization is None:
        diarization = getattr(diarization_output, "speaker_diarization", diarization_output)

    overlap_source = getattr(diarization_output, "speaker_diarization", diarization)
    overlap_intervals = extract_overlap_intervals(overlap_source)

    turns: list[SpeakerTurn] = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        if segment.end <= segment.start:
            continue
        turns.append(SpeakerTurn(start=float(segment.start), end=float(segment.end), speaker=str(speaker)))

    turns.sort(key=lambda turn: (turn.start, turn.end, turn.speaker))
    merged_turns = merge_adjacent_turns(turns)

    del overlap_source
    del diarization
    del diarization_output
    del waveform

    return merged_turns, overlap_intervals


def extract_overlap_intervals(diarization: Any) -> list[tuple[float, float]]:
    events: list[tuple[float, int, str]] = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start = float(segment.start)
        end = float(segment.end)
        if end <= start:
            continue
        events.append((start, 1, str(speaker)))
        events.append((end, -1, str(speaker)))

    if not events:
        return []

    events.sort(key=lambda event: (event[0], event[1]))
    active_speakers: set[str] = set()
    overlaps: list[tuple[float, float]] = []
    current_overlap_start: float | None = None

    for time, delta, speaker in events:
        was_overlapping = len(active_speakers) >= 2

        if delta < 0:
            active_speakers.discard(speaker)
        else:
            active_speakers.add(speaker)

        is_overlapping = len(active_speakers) >= 2

        if not was_overlapping and is_overlapping:
            current_overlap_start = time
        elif was_overlapping and not is_overlapping and current_overlap_start is not None:
            if time - current_overlap_start >= MIN_OVERLAP_MARK_SECONDS:
                overlaps.append((current_overlap_start, time))
            current_overlap_start = None

    return overlaps


def merge_adjacent_turns(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    if not turns:
        return []

    merged: list[SpeakerTurn] = [turns[0]]
    for turn in turns[1:]:
        previous = merged[-1]
        gap = turn.start - previous.end
        if turn.speaker == previous.speaker and gap <= MERGE_GAP_SECONDS:
            merged[-1] = SpeakerTurn(
                start=previous.start,
                end=max(previous.end, turn.end),
                speaker=previous.speaker,
            )
            continue
        merged.append(turn)

    return merged


@timer()
def transcribe_speaker_turns(
    whisper_model: Any,
    audio: Any,
    speaker_turns: list[SpeakerTurn],
    overlap_intervals: list[tuple[float, float]],
    language_override: str | None,
) -> list[TranscriptSegment]:
    transcript_segments: list[TranscriptSegment] = []

    for turn in speaker_turns:
        chunk = slice_audio(audio, turn.start, turn.end)
        if len(chunk) == 0:
            continue

        turn_language = language_override or detect_language(whisper_model, chunk)

        try:
            segments, info = whisper_model.transcribe(
                chunk,
                language=turn_language,
                multilingual=True,
                vad_filter=True,
                beam_size=5,
                condition_on_previous_text=False,
            )
        except Exception as exc:
            raise TranscriptionError(
                f"Transcription failed for speaker turn {turn.speaker} at {turn.start:.2f}-{turn.end:.2f}s: {exc}"
            ) from exc

        default_language = language_override or getattr(info, "language", None) or turn_language or "unknown"
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            absolute_start = turn.start + float(segment.start)
            absolute_end = turn.start + float(segment.end)
            segment_audio = slice_audio(audio, absolute_start, absolute_end)
            segment_language = (
                language_override
                or detect_language(whisper_model, segment_audio)
                or default_language
                or "unknown"
            )

            transcript_segments.append(
                TranscriptSegment(
                    start=absolute_start,
                    end=absolute_end,
                    speaker=turn.speaker,
                    language=segment_language,
                    text=text,
                    simultaneous=intersects_overlap(absolute_start, absolute_end, overlap_intervals),
                )
            )

    return transcript_segments


def intersects_overlap(
    start_seconds: float,
    end_seconds: float,
    overlap_intervals: list[tuple[float, float]],
) -> bool:
    for overlap_start, overlap_end in overlap_intervals:
        if min(end_seconds, overlap_end) - max(start_seconds, overlap_start) >= MIN_OVERLAP_MARK_SECONDS:
            return True
    return False


def detect_language(whisper_model: Any, audio_chunk: Any) -> str | None:
    min_samples = int(TARGET_SAMPLE_RATE * MIN_LANGUAGE_DETECTION_SECONDS)
    if len(audio_chunk) < min_samples:
        return None

    try:
        language, _, _ = whisper_model.detect_language(audio=audio_chunk)
    except Exception:
        return None
    return language


def slice_audio(audio: Any, start_seconds: float, end_seconds: float) -> Any:
    start_index = max(0, int(start_seconds * TARGET_SAMPLE_RATE))
    end_index = min(len(audio), int(end_seconds * TARGET_SAMPLE_RATE))
    return audio[start_index:end_index]


@timer()
def write_outputs(media_path: Path, output_dir: Path, transcript_segments: list[TranscriptSegment]) -> None:
    srt_path = output_dir / f"{media_path.stem}.srt"
    txt_path = output_dir / f"{media_path.stem}.txt"

    write_srt(srt_path, transcript_segments)
    write_txt(txt_path, transcript_segments)

    print(f"SRT written to: {srt_path}")
    print(f"TXT written to: {txt_path}")


def write_srt(srt_path: Path, transcript_segments: list[TranscriptSegment]) -> None:
    with srt_path.open("w", encoding="utf-8") as srt_file:
        for index, segment in enumerate(transcript_segments, start=1):
            srt_file.write(f"{index}\n")
            srt_file.write(f"{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}\n")
            srt_file.write(f"{render_output_text(segment)}\n\n")


def write_txt(txt_path: Path, transcript_segments: list[TranscriptSegment]) -> None:
    with txt_path.open("w", encoding="utf-8") as txt_file:
        for segment in transcript_segments:
            txt_file.write(f"{render_output_text(segment)}\n")


def render_output_text(segment: TranscriptSegment) -> str:
    if segment.simultaneous:
        return f"[Simultaneous speech] {segment.text}"
    return segment.text


def format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


def format_clock_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}"


if __name__ == "__main__":
    raise SystemExit(main())
