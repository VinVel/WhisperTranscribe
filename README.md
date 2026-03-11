# WhisperTransscribe

`WhisperTransscribe` is a local transcription CLI for audio and video files.

It uses:

- `faster-whisper` with the `large-v3` model for transcription
- `pyannote.audio` speaker diarization to separate speakers
- `ffmpeg` to normalize input media into a clean mono 16 kHz WAV before inference

The program generates:

- an `.srt` subtitle file
- a `.txt` transcript file

Language detection and speaker detection are used internally during transcription. They are not written into the final output. The only special annotation written to output is `[Simultaneous speech]` when pyannote detects real overlapping speech.

## Features

- Accepts audio or video input by file path
- Works locally after models are prepared on disk
- Uses Whisper automatic language detection by default
- Supports optional language override with `--language`
- Separates speaker turns with pyannote diarization
- Marks overlapping speech when multiple people talk at the same time
- Stores the Hugging Face token in a local `.env` file during setup

## Prerequisites

Install these before using the project:

- Python `3.12`
- `uv`
- `ffmpeg`
- `git`
- `git-lfs`

You also need a Hugging Face account and token for the pyannote offline clone workflow.

## Install Prerequisites

### 1. Python

The project expects Python `3.12.x`.

### 2. uv

Install `uv` from the official Astral instructions, then verify:

```powershell
uv --version
```

### 3. ffmpeg

`ffmpeg` must be available in `PATH`.

Verify:

```powershell
ffmpeg -version
```

### 4. git

Verify:

```powershell
git --version
```

### 5. git-lfs

`pyannote/speaker-diarization-community-1` is cloned through Git LFS.

Verify:

```powershell
git lfs version
```

## Project Setup

From the repository root:

```powershell
uv sync
```

If you use CUDA 12.6, install the matching dependency set you already configured in `pyproject.toml`.

## Model Setup

Run the model setup command once:

```powershell
uv run python .\setup_models.py
```

What happens:

- the script downloads `Systran/faster-whisper-large-v3`
- the script clones `pyannote/speaker-diarization-community-1` as a local offline checkout
- if no token is found, the script prompts once for a Hugging Face token
- the token is stored in `.env` as `HF_TOKEN=...`

The `.env` file is ignored by git.

To force a refresh of local model folders:

```powershell
uv run python .\setup_models.py --force
```

## Running Transcription

Basic usage:

```powershell
uv run python .\main.py "C:\path\to\file.mp4"
```

The output files are written next to the input file by default:

- `file.srt`
- `file.txt`

## Main Parameters

These parameters are supported by `main.py`:

- `media_path`: Path to the input audio or video file.

- `-o`, `--output-dir`: Directory where the generated `.srt` and `.txt` files will be written. If omitted, the files are written next to the input media.

- `--language`: Optional Whisper language hint such as `en` or `de`. If omitted, Whisper detects language automatically.

- `--whisper-model-path`: Path to the local `faster-whisper` model directory. Default: `models/faster-whisper-large-v3`.

- `--diarization-model-path`: Path to the local pyannote diarization checkout. Default: `models/pyannote-speaker-diarization-community-1`.

Examples:

```powershell
uv run python .\main.py "C:\media\sample.mp3" --language de
```

```powershell
uv run python .\main.py "C:\media\sample.mp4" --output-dir "C:\media\out"
```

```powershell
uv run python .\main.py "C:\media\sample.wav" --whisper-model-path "D:\models\faster-whisper-large-v3" --diarization-model-path "D:\models\pyannote-speaker-diarization-community-1"
```

## Setup Parameters

These parameters are supported by `setup_models.py`:

- `--models-dir`: Base directory where the local model folders are created. Default: `models`.

- `--hf-token`: Optional Hugging Face token. If omitted, the script checks environment variables and `.env`, then prompts interactively if needed.

- `--force`: Re-downloads Whisper and re-clones the pyannote checkout even if model folders already exist.

- `--skip-whisper`: Skip the Whisper model download step.

- `--skip-diarization`: Skip the pyannote clone step.

Examples:

```powershell
uv run python .\setup_models.py --force
```

```powershell
uv run python .\setup_models.py --skip-whisper
```

```powershell
uv run python .\setup_models.py --models-dir "D:\models"
```

## Output Behavior

### `.srt`

- subtitle timestamps in SRT format
- plain transcript text
- `[Simultaneous speech]` prefix only when overlap is detected

### `.txt`

- plain transcript text, one segment per line
- `[Simultaneous speech]` prefix only when overlap is detected

The output does not include:

- speaker names
- detected language labels

Those are used internally only.

## Notes

- `ffmpeg` is still required even though `faster-whisper` can decode many formats on its own. This project uses `ffmpeg` explicitly for stable normalization and to give both diarization and transcription the same audio input.
- pyannote runs on the local model checkout, not a remote inference API.
- Whisper language detection can fail on extremely short segments. In that case the code lets Whisper auto-detect instead of forcing an invalid language code.

## Troubleshooting

### `config.yaml` missing in pyannote folder

Your pyannote checkout is incomplete. Re-run:

```powershell
uv run python .\setup_models.py --force --skip-whisper
```

### `ffmpeg` not found

Install `ffmpeg` and make sure it is available in `PATH`.

### Slow or failed model download

Use a valid Hugging Face token. The setup script can prompt for it and save it to `.env`.

### Windows delete error during `--force`

The setup script now handles read-only Git files during cleanup. Retry the same command:

```powershell
uv run python .\setup_models.py --force
```
