from __future__ import annotations

import argparse
import getpass
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from huggingface_hub import login, snapshot_download


DEFAULT_MODELS_DIR = Path("models")
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_WHISPER_DIRNAME = "faster-whisper-large-v3"
DEFAULT_DIARIZATION_DIRNAME = "pyannote-speaker-diarization-community-1"
WHISPER_REPO_ID = "Systran/faster-whisper-large-v3"
DIARIZATION_REPO_URL = "https://hf.co/pyannote/speaker-diarization-community-1"


class SetupError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download local model assets for faster-whisper and pyannote diarization."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Base directory for downloaded models. Defaults to {DEFAULT_MODELS_DIR}.",
    )
    parser.add_argument(
        "--hf-token",
        help=(
            "Optional Hugging Face token used to authenticate git access for the pyannote "
            "offline clone workflow."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh download even if the local model directories already exist.",
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip downloading the faster-whisper model.",
    )
    parser.add_argument(
        "--skip-diarization",
        action="store_true",
        help="Skip cloning the pyannote diarization checkout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models_dir = args.models_dir.expanduser().resolve()
    env_file = Path.cwd() / DEFAULT_ENV_FILE

    try:
        if args.skip_whisper and args.skip_diarization:
            raise SetupError("Nothing to do: both downloads were skipped.")

        load_env_file(env_file)
        token = resolve_or_prompt_hf_token(args.hf_token, env_file)
        models_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_whisper:
            download_whisper(
                local_dir=models_dir / DEFAULT_WHISPER_DIRNAME,
                token=token,
                force=args.force,
            )

        if not args.skip_diarization:
            clone_pyannote_checkout(
                local_dir=models_dir / DEFAULT_DIARIZATION_DIRNAME,
                token=token,
                force=args.force,
            )
    except SetupError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130

    return 0


def resolve_hf_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token

    for env_name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN", "PYANNOTE_AUTH_TOKEN"):
        token = os.getenv(env_name)
        if token:
            return token

    return None


def resolve_or_prompt_hf_token(cli_token: str | None, env_file: Path) -> str:
    token = resolve_hf_token(cli_token)
    if token:
        return token

    if not sys.stdin.isatty():
        raise SetupError(
            "A Hugging Face token is required for the pyannote offline clone, but no interactive prompt is available."
        )

    token = getpass.getpass("Hugging Face token for pyannote checkout: ").strip()
    if not token:
        raise SetupError("No Hugging Face token was entered.")

    upsert_env_value(env_file, "HF_TOKEN", token)
    os.environ["HF_TOKEN"] = token
    print(f"Stored HF_TOKEN in {env_file}")
    return token


def load_env_file(env_file: Path) -> None:
    if not env_file.is_file():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def upsert_env_value(env_file: Path, key: str, value: str) -> None:
    lines: list[str] = []
    replaced = False

    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()

    updated_lines: list[str] = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            updated_lines.append(f"{key}={value}")
            replaced = True
        else:
            updated_lines.append(line)

    if not replaced:
        updated_lines.append(f"{key}={value}")

    env_file.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def download_whisper(local_dir: Path, token: str | None, force: bool) -> None:
    if local_dir.exists() and not force:
        print(f"Skipping {WHISPER_REPO_ID}; already present at {local_dir}")
        return

    print(f"Downloading {WHISPER_REPO_ID} -> {local_dir}")

    try:
        if local_dir.exists() and force:
            remove_tree(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=WHISPER_REPO_ID,
            local_dir=local_dir,
            token=token,
            force_download=force,
            local_files_only=False,
        )
    except Exception as exc:
        if local_dir.exists() and not any(local_dir.iterdir()):
            local_dir.rmdir()
        raise SetupError(f"Could not download {WHISPER_REPO_ID}: {exc}") from exc


def clone_pyannote_checkout(local_dir: Path, token: str | None, force: bool) -> None:
    if is_valid_pyannote_checkout(local_dir) and not force:
        print(f"Skipping pyannote diarization checkout; already present at {local_dir}")
        return

    if local_dir.exists():
        if force:
            remove_tree(local_dir)
        else:
            raise SetupError(
                f"Found an incomplete pyannote diarization directory at {local_dir}. "
                "Re-run with --force to replace it."
            )

    if token:
        try:
            login(token=token, add_to_git_credential=True, skip_if_logged_in=True)
        except Exception as exc:
            raise SetupError(f"Could not configure Hugging Face git credentials: {exc}") from exc

    run_git(["git", "lfs", "install"])
    run_git(["git", "clone", DIARIZATION_REPO_URL, str(local_dir)])

    if not is_valid_pyannote_checkout(local_dir):
        raise SetupError(
            f"Pyannote checkout at {local_dir} does not contain config.yaml after clone."
        )


def is_valid_pyannote_checkout(local_dir: Path) -> bool:
    return local_dir.is_dir() and (local_dir / "config.yaml").is_file()


def run_git(command: list[str]) -> None:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "git command failed."
        raise SetupError(f"{' '.join(command)} failed: {stderr}")


def remove_tree(path: Path) -> None:
    def onexc(function: object, target: str, excinfo: tuple[type[BaseException], BaseException, object]) -> None:
        try:
            os.chmod(target, stat.S_IWRITE)
            function(target)
        except Exception as exc:
            raise SetupError(f"Could not remove {path}: {exc}") from exc

    try:
        shutil.rmtree(path, onexc=onexc)
    except FileNotFoundError:
        return
    except SetupError:
        raise
    except Exception as exc:
        raise SetupError(f"Could not remove {path}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
