#!/usr/bin/env python3
"""
ElevenLabs bulk TTS — interactive setup, parallel generation.

CSV format: only a `text` column is needed.
  - Output filename is derived from the text with all [tags] stripped out.
  - Voice, speed, stability, and workers are chosen interactively at startup.

Voices in .env: add entries like  Natasha=<voice_id>  (any key besides ELEVENLABS_API_KEY).
ElevenLabs native tags (eleven_v3 model) — write directly in text:
  [happy] [sad] [excited] [calm] [nervous] [frustrated] [angry] [tired]
  [cheerfully] [flatly] [deadpan] [playfully] [hesitant] [regretful]
  [sigh] [laughs] [gulps] [gasps] [whispers] [clears throat] [pauses]
  [hesitates] [stammers] [resigned tone] [shouts] [quietly] [loudly] [rushed]
  [break] / [break 1.5s]  →  SSML <break time="..." /> pause

Silence check:
  After each generation the audio is inspected. If leading silence is more than
  50% of the total duration the file is automatically regenerated once. If the
  retry still has the issue it is kept as-is (so you never lose the file).
"""

from __future__ import annotations

import csv
import io
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
_RESERVED_KEYS = {"ELEVENLABS_API_KEY", "XI_API_KEY", "ELEVENLABS_DEFAULT_VOICE_ID"}

try:
    from dotenv import load_dotenv, dotenv_values
    load_dotenv(_SCRIPT_DIR / ".env")
    load_dotenv(_SCRIPT_DIR.parent / ".env")
except ImportError:
    dotenv_values = None  # type: ignore

API_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "output_audio"
DEFAULT_MODEL_ID = "eleven_v3"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_STABILITY = 0.5      # 0.0 = more expressive/variable, 1.0 = very stable/consistent
SILENCE_THRESH_DB = -40.0    # dBFS below which audio is considered silence
MAX_LEADING_SILENCE_RATIO = 0.5  # retry if leading silence > 50% of total duration

_print_lock = threading.Lock()


def safe_print(*args, **kwargs) -> None:
    with _print_lock:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# .env helpers
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = (
        os.environ.get("ELEVENLABS_API_KEY") or
        os.environ.get("XI_API_KEY") or
        ""
    ).strip()
    if not key:
        print(
            "Error: ELEVENLABS_API_KEY not set. Add it to .env as ELEVENLABS_API_KEY=your_key",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def list_voices_from_env() -> list[tuple[str, str]]:
    """
    Read voice name → voice_id pairs from the .env file directly.
    Any key that isn't a reserved API-key name is treated as a voice.
    """
    env_paths = [_SCRIPT_DIR / ".env", _SCRIPT_DIR.parent / ".env"]

    if dotenv_values is not None:
        for path in env_paths:
            if path.is_file():
                values = dotenv_values(path)
                voices = [
                    (k, v.strip())
                    for k, v in values.items()
                    if k not in _RESERVED_KEYS
                    and not k.startswith("ELEVENLABS_")
                    and v and v.strip()
                ]
                if voices:
                    return sorted(voices, key=lambda x: x[0].lower())

    # Fallback: scan environment for ELEVENLABS_VOICE_* pattern
    prefix = "ELEVENLABS_VOICE_"
    voices = [
        (k[len(prefix):], v.strip())
        for k, v in os.environ.items()
        if k.startswith(prefix) and v and v.strip()
    ]
    return sorted(voices, key=lambda x: x[0].lower())


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def prompt_voice(voices: list[tuple[str, str]]) -> tuple[str, str]:
    """Show voice list, return (label, voice_id) chosen by user."""
    print("\n--- Voice Selection ---")
    for i, (label, _) in enumerate(voices, 1):
        print(f"  {i}. {label}")
    while True:
        raw = input(f"\nChoose voice [1-{len(voices)}]: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(voices):
                label, voice_id = voices[idx - 1]
                print(f"  Using: {label}")
                return label, voice_id
        except ValueError:
            pass
        print(f"  Enter a number between 1 and {len(voices)}.")


def prompt_speed() -> float | None:
    """Ask for speech speed. Returns float or None (API default = 1.0)."""
    print("\n--- Speech Speed ---")
    print("  0.5 = slowest  |  1.0 = normal  |  2.0 = fastest")
    raw = input("Enter speed [press Enter to use default 1.0]: ").strip()
    if not raw:
        print("  Using default speed.")
        return None
    try:
        s = round(max(0.5, min(2.0, float(raw))), 2)
        print(f"  Speed: {s}")
        return s
    except ValueError:
        print("  Invalid input — using default speed.")
        return None


def prompt_stability() -> float:
    """Ask for voice stability. Returns float 0.0–1.0."""
    print("\n--- Voice Stability ---")
    print("  0.0 = more expressive & variable  |  1.0 = very stable & consistent")
    print(f"  Recommended for eleven_v3: 0.3–0.6  (default: {DEFAULT_STABILITY})")
    raw = input(f"Enter stability [press Enter for {DEFAULT_STABILITY}]: ").strip()
    if not raw:
        print(f"  Using default stability: {DEFAULT_STABILITY}")
        return DEFAULT_STABILITY
    try:
        s = round(max(0.0, min(1.0, float(raw))), 2)
        print(f"  Stability: {s}")
        return s
    except ValueError:
        print(f"  Invalid input — using default stability: {DEFAULT_STABILITY}")
        return DEFAULT_STABILITY


def prompt_workers() -> int:
    """Ask how many parallel workers to use."""
    print("\n--- Parallel Workers ---")
    raw = input("Number of parallel workers [press Enter for 5]: ").strip()
    if not raw:
        return 5
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return 5


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Convert [break] / [break Ns] shortcuts to SSML <break time="..." />."""
    if not text:
        return text

    def replace_break(m: re.Match) -> str:
        g = m.group(1)
        if g:
            try:
                sec = max(0.1, min(3.0, float(g.strip().rstrip("s"))))
                return f'<break time="{sec}s" />'
            except ValueError:
                pass
        return '<break time="0.5s" />'

    text = re.sub(r"\[break\s+([\d.]+s?)\s*\]", replace_break, text, flags=re.IGNORECASE)
    text = re.sub(r"\[break\s*\]", '<break time="0.5s" />', text, flags=re.IGNORECASE)
    return text


def make_output_name(text: str) -> str:
    """
    Derive a clean filename from the raw text:
      - Strip all [tag] brackets (emotion/delivery tags).
      - Strip SSML / XML tags.
      - Remove leading punctuation, collapse whitespace.
      - Sanitize for filesystem. Max 100 chars.
    """
    name = re.sub(r"\[.*?\]", "", text)        # remove [happy], [break 1s], etc.
    name = re.sub(r"<[^>]+>", "", name)         # remove <break ... />
    name = name.strip().strip(".,!?;:-")
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r'[<>:"/\\|?*]', "", name)   # filesystem-safe
    name = name[:100].strip()
    return name or "audio"


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------

def leading_silence_ratio(audio_bytes: bytes, ext: str) -> float:
    """
    Returns the fraction (0.0–1.0) of the total audio that is leading silence
    before the first speech is detected.

    Uses pydub + ffmpeg. Returns 0.0 if pydub is unavailable or the check fails
    so the file is always kept in that case.
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent

        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
        total_ms = len(audio)
        if total_ms == 0:
            return 0.0

        non_silent = detect_nonsilent(
            audio,
            min_silence_len=100,       # ms of continuous silence to count
            silence_thresh=SILENCE_THRESH_DB,
        )
        if not non_silent:
            return 1.0  # entirely silent

        first_speech_ms = non_silent[0][0]
        return first_speech_ms / total_ms
    except Exception:
        return 0.0  # never block generation if check fails


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def generate_speech(
    api_key: str,
    voice_id: str,
    text: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    speed: float | None = None,
    stability: float = DEFAULT_STABILITY,
) -> bytes:
    url = f"{API_BASE}/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload: dict = {"text": text, "model_id": model_id}

    # Always send voice_settings so stability is applied even at default speed
    voice_settings: dict = {"stability": stability}
    if speed is not None and speed != 1.0:
        voice_settings["speed"] = speed
    payload["voice_settings"] = voice_settings

    r = requests.post(
        url,
        headers=headers,
        params={"output_format": output_format},
        json=payload,
        timeout=120,
    )
    if r.status_code == 401:
        raise RuntimeError(
            "401 Unauthorized — check ELEVENLABS_API_KEY in .env. "
            "Get your key at https://elevenlabs.io/app/settings/api-keys"
        )
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# Per-row worker
# ---------------------------------------------------------------------------

def process_one_row(
    api_key: str,
    text_raw: str,
    output_dir: Path,
    voice_id: str,
    model_id: str,
    output_format: str,
    speed: float | None,
    stability: float,
    idx: int,
    total: int,
) -> str:
    """Generate TTS for one text entry and save to disk. Returns saved filename."""
    text = text_raw.strip()
    if not text:
        raise ValueError("Empty text")

    output_name = make_output_name(text)
    text = preprocess_text(text)
    if not text:
        raise ValueError("Empty text after preprocessing")

    ext = "mp3" if "mp3" in output_format else "wav" if "wav" in output_format else "ogg"

    def _call_api() -> bytes:
        return generate_speech(
            api_key, voice_id, text,
            model_id=model_id, output_format=output_format,
            speed=speed, stability=stability,
        )

    audio_bytes = _call_api()

    # --- Silence check: retry once if leading silence > 50% of total duration ---
    ratio = leading_silence_ratio(audio_bytes, ext)
    if ratio > MAX_LEADING_SILENCE_RATIO:
        safe_print(
            f"  [{idx}/{total}] Leading silence {ratio:.0%} detected — retrying: {output_name}.{ext}"
        )
        retry_bytes = _call_api()
        retry_ratio = leading_silence_ratio(retry_bytes, ext)
        if retry_ratio > MAX_LEADING_SILENCE_RATIO:
            safe_print(
                f"  [{idx}/{total}] Retry still has {retry_ratio:.0%} leading silence — keeping retry: {output_name}.{ext}"
            )
        else:
            safe_print(
                f"  [{idx}/{total}] Retry OK ({retry_ratio:.0%} silence): {output_name}.{ext}"
            )
        audio_bytes = retry_bytes  # always use the retry result

    out_path = output_dir / f"{output_name}.{ext}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    safe_print(f"  [{idx}/{total}] Saved: {out_path.name}")
    return out_path.name


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_texts_from_csv(csv_path: Path) -> list[str]:
    texts: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            if text:
                texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ElevenLabs bulk TTS — interactive voice/speed picker, parallel generation."
    )
    parser.add_argument(
        "csv", type=Path, nargs="?", default=None,
        help="CSV file with a 'text' column (required)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save audio files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_ID,
        help=f"ElevenLabs model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output-format", type=str, default=DEFAULT_OUTPUT_FORMAT,
        help=f"Audio format (default: {DEFAULT_OUTPUT_FORMAT})",
    )
    args = parser.parse_args()

    csv_path: Path | None = args.csv
    if not csv_path or not csv_path.is_file():
        print("Usage: python elevenlabs_bulk_tts.py <input.csv>", file=sys.stderr)
        print("  CSV must have a 'text' column.", file=sys.stderr)
        sys.exit(1)

    # --- Setup ---
    api_key = get_api_key()

    voices = list_voices_from_env()
    if not voices:
        print(
            "Error: No voices found in .env.\n"
            "Add lines like:  Natasha=<voice_id>  to your .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    voice_label, voice_id = prompt_voice(voices)
    speed = prompt_speed()
    stability = prompt_stability()
    workers = prompt_workers()

    # --- Load CSV ---
    texts = load_texts_from_csv(csv_path)
    if not texts:
        print("No text rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Voice     : {voice_label}")
    print(f"Speed     : {speed if speed is not None else '1.0 (default)'}")
    print(f"Stability : {stability}")
    print(f"Model     : {args.model}")
    print(f"Rows      : {len(texts)}")
    print(f"Workers   : {workers}")
    print(f"Output    : {output_dir}")
    print(f"{'='*50}\n")

    # --- Generate in parallel ---
    total = len(texts)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_one_row,
                api_key,
                text,
                output_dir,
                voice_id,
                args.model,
                args.output_format,
                speed,
                stability,
                i + 1,
                total,
            ): text
            for i, text in enumerate(texts)
        }
        failed = 0
        for future in as_completed(futures):
            text = futures[future]
            try:
                future.result()
            except Exception as e:
                failed += 1
                safe_print(
                    f"  ERROR ({text[:50]!r}): {e}",
                    file=sys.stderr,
                )

    print()
    if failed:
        print(f"Finished with {failed} error(s).", file=sys.stderr)
        sys.exit(1)
    print(f"All {total} files saved to: {output_dir}")


if __name__ == "__main__":
    main()
