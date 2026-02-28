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
  retry still has the issue it is saved to a separate "needs_review" folder
  (or the main folder, your choice). A full stats summary is printed at the end.
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

# ElevenLabs models: (display_name, model_id, description, supports_speed, supports_tags)
KNOWN_MODELS: list[tuple[str, str, str, bool, bool]] = [
    ("eleven_v3",              "eleven_v3",              "Latest — emotion/style tags + speed param  [DEFAULT]", True,  True),
    ("eleven_multilingual_v2", "eleven_multilingual_v2", "High quality, multilingual — speed param, no tags",    True,  False),
    ("eleven_turbo_v2_5",      "eleven_turbo_v2_5",      "Fast & low-latency, multilingual — no speed/tags",    False, False),
    ("eleven_turbo_v2",        "eleven_turbo_v2",        "Fast & low-latency, English — no speed/tags",         False, False),
    ("eleven_monolingual_v1",  "eleven_monolingual_v1",  "Legacy English-only — no speed/tags",                 False, False),
]

# Quick lookup for capability flags by model_id
_MODEL_SUPPORTS_SPEED: dict[str, bool] = {m[1]: m[3] for m in KNOWN_MODELS}
_MODEL_SUPPORTS_TAGS:  dict[str, bool] = {m[1]: m[4] for m in KNOWN_MODELS}

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


def prompt_model() -> str:
    """Show ElevenLabs model list, return chosen model_id."""
    print("\n--- Model Selection ---")
    for i, (name, _, desc, sup_speed, sup_tags) in enumerate(KNOWN_MODELS, 1):
        caps = []
        if sup_tags:
            caps.append("emotion tags")
        if sup_speed:
            caps.append("API speed control")
        cap_str = f"  [supports: {', '.join(caps)}]" if caps else "  [no speed/tag support]"
        print(f"  {i}. {name}")
        print(f"     {desc}")
        print(f"    {cap_str}")
    while True:
        raw = input(f"\nChoose model [1-{len(KNOWN_MODELS)}, press Enter for 1]: ").strip() or "1"
        try:
            idx = int(raw)
            if 1 <= idx <= len(KNOWN_MODELS):
                name, model_id, _, sup_speed, sup_tags = KNOWN_MODELS[idx - 1]
                print(f"  Using: {name}")
                if not sup_tags:
                    print("  ⚠  This model ignores emotion tags like [happy], [calm] etc.")
                if not sup_speed:
                    print("  ⚠  This model does not support API speed control (use time-stretch instead).")
                return model_id
        except ValueError:
            pass
        print(f"  Enter a number between 1 and {len(KNOWN_MODELS)}.")


def prompt_speed(model_id: str) -> float | None:
    """
    Ask for API-level speech speed.
    Only sent to models that support it (eleven_v3, eleven_multilingual_v2).
    For ultra-slow output use the post-process time-stretch (prompt_stretch).
    """
    print("\n--- Speech Speed (API) ---")
    if not _MODEL_SUPPORTS_SPEED.get(model_id, False):
        print(f"  ⚠  '{model_id}' does not support API speed — skipped.")
        print("     Use the time-stretch option below to slow down audio.")
        return None
    print("  0.25 = very slow  |  0.7 = slow  |  1.0 = normal  |  2.0 = fast")
    print("  Note: values below 0.7 can sound unnatural from the API.")
    print("        Use the time-stretch option below for smoother slow-down.")
    raw = input("Enter speed [press Enter to use default 1.0]: ").strip()
    if not raw:
        print("  Using default speed.")
        return None
    try:
        s = round(max(0.25, min(4.0, float(raw))), 2)
        print(f"  API speed: {s}")
        return s
    except ValueError:
        print("  Invalid input — using default speed.")
        return None


def prompt_stretch() -> float | None:
    """
    Ask for optional post-processing time-stretch via ffmpeg.
    This slows down (or speeds up) the final audio AFTER generation.
    0.5 = half speed (2× longer), 0.25 = quarter speed (4× longer).
    Returns None if skipped.
    """
    print("\n--- Post-Process Time Stretch (ffmpeg) ---")
    print("  Slows down the generated audio using ffmpeg after download.")
    print("  0.25 = quarter speed (very slow)  |  0.5 = half speed  |  1.0 = no change")
    print("  Requires ffmpeg installed. Press Enter to skip.")
    raw = input("Enter stretch factor [press Enter to skip]: ").strip()
    if not raw:
        print("  No time stretch.")
        return None
    try:
        s = round(max(0.1, min(2.0, float(raw))), 3)
        if s == 1.0:
            print("  Factor is 1.0 — no stretch applied.")
            return None
        print(f"  Time stretch factor: {s}  (audio will be {1/s:.1f}× longer)")
        return s
    except ValueError:
        print("  Invalid input — no time stretch.")
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


def prompt_unfixed_folder(output_dir: Path) -> Path | None:
    """
    Ask whether audios that still have leading silence after retry should go
    into a separate folder. Returns that folder path, or None for same folder.
    """
    print("\n--- Unfixed Silence Files ---")
    print("  Some audios may still have leading silence even after retry.")
    print("  Where should those files be saved?")
    print("  1. Separate folder (needs_review/)")
    print("  2. Same output folder")
    while True:
        raw = input("Choose [1/2, press Enter for 1]: ").strip() or "1"
        if raw == "1":
            folder = output_dir / "needs_review"
            print(f"  Unfixed files → {folder}")
            return folder
        if raw == "2":
            print("  Unfixed files → same output folder")
            return None
        print("  Enter 1 or 2.")


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
# Post-processing: time stretch via ffmpeg atempo
# ---------------------------------------------------------------------------

def apply_time_stretch(audio_bytes: bytes, ext: str, factor: float) -> bytes:
    """
    Slow down (or speed up) audio using ffmpeg's atempo filter.
    factor < 1.0 = slower (e.g. 0.5 = half speed, 2× longer).
    factor > 1.0 = faster.

    atempo only accepts 0.5–2.0 per filter, so we chain them for extreme values:
      0.25x → atempo=0.5,atempo=0.5
      0.1x  → atempo=0.5,atempo=0.5,atempo=0.4
    """
    import subprocess
    import tempfile

    # Build the atempo filter chain
    filters: list[str] = []
    remaining = factor
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.6f}")
    filter_str = ",".join(filters)

    inp_file = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
    out_file = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
    try:
        inp_file.write(audio_bytes)
        inp_file.close()
        out_file.close()
        subprocess.run(
            ["ffmpeg", "-y", "-i", inp_file.name, "-af", filter_str, out_file.name],
            check=True,
            capture_output=True,
        )
        with open(out_file.name, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(inp_file.name)
        except OSError:
            pass
        try:
            os.unlink(out_file.name)
        except OSError:
            pass


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

    voice_settings: dict = {"stability": stability}
    # Only send speed if the chosen model actually supports it — turbo models reject it with 400
    if speed is not None and speed != 1.0 and _MODEL_SUPPORTS_SPEED.get(model_id, False):
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
    if not r.ok:
        # Include the API's own error message for easier debugging
        try:
            detail = r.json().get("detail", {})
            msg = detail.get("message", r.text) if isinstance(detail, dict) else str(detail)
        except Exception:
            msg = r.text[:300]
        raise RuntimeError(f"{r.status_code} {r.reason}: {msg}")
    return r.content


# ---------------------------------------------------------------------------
# Per-row worker
# ---------------------------------------------------------------------------

class RowResult:
    """Outcome of processing one CSV row."""
    __slots__ = ("filename", "retried", "fixed", "unfixed")

    def __init__(self, filename: str, retried: bool = False, fixed: bool = False, unfixed: bool = False):
        self.filename = filename
        self.retried = retried   # True if a retry was attempted
        self.fixed   = fixed     # True if retry resolved the silence
        self.unfixed = unfixed   # True if retry did NOT resolve the silence


def process_one_row(
    api_key: str,
    text_raw: str,
    output_dir: Path,
    unfixed_dir: Path | None,
    voice_id: str,
    model_id: str,
    output_format: str,
    speed: float | None,
    stability: float,
    stretch: float | None,
    idx: int,
    total: int,
) -> RowResult:
    """Generate TTS for one text entry and save to disk. Returns a RowResult."""
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
    retried = fixed = unfixed = False

    # --- Silence check: retry once if leading silence > 50% of total duration ---
    ratio = leading_silence_ratio(audio_bytes, ext)
    if ratio > MAX_LEADING_SILENCE_RATIO:
        retried = True
        safe_print(
            f"  [{idx}/{total}] Leading silence {ratio:.0%} detected — retrying: {output_name}.{ext}"
        )
        retry_bytes = _call_api()
        retry_ratio = leading_silence_ratio(retry_bytes, ext)
        audio_bytes = retry_bytes  # always use the retry result

        if retry_ratio > MAX_LEADING_SILENCE_RATIO:
            unfixed = True
            safe_print(
                f"  [{idx}/{total}] Retry still has {retry_ratio:.0%} leading silence — flagged: {output_name}.{ext}"
            )
        else:
            fixed = True
            safe_print(
                f"  [{idx}/{total}] Retry OK ({retry_ratio:.0%} silence): {output_name}.{ext}"
            )

    # --- Optional post-process time stretch ---
    if stretch is not None:
        try:
            audio_bytes = apply_time_stretch(audio_bytes, ext, stretch)
        except Exception as e:
            safe_print(f"  [{idx}/{total}] WARNING: time stretch failed ({e}) — saving original speed")

    # Route unfixed files to the separate folder if configured
    save_dir = (unfixed_dir if unfixed and unfixed_dir is not None else output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = save_dir / f"{output_name}.{ext}"
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    safe_print(f"  [{idx}/{total}] Saved: {out_path.relative_to(out_path.parent.parent)}")
    return RowResult(filename=out_path.name, retried=retried, fixed=fixed, unfixed=unfixed)


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
    model_id = prompt_model()
    speed = prompt_speed(model_id)
    stretch = prompt_stretch()
    stability = prompt_stability()
    workers = prompt_workers()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unfixed_dir = prompt_unfixed_folder(output_dir)

    # --- Load CSV ---
    texts = load_texts_from_csv(csv_path)
    if not texts:
        print("No text rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"Voice     : {voice_label}")
    print(f"Model     : {model_id}")
    print(f"API Speed : {speed if speed is not None else '1.0 (default)'}")
    print(f"Stretch   : {stretch if stretch is not None else 'none'}")
    print(f"Stability : {stability}")
    print(f"Rows      : {len(texts)}")
    print(f"Workers   : {workers}")
    print(f"Output    : {output_dir}")
    if unfixed_dir:
        print(f"Unfixed → : {unfixed_dir}")
    print(f"{'='*50}\n")

    # --- Generate in parallel ---
    total = len(texts)
    results: list[RowResult] = []
    api_errors = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_one_row,
                api_key,
                text,
                output_dir,
                unfixed_dir,
                voice_id,
                model_id,
                args.output_format,
                speed,
                stability,
                stretch,
                i + 1,
                total,
            ): text
            for i, text in enumerate(texts)
        }
        for future in as_completed(futures):
            text = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                api_errors += 1
                safe_print(f"  ERROR ({text[:50]!r}): {e}", file=sys.stderr)

    # --- Final stats ---
    n_retried = sum(1 for r in results if r.retried)
    n_fixed   = sum(1 for r in results if r.fixed)
    n_unfixed = sum(1 for r in results if r.unfixed)
    n_clean   = total - api_errors - n_retried

    print()
    print(f"{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    print(f"  Total submitted     : {total}")
    print(f"  Clean (no issue)    : {n_clean}")
    print(f"  Issues detected     : {n_retried}")
    print(f"    ↳ Fixed by retry  : {n_fixed}")
    print(f"    ↳ Still unfixed   : {n_unfixed}")
    if api_errors:
        print(f"  API errors (failed) : {api_errors}")
    if n_unfixed and unfixed_dir:
        print(f"\n  Unfixed files saved to: {unfixed_dir}")
    elif n_unfixed:
        print(f"\n  Unfixed files are in the main output folder (flagged in log above).")
    print(f"{'='*50}")
    print(f"\n  All audio saved to: {output_dir}")

    if api_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
