from pathlib import Path
from typing import List, Optional

import subprocess


def list_voice_models(voices_dir: Path) -> List[Path]:
    if not voices_dir.is_dir():
        return []
    return sorted(voices_dir.glob("*.onnx"))


def choose_voice_model_interactive(voices_dir: Path) -> Optional[Path]:
    voices = list_voice_models(voices_dir)
    if not voices:
        print(f"No .onnx voices found in {voices_dir}, TTS will use placeholder.")
        return None

    print(f"Found the following TTS voices in {voices_dir}:")
    for idx, v in enumerate(voices):
        print(f"[{idx}] {v.name}")

    while True:
        choice = input("Select voice index (or leave blank for no TTS): ").strip()
        if choice == "":
            return None
        if not choice.isdigit():
            print("Enter a number or blank.")
            continue
        idx = int(choice)
        if 0 <= idx < len(voices):
            return voices[idx]
        print("Out of range.")


def text_to_speech(text: str, out_path: Path, voice_model: Optional[Path]) -> None:
    """
    Convert text → WAV using Piper only.

    If voice_model is None, write a small placeholder file instead.
    """
    if voice_model is None:
        out_path.write_bytes(b"WAV PLACEHOLDER\n")
        return

    print(f"  [TTS] Using voice: {voice_model.name}")
    try:
        subprocess.run(
            [
                "piper",
                "-m",
                str(voice_model),
                "-f",
                str(out_path),
            ],
            input=text.encode("utf-8"),
            check=True,
        )
    except FileNotFoundError:
        print("  [TTS] Piper not found on PATH. Writing placeholder instead.")
        out_path.write_bytes(b"WAV PLACEHOLDER (piper not found)\n")
    except subprocess.CalledProcessError as e:
        print(f"  [TTS] Piper failed ({e}). Writing placeholder instead.")
        out_path.write_bytes(b"WAV PLACEHOLDER (piper error)\n")
