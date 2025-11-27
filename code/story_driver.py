#!/usr/bin/env python3
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import subprocess

LMSTUDIO_BASE = "http://localhost:1234"  # LM Studio server: lms server start
REST_MODELS = f"{LMSTUDIO_BASE}/api/v0/models"
REST_CHAT = f"{LMSTUDIO_BASE}/api/v0/chat/completions"

# Rough tokens-per-word factor (depends on language/content, but this is fine)
TOKENS_PER_WORD = 1.4  # NEW: used to map words → tokens


# ---------- Data classes ----------

@dataclass
class RunChapterInfo:
    chapter_id: int
    chapter_slug: str
    text_file: str
    audio_file: str
    usage: Dict[str, Any]


@dataclass
class RunInfo:
    run_id: str
    timestamp: str
    model_id: str
    temperature: float
    max_tokens_ceiling: int  # NEW: ceiling, not per-chapter exact
    default_target_words: int  # NEW
    chapters: List[RunChapterInfo]


# ---------- LM Studio helpers ----------

def list_lmstudio_models() -> List[Dict[str, Any]]:
    """Return list of models from LM Studio REST API."""
    resp = requests.get(REST_MODELS, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def choose_model_interactive(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple CLI picker for model."""
    print("Available LM Studio models:")
    usable = [m for m in models if m.get("type") == "llm"]
    for idx, m in enumerate(usable):
        print(
            f"[{idx}] {m['id']:<40} "
            f"(arch={m.get('arch')}, quant={m.get('quantization')}, "
            f"ctx={m.get('max_context_length')})"
        )
    while True:
        choice = input("Select model index: ").strip()
        if not choice.isdigit():
            print("Enter a number.")
            continue
        idx = int(choice)
        if 0 <= idx < len(usable):
            return usable[idx]
        print("Out of range.")


def generate_chapter(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> (str, Dict[str, Any]):
    """Call LM Studio chat completions and return (text, usage)."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,  # may vary per chapter based on target words
        "stream": False,
    }
    resp = requests.post(REST_CHAT, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, usage


# ---------- TTS stub ----------

def text_to_speech(text: str, out_path: Path):
    """
    Stub: convert text → MP3.

    Replace this with your preferred engine, e.g. Piper:
        echo "text" | piper -m voice.onnx -f out_path

    For now this just writes a dummy file to prove the pipeline works.
    """
    out_path.write_bytes(b"MP3 PLACEHOLDER\n")
    # Example Piper call (commented out):
    # proc = subprocess.run(
    #     ["piper", "-m", "en_GB-some-voice.onnx", "-f", str(out_path)],
    #     input=text.encode("utf-8"),
    #     check=True,
    # )


# ---------- Story JSON helpers ----------

def load_story(story_path: Path) -> Dict[str, Any]:
    with story_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_story(story_path: Path, story: Dict[str, Any]):
    tmp_path = story_path.with_suffix(".tmp.json")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(story, f, indent=2, ensure_ascii=False)
    tmp_path.replace(story_path)


def build_system_prompt(story: Dict[str, Any]) -> str:
    """One centralized place to define your 'series bible' prompt."""
    title = story.get("title", "Untitled Story")
    return (
        f"You are an expert long-form fiction writer.\n\n"
        f"Story title: {title}\n\n"
        f"Use a consistent, immersive style, third-person limited POV when appropriate.\n"
        f"Stay grounded in the established world, characters, items and locations.\n"
        f"Do NOT break character or mention that you are an AI.\n"
        f"Write vivid scenes with natural pacing.\n"
        f"Keep continuity with the outline and prior chapters in the prompt.\n"
    )


def build_chapter_prompt(
    story: Dict[str, Any],
    chapter_outline: Dict[str, Any],
    previous_chapters_summary: str,
    target_words: int,  # NEW
) -> str:
    """Build the user prompt for one chapter."""
    beats = "\n".join(f"- {b}" for b in chapter_outline.get("beat_summary", []))
    end_hook = chapter_outline.get("end_hook", "")
    title = chapter_outline.get("title", "Untitled Chapter")

    return (
        f"You are writing chapter '{title}'.\n\n"
        f"Previous chapters (summary):\n{previous_chapters_summary}\n\n"
        f"Chapter outline:\n{beats}\n\n"
        f"Target length: around {target_words} words. "
        f"It is OK to be within about ±15% of this, but do not go wildly shorter or longer.\n\n"
        f"End this chapter so that it naturally sets up:\n{end_hook}\n\n"
        f"Now write the full chapter as continuous prose."
    )


def summarise_previous_chapters(story: Dict[str, Any], runs: List[RunInfo]) -> str:
    """
    Minimal placeholder: you probably already have a real summariser.
    For now we just list which chapters exist in previous runs.
    """
    if not runs:
        return "No prior chapters exist yet; this is the first chapter."

    lines = []
    last_run = runs[-1]
    for ch in last_run.chapters:
        lines.append(f"Chapter {ch.chapter_id}: see file {ch.text_file}")
    return "\n".join(lines)


# ---------- Main pipeline ----------

def run_story(story_path: Path):
    story = load_story(story_path)

    # Increment version counter
    story.setdefault("version_counter", 0)
    story["version_counter"] += 1
    version = story["version_counter"]

    # Story-level default target words
    default_target_words = int(story.get("default_target_words", 2000))  # NEW
    print(f"Default target words per chapter from story.json: {default_target_words}")
    tw_input = input(f"Override default target words per chapter [{default_target_words}]: ").strip()
    if tw_input:
        default_target_words = int(tw_input)
    print(f"Using default_target_words = {default_target_words}")

    # Discover models from LM Studio
    models = list_lmstudio_models()
    if not models:
        print("No models found from LM Studio. Is the server running?")
        sys.exit(1)

    model_info = choose_model_interactive(models)
    model_id = model_info["id"]
    max_ctx = int(model_info.get("max_context_length", 4096))
    # Heuristic: leave 40% for prompt, 60% for generation
    suggested_ceiling = int(max_ctx * 0.6)

    print(f"Selected model: {model_id}")
    print(f"Model max context length: {max_ctx} tokens")
    print(f"Suggested max_tokens ceiling for generation: {suggested_ceiling}")

    # Allow user to tweak sampling + global max_tokens ceiling
    temp_str = input("Temperature [0.8]: ").strip()
    temperature = float(temp_str) if temp_str else 0.8

    mt_str = input(f"Global max_tokens ceiling [{suggested_ceiling}]: ").strip()
    max_tokens_ceiling = int(mt_str) if mt_str else suggested_ceiling

    # Build run id and directory
    ts = time.strftime("%Y%m%dT%H%M%S")
    run_id = f"run_{version:04d}_{ts}"
    story_root = story_path.parent
    run_dir = story_root / "runs" / run_id
    chapters_dir = run_dir / "chapters"
    audio_dir = run_dir / "audio"
    for d in (run_dir, chapters_dir, audio_dir):
        d.mkdir(parents=True, exist_ok=True)

    system_prompt = build_system_prompt(story)

    outline = story.get("outline", {})
    chapter_outlines = outline.get("chapters", [])

    # For now, previous_chapters_summary = simple stub
    previous_chapters_summary = summarise_previous_chapters(story, [])

    new_run_chapters: List[RunChapterInfo] = []

    for ch_outline in chapter_outlines:
        ch_id = ch_outline["id"]
        slug = ch_outline.get("slug", f"ch{ch_id:02d}")

        # Determine per-chapter target words
        chapter_target_words = int(
            ch_outline.get("target_words", default_target_words)
        )  # NEW
        # Convert to tokens, with a safety factor
        ideal_tokens = int(chapter_target_words * TOKENS_PER_WORD)  # NEW
        per_chapter_max_tokens = min(ideal_tokens, max_tokens_ceiling)  # NEW

        print(
            f"\n=== Generating Chapter {ch_id}: {slug} ===\n"
            f"Target words: {chapter_target_words}, "
            f"ideal_tokens ≈ {ideal_tokens}, "
            f"per_chapter_max_tokens = {per_chapter_max_tokens}"
        )

        user_prompt = build_chapter_prompt(
            story=story,
            chapter_outline=ch_outline,
            previous_chapters_summary=previous_chapters_summary,
            target_words=chapter_target_words,  # NEW
        )

        chapter_text, usage = generate_chapter(
            model_id=model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=per_chapter_max_tokens,  # NEW
        )

        chapter_filename = f"{slug}.md"
        chapter_path = chapters_dir / chapter_filename
        chapter_path.write_text(chapter_text, encoding="utf-8")

        audio_filename = f"{slug}.mp3"
        audio_path = audio_dir / audio_filename
        text_to_speech(chapter_text, audio_path)

        ch_info = RunChapterInfo(
            chapter_id=ch_id,
            chapter_slug=slug,
            text_file=str(Path("chapters") / chapter_filename),
            audio_file=str(Path("audio") / audio_filename),
            usage=usage,
        )
        new_run_chapters.append(ch_info)

        # Optionally update previous_chapters_summary using a real summariser
        previous_chapters_summary += (
            f"\nChapter {ch_id}: generated in {ch_info.text_file}"
        )

    run_info = RunInfo(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model_id=model_id,
        temperature=temperature,
        max_tokens_ceiling=max_tokens_ceiling,      # NEW
        default_target_words=default_target_words,  # NEW
        chapters=new_run_chapters,
    )

    # Add run info into story JSON
    story.setdefault("runs", [])
    story["runs"].append(asdict(run_info))

    # Save updated story in root and in run dir
    save_story(story_path, story)
    save_story(run_dir / "story.json", story)

    # Save run config separately
    config = {
        "run_id": run_id,
        "model_id": model_id,
        "temperature": temperature,
        "max_tokens_ceiling": max_tokens_ceiling,
        "default_target_words": default_target_words,
        "lmstudio_base": LMSTUDIO_BASE,
        "tokens_per_word": TOKENS_PER_WORD,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"\nRun complete: {run_id}")
    print(f"Chapters in: {chapters_dir}")
    print(f"Audio in: {audio_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: story_driver.py data/my_story/story.json")
        sys.exit(1)

    story_json_path = Path(sys.argv[1]).resolve()
    if not story_json_path.is_file():
        print(f"Story JSON not found: {story_json_path}")
        sys.exit(1)

    run_story(story_json_path)
