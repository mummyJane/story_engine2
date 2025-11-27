#!/usr/bin/env python3
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests  # type: ignore
import subprocess

LMSTUDIO_BASE = "http://localhost:1234"  # LM Studio server: lms server start
REST_MODELS = f"{LMSTUDIO_BASE}/api/v0/models"
REST_CHAT = f"{LMSTUDIO_BASE}/api/v0/chat/completions"

# Rough tokens-per-word factor (depends on language/content, but this is fine)
TOKENS_PER_WORD = 1.4

# Folder containing Piper .onnx voice models (change this if needed)
VOICES_DIR = Path(__file__).parent / "voices"


# ---------- Data classes for run metadata ----------

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
    max_tokens_ceiling: int
    default_target_words: int
    voice_model: Optional[str]
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


def lmstudio_chat(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Low-level wrapper around LM Studio /api/v0/chat/completions."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = requests.post(REST_CHAT, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data


def generate_chapter(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> (str, Dict[str, Any]):
    """Generate chapter text via LM Studio."""
    data = lmstudio_chat(
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, usage


# ---------- TTS helpers ----------

def list_voice_models(voices_dir: Path) -> List[Path]:
    """Return all .onnx voice models under voices_dir."""
    if not voices_dir.is_dir():
        return []
    return sorted(voices_dir.glob("*.onnx"))


def choose_voice_model_interactive(voices_dir: Path) -> Optional[Path]:
    """Pick a Piper voice model (.onnx) from VOICES_DIR."""
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


def text_to_speech(text: str, out_path: Path, voice_model: Optional[Path]):
    """
    Convert text → MP3 using Piper + ffmpeg.

    - voice_model is a Path to a Piper .onnx file.
    - out_path is the desired .mp3 path.

    If no voice_model is provided, we write a small placeholder file instead.
    """
    if voice_model is None:
        out_path.write_bytes(b"MP3 PLACEHOLDER\n")
        return

    # First generate a temporary WAV with Piper
    wav_path = out_path.with_suffix(".wav")

    print(f"  [TTS] Using voice: {voice_model.name}")
    try:
        subprocess.run(
            [
                "piper",
                "-m",
                str(voice_model),
                "-f",
                str(wav_path),
            ],
            input=text.encode("utf-8"),
            check=True,
        )
    except FileNotFoundError:
        print("  [TTS] Piper not found on PATH. Writing placeholder instead.")
        out_path.write_bytes(b"MP3 PLACEHOLDER (piper not found)\n")
        return
    except subprocess.CalledProcessError as e:
        print(f"  [TTS] Piper failed ({e}). Writing placeholder instead.")
        out_path.write_bytes(b"MP3 PLACEHOLDER (piper error)\n")
        return

    # Then convert WAV → MP3 using ffmpeg
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(wav_path),
                str(out_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("  [TTS] ffmpeg not found on PATH. Leaving WAV only.")
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass
        wav_path.rename(out_path)
        return
    except subprocess.CalledProcessError as e:
        print(f"  [TTS] ffmpeg failed ({e}). Leaving WAV only.")
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass
        wav_path.rename(out_path)
        return

    # Cleanup WAV if everything succeeded
    try:
        wav_path.unlink()
    except FileNotFoundError:
        pass


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
    target_words: int,
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


# ---------- NEW: metadata analysis & merging ----------

def safe_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from the model's text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to strip Markdown code fences or extra text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def analyse_chapter_metadata(
    model_id: str,
    story: Dict[str, Any],
    chapter_text: str,
    chapter_outline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ask LM Studio to produce structured JSON metadata for the chapter:
    summary, people, locations, items, events, new_people, new_locations, new_items.
    """

    world = story.get("world", {})
    people = world.get("people", {})
    locations = world.get("locations", {})
    items = world.get("items", {})

    def format_map(m: Dict[str, Any]) -> str:
        lines = []
        for k, v in m.items():
            name = v.get("name", "")
            lines.append(f"- {k}: {name}")
        return "\n".join(lines) if lines else "(none)"

    people_desc = format_map(people)
    locations_desc = format_map(locations)
    items_desc = format_map(items)

    sys_prompt = (
        "You are a tool that analyses chapters of a long-form novel and emits ONLY JSON.\n"
        "You never add commentary, explanations, or markdown fences.\n"
    )

    user_prompt = f"""
We are maintaining a structured story bible.

Existing people (id: name):
{people_desc}

Existing locations (id: name):
{locations_desc}

Existing items (id: name):
{items_desc}

Now analyse the following chapter and output a single JSON object with this structure:

{{
  "summary": "3–6 sentences summarising the chapter.",
  "people": ["list", "of", "character_ids_or_new_names"],
  "locations": ["list", "of", "location_ids_or_new_names"],
  "items": ["list", "of", "item_ids_or_new_names"],
  "events": [
    {{
      "id": null,
      "summary": "Short one-line event description.",
      "time_hint": "Optional approximate time, or null"
    }}
  ],
  "new_people": [
    {{
      "id": "machine_id_like_nurse_jane",
      "name": "Human Name",
      "notes": "Optional notes"
    }}
  ],
  "new_locations": [
    {{
      "id": "machine_id_like_unit_1_babies",
      "name": "Human Name",
      "notes": "Optional notes"
    }}
  ],
  "new_items": [
    {{
      "id": "machine_id_like_pink_sleeper",
      "name": "Human Name",
      "notes": "Optional notes"
    }}
  ]
}}

Rules:
- If there are no entries for a field, use an empty list [].
- Prefer existing ids when referring to known people/locations/items.
- For new_* entries, always provide an 'id' and 'name'.

Chapter text:
\"\"\"{chapter_text}\"\"\"
"""

    data = lmstudio_chat(
        model_id=model_id,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=1024,
    )
    raw = data["choices"][0]["message"]["content"]
    meta = safe_json_from_text(raw)

    if meta is None:
        # Fallback if parsing fails
        print("  [META] Failed to parse JSON from metadata response; using fallback.")
        fallback = {
            "summary": chapter_text[:400] + "...",
            "people": [],
            "locations": [],
            "items": [],
            "events": [],
            "new_people": [],
            "new_locations": [],
            "new_items": [],
        }
        return fallback

    # Ensure all expected keys exist
    for key in [
        "summary",
        "people",
        "locations",
        "items",
        "events",
        "new_people",
        "new_locations",
        "new_items",
    ]:
        meta.setdefault(key, [] if key != "summary" else "")

    return meta


def merge_metadata_into_story(
    story: Dict[str, Any],
    ch_id: int,
    slug: str,
    title: str,
    meta: Dict[str, Any],
) -> None:
    """
    Update story["world"], story["timeline"], and story["chapters_state"]
    using the metadata from one chapter.
    """

    world = story.setdefault("world", {})
    people_map = world.setdefault("people", {})
    locations_map = world.setdefault("locations", {})
    items_map = world.setdefault("items", {})
    timeline = story.setdefault("timeline", [])
    chapters_state = story.setdefault("chapters_state", [])

    # Merge new_* into world
    for p in meta.get("new_people", []):
        pid = p.get("id")
        if pid and pid not in people_map:
            people_map[pid] = {
                "name": p.get("name", pid),
                "notes": p.get("notes", ""),
            }
    for loc in meta.get("new_locations", []):
        lid = loc.get("id")
        if lid and lid not in locations_map:
            locations_map[lid] = {
                "name": loc.get("name", lid),
                "notes": loc.get("notes", ""),
            }
    for it in meta.get("new_items", []):
        iid = it.get("id")
        if iid and iid not in items_map:
            items_map[iid] = {
                "name": it.get("name", iid),
                "notes": it.get("notes", ""),
            }

    # Append events to timeline, record ids for this chapter
    timeline_ids: List[str] = []
    for ev in meta.get("events", []):
        ev_id = ev.get("id")
        if not ev_id:
            ev_id = f"ev_{len(timeline) + 1:04d}"
        ev["id"] = ev_id
        ev["chapter_id"] = ch_id
        timeline.append(ev)
        timeline_ids.append(ev_id)

    # Upsert per-chapter state
    chapter_entry = None
    for cs in chapters_state:
        if cs.get("id") == ch_id:
            chapter_entry = cs
            break
    if chapter_entry is None:
        chapter_entry = {"id": ch_id, "slug": slug, "title": title}
        chapters_state.append(chapter_entry)

    chapter_entry["summary"] = meta.get("summary", "")
    chapter_entry["people"] = meta.get("people", [])
    chapter_entry["locations"] = meta.get("locations", [])
    chapter_entry["items"] = meta.get("items", [])
    chapter_entry["timeline_refs"] = timeline_ids


def build_previous_chapters_summary(story: Dict[str, Any], upto_chapter_id: int) -> str:
    """
    Build a compact summary of previous chapters using story['chapters_state'].
    """
    chapters_state = story.get("chapters_state", [])
    relevant = [cs for cs in chapters_state if cs.get("id", 0) < upto_chapter_id]
    relevant.sort(key=lambda cs: cs.get("id", 0))

    if not relevant:
        return "No prior chapters exist yet; this is the first chapter."

    lines = []
    for cs in relevant:
        cid = cs.get("id")
        title = cs.get("title", f"Chapter {cid}")
        summary = cs.get("summary", "")
        lines.append(f"Chapter {cid} – {title}:\n{summary}")
    return "\n\n".join(lines)


# ---------- Main pipeline ----------

def run_story(story_path: Path):
    story = load_story(story_path)

    # Increment version counter
    story.setdefault("version_counter", 0)
    story["version_counter"] += 1
    version = story["version_counter"]

    # Story-level default target words
    default_target_words = int(story.get("default_target_words", 2000))
    print(f"Default target words per chapter from story.json: {default_target_words}")
    tw_input = input(
        f"Override default target words per chapter [{default_target_words}]: "
    ).strip()
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

    # TTS voice selection
    use_tts = input("Generate audio (Piper + ffmpeg)? [Y/n]: ").strip().lower()
    if use_tts in ("", "y", "yes"):
        voice_model = choose_voice_model_interactive(VOICES_DIR)
    else:
        voice_model = None

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

    new_run_chapters: List[RunChapterInfo] = []

    for ch_outline in chapter_outlines:
        ch_id = ch_outline["id"]
        slug = ch_outline.get("slug", f"ch{ch_id:02d}")
        title = ch_outline.get("title", f"Chapter {ch_id}")

        # Determine per-chapter target words
        chapter_target_words = int(
            ch_outline.get("target_words", default_target_words)
        )
        # Convert to tokens, with a safety factor
        ideal_tokens = int(chapter_target_words * TOKENS_PER_WORD)
        per_chapter_max_tokens = min(ideal_tokens, max_tokens_ceiling)

        print(
            f"\n=== Generating Chapter {ch_id}: {slug} ===\n"
            f"Target words: {chapter_target_words}, "
            f"ideal_tokens ≈ {ideal_tokens}, "
            f"per_chapter_max_tokens = {per_chapter_max_tokens}"
        )

        previous_chapters_summary = build_previous_chapters_summary(
            story, upto_chapter_id=ch_id
        )

        user_prompt = build_chapter_prompt(
            story=story,
            chapter_outline=ch_outline,
            previous_chapters_summary=previous_chapters_summary,
            target_words=chapter_target_words,
        )

        chapter_text, usage = generate_chapter(
            model_id=model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=per_chapter_max_tokens,
        )

        # Save chapter text
        chapter_filename = f"{slug}.md"
        chapter_path = chapters_dir / chapter_filename
        chapter_path.write_text(chapter_text, encoding="utf-8")

        # TTS
        audio_filename = f"{slug}.mp3"
        audio_path = audio_dir / audio_filename
        text_to_speech(chapter_text, audio_path, voice_model)

        # NEW: analyse metadata and merge into story.json
        print("  [META] Analysing chapter metadata...")
        meta = analyse_chapter_metadata(
            model_id=model_id,
            story=story,
            chapter_text=chapter_text,
            chapter_outline=ch_outline,
        )
        merge_metadata_into_story(
            story=story,
            ch_id=ch_id,
            slug=slug,
            title=title,
            meta=meta,
        )

        ch_info = RunChapterInfo(
            chapter_id=ch_id,
            chapter_slug=slug,
            text_file=str(Path("chapters") / chapter_filename),
            audio_file=str(Path("audio") / audio_filename),
            usage=usage,
        )
        new_run_chapters.append(ch_info)

        # Save intermediate story.json after each chapter (so crashes don't lose everything)
        save_story(story_path, story)

    run_info = RunInfo(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model_id=model_id,
        temperature=temperature,
        max_tokens_ceiling=max_tokens_ceiling,
        default_target_words=default_target_words,
        voice_model=str(voice_model) if voice_model else None,
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
        "voice_model": str(voice_model) if voice_model else None,
        "lmstudio_base": LMSTUDIO_BASE,
        "tokens_per_word": TOKENS_PER_WORD,
        "voices_dir": str(VOICES_DIR),
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
