import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
import copy

from .config import TOKENS_PER_WORD, default_voices_dir
from .story_io import load_story, save_story
from .lmstudio_client import (
    list_lmstudio_models,
    choose_model_interactive,
    generate_chapter,
)
from .tts import choose_voice_model_interactive, text_to_speech
from .metadata import (
    build_system_prompt,
    build_chapter_prompt,
    analyse_chapter_metadata,
    merge_metadata_into_story,
    build_previous_chapters_summary,
)
from .model import RunChapterInfo, RunInfo, RunSettings, ImageSettings
from .image_core import (
    build_image_prompt_for_chapter_start,
    build_image_prompt_for_chapter_end,
    build_image_prompt_for_event,
    generate_image,
)

def run_story_with_settings(story_path: Path, settings: RunSettings) -> RunInfo:
    """
    Non-interactive: used by web UI and CLI wrapper.

    - story_path: path to base data/<story_id>/story.json (read-only)
    - settings: model/temp/tokens/words/voice/image

    The base story.json is NOT modified. A copy with updated metadata is
    written into data/<story_id>/runs/<run_id>/story.json.
    """
    base_story = load_story(story_path)

    story_root = story_path.parent
    runs_root = story_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    # Find next run index from existing run_* directories
    max_index = 0
    for d in runs_root.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith("run_"):
            continue
        parts = name.split("_", 2)
        if len(parts) < 2:
            continue
        idx_str = parts[1]
        if idx_str.isdigit():
            max_index = max(max_index, int(idx_str))

    next_index = max_index + 1

    # Working copy of the story for THIS run only
    story = copy.deepcopy(base_story)

    # Default target words for this run
    story["default_target_words"] = settings.default_target_words
    default_target_words = settings.default_target_words

    # timestamps + run dirs
    ts = time.strftime("%Y%m%dT%H%M%S")
    run_id = f"run_{next_index:04d}_{ts}"
    run_dir = runs_root / run_id
    chapters_dir = run_dir / "chapters"
    audio_dir = run_dir / "audio"
    images_dir = run_dir / "images"
    events_img_dir = images_dir / "events"
    for d in (run_dir, chapters_dir, audio_dir, images_dir, events_img_dir):
        d.mkdir(parents=True, exist_ok=True)

    # This is the ONLY story.json we will write
    run_story_path = run_dir / "story.json"

    system_prompt = build_system_prompt(story)

    outline = story.get("outline", {})
    chapter_outlines = outline.get("chapters", [])

    new_run_chapters: List[RunChapterInfo] = []

    voice_path: Optional[Path] = (
        Path(settings.voice_model) if settings.voice_model else None
    )

    # Image metadata structure to attach at end
    images_meta = {
        "chapters": {},  # chapter_id(str) -> {"start": path, "end": path}
        "events": {},    # event_id -> path
    }

    img_settings = settings.image_settings

    for ch_outline in chapter_outlines:
        ch_id = ch_outline["id"]
        slug = ch_outline.get("slug", f"ch{ch_id:02d}")
        title = ch_outline.get("title", f"Chapter {ch_id}")

        chapter_target_words = int(
            ch_outline.get("target_words", default_target_words)
        )
        ideal_tokens = int(chapter_target_words * TOKENS_PER_WORD)
        per_chapter_max_tokens = min(ideal_tokens, settings.max_tokens_ceiling)

        print(
            f"\n=== Generating Chapter {ch_id}: {slug} ===\n"
            f"Target words: {chapter_target_words}, "
            f"ideal_tokens ≈ {ideal_tokens}, "
            f"per_chapter_max_tokens = {per_chapter_max_tokens}"
        )

        # --- Chapter start image ---
        if img_settings and img_settings.enabled:
            start_prompt = build_image_prompt_for_chapter_start(story, ch_outline)
            start_name = f"ch{ch_id:02d}-start.png"
            start_path = images_dir / start_name
            generate_image(start_prompt, start_path, img_settings)

            images_meta["chapters"].setdefault(str(ch_id), {})["start"] = \
                f"images/{start_name}"

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
            model_id=settings.model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=settings.temperature,
            max_tokens=per_chapter_max_tokens,
        )

        chapter_filename = f"{slug}.md"
        chapter_path = chapters_dir / chapter_filename
        chapter_path.write_text(chapter_text, encoding="utf-8")

        audio_filename = f"{slug}.wav"
        audio_path = audio_dir / audio_filename
        text_to_speech(chapter_text, audio_path, voice_path)

        print("  [META] Analysing chapter metadata...")
        meta = analyse_chapter_metadata(
            model_id=settings.model_id,
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

        # --- Chapter end image ---
        if img_settings and img_settings.enabled:
            end_prompt = build_image_prompt_for_chapter_end(story, ch_outline, meta)
            end_name = f"ch{ch_id:02d}-end.png"
            end_path = images_dir / end_name
            generate_image(end_prompt, end_path, img_settings)

            images_meta["chapters"].setdefault(str(ch_id), {})["end"] = \
                f"images/{end_name}"

            # Per-event images
            for ev in meta.get("events", []):
                ev_id = ev.get("id")
                if not ev_id:
                    continue
                ev_prompt = build_image_prompt_for_event(story, ev)
                ev_name = f"{ev_id}.png"
                ev_path = events_img_dir / ev_name
                generate_image(ev_prompt, ev_path, img_settings)

                images_meta["events"][ev_id] = f"images/events/{ev_name}"

        ch_info = RunChapterInfo(
            chapter_id=ch_id,
            chapter_slug=slug,
            text_file=str(Path("chapters") / chapter_filename),
            audio_file=str(Path("audio") / audio_filename),
            usage=usage,
        )
        new_run_chapters.append(ch_info)

        # Save after each chapter so we don't lose progress (to run copy only)
        save_story(run_story_path, story)

    # Attach THIS run's info + image metadata
    run_info = RunInfo(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model_id=settings.model_id,
        temperature=settings.temperature,
        max_tokens_ceiling=settings.max_tokens_ceiling,
        default_target_words=default_target_words,
        voice_model=settings.voice_model,
        chapters=new_run_chapters,
    )

    story.setdefault("runs", [])
    story["runs"].append(asdict(run_info))
    story["images"] = images_meta

    # Final save of run story
    save_story(run_story_path, story)

    # Config for quick view
    config = {
        "run_id": run_id,
        "timestamp": run_info.timestamp,
        "model_id": settings.model_id,
        "temperature": settings.temperature,
        "max_tokens_ceiling": settings.max_tokens_ceiling,
        "default_target_words": default_target_words,
        "voice_model": settings.voice_model,
        "tokens_per_word": TOKENS_PER_WORD,
        "images_enabled": bool(img_settings and img_settings.enabled),
    }
    (run_dir / "config.json").write_text(
        __import__("json").dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"\nRun complete: {run_id}")
    print(f"Chapters in: {chapters_dir}")
    print(f"Audio in: {audio_dir}")
    print(f"Images in: {images_dir}")
    print(f"Run story JSON: {run_story_path}")

    return run_info

def run_story(story_path: Path) -> None:
    """
    Interactive CLI wrapper: asks questions, then calls run_story_with_settings().
    """
    story = load_story(story_path)
    default_target_words = int(story.get("default_target_words", 2000))

    print(f"Default target words per chapter from story.json: {default_target_words}")
    tw_input = input(
        f"Override default target words per chapter [{default_target_words}]: "
    ).strip()
    if tw_input:
        default_target_words = int(tw_input)
    print(f"Using default_target_words = {default_target_words}")

    models = list_lmstudio_models()
    if not models:
        print("No models found from LM Studio. Is the server running?")
        return

    model_info = choose_model_interactive(models)
    model_id = model_info["id"]
    max_ctx = int(model_info.get("max_context_length", 4096))
    suggested_ceiling = int(max_ctx * 0.6)

    print(f"Selected model: {model_id}")
    print(f"Model max context length: {max_ctx} tokens")
    print(f"Suggested max_tokens ceiling for generation: {suggested_ceiling}")

    temp_str = input("Temperature [0.8]: ").strip()
    temperature = float(temp_str) if temp_str else 0.8

    mt_str = input(f"Global max_tokens ceiling [{suggested_ceiling}]: ").strip()
    max_tokens_ceiling = int(mt_str) if mt_str else suggested_ceiling

    use_tts = input("Generate audio (Piper)? [Y/n]: ").strip().lower()
    if use_tts in ("", "y", "yes"):
        voices_dir = default_voices_dir()
        voice_model_path = choose_voice_model_interactive(voices_dir)
        voice_model_str = str(voice_model_path) if voice_model_path else None
    else:
        voice_model_str = None

    settings = RunSettings(
        model_id=model_id,
        temperature=temperature,
        max_tokens_ceiling=max_tokens_ceiling,
        default_target_words=default_target_words,
        voice_model=voice_model_str,
    )

    run_story_with_settings(story_path, settings)
