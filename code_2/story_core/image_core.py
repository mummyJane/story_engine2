from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import base64
import requests  # type: ignore

from .model import ImageSettings


def _apply_style_hint(prompt: str, settings: ImageSettings) -> str:
    if settings.style_hint:
        return f"{prompt}\n\nStyle: {settings.style_hint}"
    return prompt


def build_image_prompt_for_chapter_start(story: Dict[str, Any], ch_outline: Dict[str, Any]) -> str:
    title = ch_outline.get("title", "Untitled Chapter")
    beats = ch_outline.get("beat_summary", [])
    first = beats[0] if beats else ""
    return (
        f"Illustration for the beginning of chapter '{title}'. "
        f"Scene: {first}. "
        f"Single frame, no text, cinematic composition, focus on mood and setting."
    )


def build_image_prompt_for_chapter_end(
    story: Dict[str, Any],
    ch_outline: Dict[str, Any],
    meta: Dict[str, Any],
) -> str:
    title = ch_outline.get("title", "Untitled Chapter")
    summary = meta.get("summary", "")
    return (
        f"Illustration capturing the end of chapter '{title}'. "
        f"Key moment: {summary[:400]} "
        f"Single frame, no text, focus on emotional tone."
    )


def build_image_prompt_for_event(story: Dict[str, Any], ev: Dict[str, Any]) -> str:
    return (
        f"Illustration of the event: {ev.get('summary', '')}. "
        f"Single frame, no text, clear depiction of the physical situation."
    )


def generate_image(prompt: str, out_path: Path, settings: ImageSettings) -> None:
    """
    Generate an image using Automatic1111's /sdapi/v1/txt2img API.

    - prompt: text prompt
    - out_path: where to save a PNG
    - settings: ImageSettings (backend_url, size, steps, etc.)
    """
    if not settings.enabled:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    full_prompt = _apply_style_hint(prompt, settings)
    endpoint = settings.backend_url.rstrip("/") + "/sdapi/v1/txt2img"

    payload: Dict[str, Any] = {
        "prompt": full_prompt,
        "negative_prompt": settings.negative_prompt,
        "width": settings.width,
        "height": settings.height,
        "steps": settings.steps,
        "cfg_scale": settings.cfg_scale,
        "sampler_name": settings.sampler_name,
        "batch_size": 1,
        "n_iter": 1,
        "seed": -1,
    }

    try:
        resp = requests.post(endpoint, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        images = data.get("images", [])
        if not images:
            print(f"[IMG] No images returned for prompt: {prompt[:80]!r}")
            return

        img_b64 = images[0]
        # Some backends prefix with "data:image/png;base64,..."
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(img_b64)
        out_path.write_bytes(img_bytes)
        print(f"[IMG] Wrote image: {out_path}")
    except Exception as e:
        print(f"[IMG] Failed to generate image: {e}")
        # Optional: write prompt placeholder for debugging
        out_path.with_suffix(".txt").write_text(full_prompt, encoding="utf-8")
