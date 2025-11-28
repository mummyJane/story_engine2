from typing import Any, Dict, List

from .lmstudio_client import lmstudio_chat


def build_system_prompt(story: Dict[str, Any]) -> str:
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


def safe_json_from_text(text: str) -> Dict[str, Any] | None:
    import json

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

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
    world = story.get("world", {})
    people = world.get("people", {})
    locations = world.get("locations", {})
    items = world.get("items", {})

    def format_map(m: Dict[str, Any]) -> str:
        lines: List[str] = []
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
    world = story.setdefault("world", {})
    people_map = world.setdefault("people", {})
    locations_map = world.setdefault("locations", {})
    items_map = world.setdefault("items", {})
    timeline = story.setdefault("timeline", [])
    chapters_state = story.setdefault("chapters_state", [])

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

    timeline_ids: List[str] = []
    for ev in meta.get("events", []):
        ev_id = ev.get("id")
        if not ev_id:
            ev_id = f"ev_{len(timeline) + 1:04d}"
        ev["id"] = ev_id
        ev["chapter_id"] = ch_id
        timeline.append(ev)
        timeline_ids.append(ev_id)

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
    chapters_state = story.get("chapters_state", [])
    relevant = [cs for cs in chapters_state if cs.get("id", 0) < upto_chapter_id]
    relevant.sort(key=lambda cs: cs.get("id", 0))

    if not relevant:
        return "No prior chapters exist yet; this is the first chapter."

    lines: List[str] = []
    for cs in relevant:
        cid = cs.get("id")
        title = cs.get("title", f"Chapter {cid}")
        summary = cs.get("summary", "")
        lines.append(f"Chapter {cid} – {title}:\n{summary}")
    return "\n\n".join(lines)
