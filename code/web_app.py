#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import json

# Adjust as needed
BASE_DIR = Path(__file__).parent
DATA_ROOT = BASE_DIR / "data"

app = FastAPI(title="Story Engine Web UI")

# Mount data for serving MP3s etc
app.mount("/data", StaticFiles(directory=DATA_ROOT), name="data")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---------- Helpers ----------

def story_dir(story_id: str) -> Path:
    return DATA_ROOT / story_id


def story_json_path(story_id: str) -> Path:
    return story_dir(story_id) / "story.json"


def load_story(story_id: str) -> Dict[str, Any]:
    path = story_json_path(story_id)
    if not path.is_file():
        raise FileNotFoundError(f"story.json not found for {story_id}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_story(story_id: str, story: Dict[str, Any]) -> None:
    path = story_json_path(story_id)
    tmp = path.with_suffix(".tmp.json")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(story, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def list_stories() -> List[str]:
    if not DATA_ROOT.is_dir():
        return []
    stories = []
    for d in DATA_ROOT.iterdir():
        if d.is_dir() and (d / "story.json").is_file():
            stories.append(d.name)
    return sorted(stories)


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    stories = list_stories()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "stories": stories},
    )

@app.get("/story/{story_id}", response_class=HTMLResponse)
async def story_overview(request: Request, story_id: str):
    story = load_story(story_id)

    world = story.setdefault("world", {})
    world.setdefault("people", {})
    world.setdefault("locations", {})
    world.setdefault("items", {})

    outline = story.setdefault("outline", {})
    outline.setdefault("chapters", [])

    chapters_state = story.setdefault("chapters_state", [])
    runs = story.setdefault("runs", [])
    story.setdefault("timeline", [])  # <--- add this line

    return templates.TemplateResponse(
        "story_overview.html",
        {
            "request": request,
            "story_id": story_id,
            "story": story,
            "world": world,
            "outline": outline,
            "chapters_state": chapters_state,
            "runs": runs,
        },
    )

# ----- World: People -----

@app.get("/story/{story_id}/people", response_class=HTMLResponse)
async def world_people(request: Request, story_id: str):
    story = load_story(story_id)
    world = story.setdefault("world", {})
    people = world.setdefault("people", {})

    # Normalise entries so template can safely read age/tags/notes
    for pid, pdata in people.items():
        pdata.setdefault("age", None)
        pdata.setdefault("tags", [])
        pdata.setdefault("notes", "")

    return templates.TemplateResponse(
        "world_people.html",
        {
            "request": request,
            "story_id": story_id,
            "people": people,
        },
    )

@app.post("/story/{story_id}/people/add")
async def world_people_add(
    story_id: str,
    person_id: str = Form(...),
    name: str = Form(...),
    age: Optional[str] = Form(None),
    tags: str = Form(""),
    notes: str = Form(""),
):
    story = load_story(story_id)
    world = story.setdefault("world", {})
    people = world.setdefault("people", {})

    if person_id:
        pdata: Dict[str, Any] = {
            "name": name,
            "notes": notes,
            "tags": parse_tags(tags),
        }
        if age:
            try:
                pdata["age"] = int(age)
            except ValueError:
                pdata["age"] = None
        else:
            pdata["age"] = None

        people[person_id] = pdata

    save_story(story_id, story)
    return RedirectResponse(
        url=f"/story/{story_id}/people", status_code=303
    )


@app.post("/story/{story_id}/people/update")
async def world_people_update(
    story_id: str,
    person_id: str = Form(...),
    name: str = Form(...),
    age: Optional[str] = Form(None),
    tags: str = Form(""),
    notes: str = Form(""),
):
    story = load_story(story_id)
    world = story.setdefault("world", {})
    people = world.setdefault("people", {})

    if person_id in people:
        pdata = people[person_id]
        pdata["name"] = name
        pdata["notes"] = notes
        pdata["tags"] = parse_tags(tags)

        if age:
            try:
                pdata["age"] = int(age)
            except ValueError:
                pdata["age"] = None
        else:
            pdata["age"] = None

    save_story(story_id, story)
    return RedirectResponse(
        url=f"/story/{story_id}/people", status_code=303
    )


# ----- World: Locations -----

@app.get("/story/{story_id}/locations", response_class=HTMLResponse)
async def world_locations(request: Request, story_id: str):
    story = load_story(story_id)
    world = story.setdefault("world", {})
    locations = world.setdefault("locations", {})

    # Normalise location entries so template can rely on 'type' and 'notes'
    for lid, loc in locations.items():
        loc.setdefault("type", "")
        loc.setdefault("notes", "")

    return templates.TemplateResponse(
        "world_locations.html",
        {
            "request": request,
            "story_id": story_id,
            "locations": locations,
        },
    )


@app.post("/story/{story_id}/locations/add")
async def world_locations_add(
    story_id: str,
    loc_id: str = Form(...),
    name: str = Form(...),
    loc_type: str = Form(""),
    notes: str = Form(""),
):
    story = load_story(story_id)
    world = story.setdefault("world", {})
    locations = world.setdefault("locations", {})

    if loc_id:
        locations[loc_id] = {
            "name": name,
            "type": loc_type,
            "notes": notes,
        }

    save_story(story_id, story)
    return RedirectResponse(
        url=f"/story/{story_id}/locations", status_code=303
    )


@app.post("/story/{story_id}/locations/update")
async def world_locations_update(
    story_id: str,
    loc_id: str = Form(...),
    name: str = Form(...),
    loc_type: str = Form(""),
    notes: str = Form(""),
):
    story = load_story(story_id)
    world = story.setdefault("world", {})
    locations = world.setdefault("locations", {})

    if loc_id in locations:
        loc = locations[loc_id]
        loc["name"] = name
        loc["type"] = loc_type
        loc["notes"] = notes

    save_story(story_id, story)
    return RedirectResponse(
        url=f"/story/{story_id}/locations", status_code=303
    )


# ----- World: Items -----

@app.get("/story/{story_id}/items", response_class=HTMLResponse)
async def world_items(request: Request, story_id: str):
    story = load_story(story_id)
    world = story.setdefault("world", {})
    items = world.setdefault("items", {})
    return templates.TemplateResponse(
        "world_items.html",
        {
            "request": request,
            "story_id": story_id,
            "items": items,
        },
    )


@app.post("/story/{story_id}/items/add")
async def world_items_add(
    story_id: str,
    item_id: str = Form(...),
    name: str = Form(...),
    notes: str = Form(""),
):
    story = load_story(story_id)
    world = story.setdefault("world", {})
    items = world.setdefault("items", {})
    if item_id:
        items[item_id] = {"name": name, "notes": notes}
    save_story(story_id, story)
    return RedirectResponse(
        url=f"/story/{story_id}/items", status_code=303
    )


@app.post("/story/{story_id}/items/update")
async def world_items_update(
    story_id: str,
    item_id: str = Form(...),
    name: str = Form(...),
    notes: str = Form(""),
):
    story = load_story(story_id)
    world = story.setdefault("world", {})
    items = world.setdefault("items", {})
    if item_id in items:
        items[item_id]["name"] = name
        items[item_id]["notes"] = notes
    save_story(story_id, story)
    return RedirectResponse(
        url=f"/story/{story_id}/items", status_code=303
    )


# ----- Chapters (outline) -----

@app.get("/story/{story_id}/chapters", response_class=HTMLResponse)
async def chapters_list(request: Request, story_id: str):
    story = load_story(story_id)
    outline = story.get("outline", {})
    chapters = outline.get("chapters", [])
    return templates.TemplateResponse(
        "chapters_list.html",
        {
            "request": request,
            "story_id": story_id,
            "chapters": chapters,
        },
    )


# ADD CHAPTER (no chapter_id here!)
@app.post("/story/{story_id}/chapters/add")
async def chapters_add(
    story_id: str,
    title: str = Form(...),
    slug: str = Form(""),
    target_words: str = Form("2000"),  # string, we parse ourselves
    end_hook: str = Form(""),
    beats_text: str = Form(""),
):
    story = load_story(story_id)
    outline = story.setdefault("outline", {})
    chapters = outline.setdefault("chapters", [])

    # Next ID = max existing id + 1
    next_id = 1 + max((int(c.get("id", 0)) for c in chapters), default=0)

    if not slug:
        slug = f"ch{next_id:02d}"

    # Safe parse of target_words
    default_tw = int(story.get("default_target_words", 2000))
    try:
        tw_val = int(target_words) if target_words.strip() else default_tw
    except ValueError:
        tw_val = default_tw

    beats = [b.strip() for b in beats_text.splitlines() if b.strip()]

    new_chapter = {
        "id": next_id,
        "slug": slug,
        "title": title,
        "target_words": tw_val,
        "end_hook": end_hook,
        "beat_summary": beats,
    }
    chapters.append(new_chapter)

    save_story(story_id, story)

    # Go straight to edit page for that chapter
    return RedirectResponse(
        url=f"/story/{story_id}/chapters/{next_id}",
        status_code=303,
    )


# EDIT CHAPTER – note the :int in the path
@app.get("/story/{story_id}/chapters/{chapter_id:int}", response_class=HTMLResponse)
async def chapter_edit(
    request: Request,
    story_id: str,
    chapter_id: int,
):
    story = load_story(story_id)
    outline = story.setdefault("outline", {})
    chapters = outline.setdefault("chapters", [])
    chapter = next((c for c in chapters if int(c.get("id", 0)) == chapter_id), None)
    if chapter is None:
        return HTMLResponse(
            f"Chapter {chapter_id} not found in {story_id}.", status_code=404
        )

    beats = "\n".join(chapter.get("beat_summary", []))
    return templates.TemplateResponse(
        "chapter_edit.html",
        {
            "request": request,
            "story_id": story_id,
            "chapter": chapter,
            "beats_text": beats,
        },
    )


# UPDATE CHAPTER – same :int in the path
@app.post("/story/{story_id}/chapters/{chapter_id:int}")
async def chapter_update(
    story_id: str,
    chapter_id: int,
    title: str = Form(...),
    slug: str = Form(...),
    target_words: str = Form("2000"),
    end_hook: str = Form(""),
    beats_text: str = Form(""),
):
    story = load_story(story_id)
    outline = story.setdefault("outline", {})
    chapters = outline.setdefault("chapters", [])

    chapter = next((c for c in chapters if int(c.get("id", 0)) == chapter_id), None)
    if chapter is None:
        return HTMLResponse(
            f"Chapter {chapter_id} not found in {story_id}.", status_code=404
        )

    default_tw = int(story.get("default_target_words", 2000))
    try:
        tw_val = int(target_words) if target_words.strip() else default_tw
    except ValueError:
        tw_val = default_tw

    chapter["title"] = title
    chapter["slug"] = slug
    chapter["target_words"] = tw_val
    chapter["end_hook"] = end_hook
    chapter["beat_summary"] = [
        b.strip() for b in beats_text.splitlines() if b.strip()
    ]

    save_story(story_id, story)
    return RedirectResponse(
        url=f"/story/{story_id}/chapters/{chapter_id}",
        status_code=303
    )

# ----- Runs -----

@app.get("/story/{story_id}/runs", response_class=HTMLResponse)
async def runs_list(request: Request, story_id: str):
    story = load_story(story_id)
    runs = story.get("runs", [])
    return templates.TemplateResponse(
        "runs_list.html",
        {
            "request": request,
            "story_id": story_id,
            "runs": runs,
        },
    )


@app.get("/story/{story_id}/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(
    request: Request,
    story_id: str,
    run_id: str,
):
    story = load_story(story_id)
    runs = story.get("runs", [])
    run = next((r for r in runs if r.get("run_id") == run_id), None)
    if run is None:
        return HTMLResponse(
            f"Run {run_id} not found in {story_id}.", status_code=404
        )

    return templates.TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "story_id": story_id,
            "run": run,
        },
    )


@app.get(
    "/story/{story_id}/runs/{run_id}/chapters/{chapter_id}",
    response_class=HTMLResponse,
)
async def run_chapter_detail(
    request: Request,
    story_id: str,
    run_id: str,
    chapter_id: int,
):
    story = load_story(story_id)
    runs = story.get("runs", [])
    run = next((r for r in runs if r.get("run_id") == run_id), None)
    if run is None:
        return HTMLResponse(
            f"Run {run_id} not found.", status_code=404
        )

    run_chapters = run.get("chapters", [])
    ch_info = next(
        (c for c in run_chapters if c.get("chapter_id") == chapter_id), None
    )
    if ch_info is None:
        return HTMLResponse(
            f"Chapter {chapter_id} not in run {run_id}.", status_code=404
        )

    # Load chapter text from file
    rel_text_path = Path(ch_info["text_file"])
    full_text_path = story_dir(story_id) / "runs" / run_id / rel_text_path
    chapter_text = ""
    if full_text_path.is_file():
        chapter_text = full_text_path.read_text(encoding="utf-8")

    # Summary / metadata from chapters_state
    chapters_state = story.get("chapters_state", [])
    chap_state = next(
        (cs for cs in chapters_state if cs.get("id") == chapter_id), None
    )

    # Timeline events for this chapter
    timeline = story.get("timeline", [])
    chapter_events = [
        ev for ev in timeline if ev.get("chapter_id") == chapter_id
    ]

    # MP3 URL
    rel_audio_path = Path(ch_info["audio_file"])
    mp3_url = f"/data/{story_id}/runs/{run_id}/{rel_audio_path.as_posix()}"

    return templates.TemplateResponse(
        "run_chapter_detail.html",
        {
            "request": request,
            "story_id": story_id,
            "run": run,
            "ch_info": ch_info,
            "chapter_text": chapter_text,
            "chap_state": chap_state,
            "events": chapter_events,
            "mp3_url": mp3_url,
        },
    )

@app.post("/stories/create")
async def create_story(
    story_id: str = Form(...),
    title: str = Form(...),
    default_target_words: int = Form(2000),
):
    """
    Create a new story folder under /data/<story_id> with a minimal story.json,
    then redirect to the story overview page.
    """
    story_id = story_id.strip()
    if not story_id:
        # could return a nicer page, but this will do for now
        return HTMLResponse("story_id is required", status_code=400)

    # Basic safety: no path separators
    if "/" in story_id or "\\" in story_id:
        return HTMLResponse("story_id cannot contain / or \\", status_code=400)

    sdir = story_dir(story_id)
    if sdir.exists():
        return HTMLResponse(f"Story '{story_id}' already exists.", status_code=400)

    sdir.mkdir(parents=True, exist_ok=False)

    story = {
        "story_id": story_id,
        "title": title or story_id,
        "version_counter": 0,
        "default_target_words": int(default_target_words),
        "world": {
            "people": {},
            "locations": {},
            "items": {}
        },
        "timeline": [],
        "outline": {
            "chapters": []
        },
        "chapters_state": [],
        "runs": []
    }

    save_story(story_id, story)

    return RedirectResponse(
        url=f"/story/{story_id}",
        status_code=303,
    )

def parse_tags(s: str) -> List[str]:
    """Split a comma-separated tags string into a clean list."""
    return [t.strip() for t in s.split(",") if t.strip()]

@app.get("/story/{story_id}/timeline", response_class=HTMLResponse)
async def story_timeline(request: Request, story_id: str):
    story = load_story(story_id)
    timeline = story.setdefault("timeline", [])

    # Build a map of chapter_id -> title for display
    chapter_titles: Dict[int, str] = {}

    chapters_state = story.get("chapters_state", [])
    for cs in chapters_state:
        cid = cs.get("id")
        if isinstance(cid, int):
            chapter_titles[cid] = cs.get("title", f"Chapter {cid}")

    outline = story.get("outline", {})
    for ch in outline.get("chapters", []):
        cid = ch.get("id")
        if isinstance(cid, int) and cid not in chapter_titles:
            chapter_titles[cid] = ch.get("title", f"Chapter {cid}")

    # Sort timeline by id string for now
    timeline_sorted = sorted(
        timeline,
        key=lambda ev: str(ev.get("id", ""))
    )

    return templates.TemplateResponse(
        "timeline.html",
        {
            "request": request,
            "story_id": story_id,
            "timeline": timeline_sorted,
            "chapter_titles": chapter_titles,
        },
    )


@app.post("/story/{story_id}/timeline/add")
async def timeline_add(
    story_id: str,
    summary: str = Form(...),
    time_hint: str = Form(""),
    chapter_id: str = Form(""),
):
    story = load_story(story_id)
    timeline = story.setdefault("timeline", [])

    ev_id = f"ev_{len(timeline) + 1:04d}"

    event: Dict[str, Any] = {
        "id": ev_id,
        "summary": summary,
        "time_hint": time_hint or None,
    }
    if chapter_id.strip():
        try:
            event["chapter_id"] = int(chapter_id)
        except ValueError:
            pass  # ignore bad chapter_id input

    timeline.append(event)
    save_story(story_id, story)

    return RedirectResponse(
        url=f"/story/{story_id}/timeline",
        status_code=303,
    )


@app.post("/story/{story_id}/timeline/update")
async def timeline_update(
    story_id: str,
    ev_id: str = Form(...),
    summary: str = Form(...),
    time_hint: str = Form(""),
    chapter_id: str = Form(""),
):
    story = load_story(story_id)
    timeline = story.setdefault("timeline", [])

    for ev in timeline:
        if ev.get("id") == ev_id:
            ev["summary"] = summary
            ev["time_hint"] = time_hint or None
            if chapter_id.strip():
                try:
                    ev["chapter_id"] = int(chapter_id)
                except ValueError:
                    pass
            break

    save_story(story_id, story)
    return RedirectResponse(
        url=f"/story/{story_id}/timeline",
        status_code=303,
    )
