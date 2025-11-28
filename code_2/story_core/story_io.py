import json
from pathlib import Path
from typing import Any, Dict


def load_story(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_story(path: Path, story: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp.json")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(story, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
