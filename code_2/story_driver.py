#!/usr/bin/env python3
import sys
from pathlib import Path

from story_core.runner import run_story


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: story_driver.py data/my_story/story.json")
        sys.exit(1)

    story_json_path = Path(sys.argv[1]).resolve()
    if not story_json_path.is_file():
        print(f"Story JSON not found: {story_json_path}")
        sys.exit(1)

    run_story(story_json_path)
