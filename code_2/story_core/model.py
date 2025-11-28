from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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


@dataclass
class RunSettings:
    model_id: str
    temperature: float
    max_tokens_ceiling: int
    default_target_words: int
    # full path to voice .onnx, or None
    voice_model: Optional[str] = None
