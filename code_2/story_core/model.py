from dataclasses import dataclass
from typing import Any, Dict, List, Optional
#from .image_core import ImageSettings

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
class ImageSettings:
    enabled: bool = False
    backend_url: str = "http://127.0.0.1:7860"  # Automatic1111 default
    width: int = 768
    height: int = 512
    steps: int = 25
    cfg_scale: float = 7.0
    sampler_name: str = "DPM++ 2M Karras"
    negative_prompt: str = (
        "text, caption, watermark, logo, ui, low quality, blurry, distorted, extra limbs"
    )
    style_hint: Optional[str] = (
        "semi-realistic, consistent characters, no text, clean composition"
    )

@dataclass
class RunSettings:
    model_id: str
    temperature: float
    max_tokens_ceiling: int
    default_target_words: int
    voice_model: Optional[str] = None
    image_settings: Optional[ImageSettings] = None