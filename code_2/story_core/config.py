from pathlib import Path

# LM Studio REST base
LMSTUDIO_BASE = "http://localhost:1234"
REST_MODELS = f"{LMSTUDIO_BASE}/api/v0/models"
REST_CHAT = f"{LMSTUDIO_BASE}/api/v0/chat/completions"

# Rough tokens-per-word factor
TOKENS_PER_WORD = 1.4


def default_voices_dir() -> Path:
    """
    Default location for Piper .onnx voices.
    Assumes a 'voices' folder next to story_driver.py / web_app.py.
    """
    return Path(__file__).resolve().parent.parent / "voices"
