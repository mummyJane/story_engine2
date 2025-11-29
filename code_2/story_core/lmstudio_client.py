from typing import Any, Dict, List

import requests  # type: ignore

from .config import REST_MODELS, REST_CHAT


def list_lmstudio_models() -> List[Dict[str, Any]]:
    resp = requests.get(REST_MODELS, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def choose_model_interactive(models: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    print(f"Chapter - {payload}")
    resp = requests.post(REST_CHAT, json=payload, timeout=60000)
    resp.raise_for_status()
    return resp.json()


def generate_chapter(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> (str, dict):
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
