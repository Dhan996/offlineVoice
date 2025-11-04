import json
import os

# Minimal CONFIG shim so modules importing from `config` work in this repo.
# Attempts to load config.json next to this file, then falls back to sane defaults.

_DEFAULT = {
    "system_prompt": "You are a concise helpful voice assistant. Keep responses brief and conversational.",
    "ollama_model": "llama3:latest",
    "ollama_base": "http://localhost:11434",
    "llm_temperature": 0.6,
    "llm_max_tokens": 256,
}

def _load_config_json() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                return {**_DEFAULT, **data}
    except Exception:
        pass
    return dict(_DEFAULT)


CONFIG = _load_config_json()

