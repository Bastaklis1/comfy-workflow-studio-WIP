"""
Config persistence for ComfyUI Workflow Studio.
Saves/loads wf_studio_config.json in the project root (parent of scripts folder).
"""

import json
from pathlib import Path

_script_dir = Path(__file__).parent      # …/scripts/
_root_dir   = _script_dir.parent         # …/Workflow-Studio/

CONFIG_PATH = _root_dir / 'wf_studio_config.json'


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_config(cfg: dict):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    except Exception:
        pass
