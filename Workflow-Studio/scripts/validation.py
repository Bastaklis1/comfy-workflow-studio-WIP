"""
Workflow validation for ComfyUI Workflow Studio.
Detects genuine ComfyUI workflow JSON files by signature.
"""

import json
from pathlib import Path


def is_comfyui_workflow(filepath: Path) -> bool:
    """
    Returns True only if the file is a genuine ComfyUI workflow.
    Signature: JSON object with both 'nodes' (list) and 'links' (list).
    Filters out wildcards, model metadata, config files, etc.
    """
    try:
        with open(filepath, encoding='utf-8', errors='replace') as f:
            # Quick size sanity — real workflows are at least ~500 bytes
            raw = f.read(10)
            if not raw.strip().startswith('{'):
                return False
        with open(filepath, encoding='utf-8', errors='replace') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return False
        return isinstance(data.get('nodes'), list) and isinstance(data.get('links'), list)
    except Exception:
        return False
