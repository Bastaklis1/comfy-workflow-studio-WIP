"""
LLM backend integrations for ComfyUI Workflow Studio.
Supports Ollama (local) and Claude API enrichment, with OpenAI-compatible
endpoint support intended for a future update.
"""

import json
import re
from typing import Optional

# ── Optional backend availability ────────────────────────────────────────────

try:
    import urllib.request
    _URLLIB_OK = True
except Exception:
    _URLLIB_OK = False

try:
    import anthropic as _anthropic_lib
    _ANTHROPIC_OK = True
except Exception:
    _ANTHROPIC_OK = False


# ── Enrichment prompt ─────────────────────────────────────────────────────────

ENRICH_PROMPT = """\
You are analyzing a ComfyUI workflow. Return ONLY valid JSON — no markdown, no explanation.

Workflow: {name}
Path: {path}
Node types present: {nodes}
Model files: {models}
Notes/text found: {notes}

Return this exact JSON structure:
{{
  "summary": "<one sentence: what this workflow does>",
  "primary_purpose": "<single phrase: e.g. 'SDXL portrait generation', 'Wan2.2 text-to-video', 'face swap with inpainting'>",
  "notable": "<what makes this workflow unusual or interesting, or 'nothing notable' if standard>",
  "quality_signal": "<'template/placeholder', 'basic', 'intermediate', 'advanced', or 'specialized'>",
  "suggested_tags": ["<tag1>", "<tag2>"]
}}"""


def _build_prompt(workflow_data: dict) -> str:
    notes = ' | '.join(
        nd['params'].get('text', '')[:100]
        for nd in workflow_data.get('nodes', [])
        if nd.get('is_text_node') and nd['params'].get('text', '').strip()
    )[:300]

    return ENRICH_PROMPT.format(
        name   = workflow_data.get('path', '?'),
        path   = workflow_data.get('path', '?'),
        nodes  = ', '.join(sorted({
            nd['type'] for nd in workflow_data.get('nodes', [])
            if not nd['type'].startswith('[SUBGRAPH:')
        }))[:800],
        models = ', '.join(workflow_data.get('models', []))[:300],
        notes  = notes or 'none',
    )


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences that some models add despite instructions."""
    return re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()


# ── Ollama ────────────────────────────────────────────────────────────────────

def ollama_enrich(workflow_data: dict, model: str, endpoint: str) -> Optional[dict]:
    """Send one workflow to Ollama for enrichment. Returns parsed dict or None."""
    if not _URLLIB_OK:
        return None

    payload = json.dumps({
        'model':   model,
        'prompt':  _build_prompt(workflow_data),
        'stream':  False,
        'options': {'temperature': 0.1, 'num_predict': 400},
    }).encode()

    try:
        req = urllib.request.Request(
            f'{endpoint.rstrip("/")}/api/generate',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return json.loads(_strip_fences(result.get('response', '').strip()))
    except Exception:
        return None


def list_ollama_models(endpoint: str) -> list[str]:
    """Fetch available models from Ollama. Returns empty list on failure."""
    if not _URLLIB_OK:
        return []
    try:
        req = urllib.request.Request(
            f'{endpoint.rstrip("/")}/api/tags', method='GET')
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m['name'] for m in data.get('models', [])]
    except Exception:
        return []


# ── OpenAI-compatible endpoint (future: LM Studio, llama.cpp server, etc.) ───

def openai_compat_enrich(workflow_data: dict, model: str, endpoint: str,
                          api_key: str = '') -> Optional[dict]:
    """
    Send one workflow to any OpenAI-compatible /v1/chat/completions endpoint.
    Works with LM Studio, llama.cpp server, Ollama's OpenAI-compat layer, etc.
    Pass api_key='' for backends that don't require one.
    """
    if not _URLLIB_OK:
        return None

    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    payload = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': _build_prompt(workflow_data)}],
        'temperature': 0.1,
        'max_tokens': 400,
    }).encode()

    try:
        req = urllib.request.Request(
            f'{endpoint.rstrip("/")}/v1/chat/completions',
            data=payload,
            headers=headers,
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            raw = result['choices'][0]['message']['content'].strip()
            return json.loads(_strip_fences(raw))
    except Exception:
        return None


# ── Claude API ────────────────────────────────────────────────────────────────

def claude_enrich(workflow_data: dict, api_key: str) -> Optional[dict]:
    """Send one workflow to Claude API for enrichment."""
    if not _ANTHROPIC_OK:
        return None

    try:
        client = _anthropic_lib.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=500,
            messages=[{'role': 'user', 'content': _build_prompt(workflow_data)}],
        )
        raw = msg.content[0].text.strip()
        return json.loads(_strip_fences(raw))
    except Exception:
        return None
