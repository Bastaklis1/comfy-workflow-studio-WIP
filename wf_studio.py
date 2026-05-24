#!/usr/bin/env python3
"""
ComfyUI Workflow Studio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unified GUI for workflow analysis, tagging, deduplication, and LLM enrichment.

Requires: wf_extract.py in the same folder (or on sys.path)
Optional: Ollama running locally for LLM enrichment
Optional: anthropic package for Claude API enrichment

Run:  python wf_studio.py
"""

import json
import os
import re
import sys
import threading
import hashlib
import shutil
import time
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

# ── Try to import the extractor ───────────────────────────────────────────────
_extractor_path = Path(__file__).parent / 'wf_extract.py'
_extractor_available = False
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location('wf_extractor', _extractor_path)
    _ext_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_ext_mod)
    SchemaRegistry = _ext_mod.SchemaRegistry
    extract        = _ext_mod.extract
    render         = _ext_mod.render
    STATIC_PARAMS  = _ext_mod.STATIC_PARAMS
    SKIP_TYPES     = _ext_mod.SKIP_TYPES
    REROUTE_TYPES  = _ext_mod.REROUTE_TYPES
    _extractor_available = True
except Exception as e:
    print(f'[warn] Extractor not found or failed to load: {e}')
    print(f'       Expected: {_extractor_path}')

# ── Optional LLM backends ─────────────────────────────────────────────────────
try:
    import urllib.request
    _URLLIB_OK = True
except:
    _URLLIB_OK = False

try:
    import anthropic as _anthropic_lib
    _ANTHROPIC_OK = True
except:
    _ANTHROPIC_OK = False


# ═════════════════════════════════════════════════════════════════════════════
# WORKFLOW VALIDATION — ComfyUI signature detection
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════

CONFIG_PATH = Path(__file__).parent / 'wf_studio_config.json'

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

# ═════════════════════════════════════════════════════════════════════════════
# ECOSYSTEM TAGGING — deterministic, no LLM needed
# ═════════════════════════════════════════════════════════════════════════════

# Anchor node types that strongly indicate an ecosystem.
# Ordered by specificity — more specific entries first.
ECOSYSTEM_ANCHORS = {
    # Video — model-specific
    'wan':         {'WanVideoSampler', 'WanVideoLoader', 'WanVideoEncode', 'WanVideoLora',
                    'WanVideoNAG', 'WanVideoBlockSwap', 'WanVideoEnhanceAVideo',
                    'CLIPTextEncodeWan', 'DaSiWa_Wan22'},
    'ltx':         {'LTXVLoader', 'LTXVSampler', 'LTXVScheduler', 'LTXVConditioning',
                    'LTXVImgToVideo', 'LTXVPreprocess', 'LTXVAddAudio',
                    'LTXVAudioVAEDecode', 'LTXVConcatAVLatent', 'CLIPTextEncodeLTXV'},
    'hunyuan_vid': {'HunyuanVideoSampler', 'HunyuanVideoLoader', 'HunyuanVideoDecode',
                    'CLIPTextEncodeHunyuan', 'HunyuanVideoVAELoader'},
    'mochi':       {'MochiWrapper', 'MochiSampler', 'MochiModelConfig'},
    'cogvideo':    {'CogVideoSampler', 'CogVideoLoader', 'CogVideoEncode'},
    'animatediff': {'AnimateDiffLoader', 'AnimateDiffSampler', 'ADE_AnimateDiffLoaderV1Gen2'},
    # Image — model-specific
    'nunchaku':    {'NunchakuFluxDiTLoader', 'NunchakuFluxLoraLoader',
                    'NunchakuTextEncoderLoaderV2', 'NunchakuFluxPuLIDApplyV2'},
    'zimage':      {'ZSamplerTurbo2', 'StyleStringInjector2', 'StylePromptEncoder2',
                    'ZImageAnalyzerSelectiveLoaderV2', 'ZEngineer', 'ZSamplerClass'},
    'flux':        {'FluxGuidance', 'ModelSamplingFlux', 'CLIPTextEncodeFlux',
                    'Flux2Scheduler', 'EmptyFlux2LatentImage', 'FluxKontextImageScale'},
    'sd3':         {'ModelSamplingSD3', 'CLIPTextEncodeSD3', 'EmptySD3LatentImage',
                    'TripleCLIPLoader'},
    'qwen_edit':   {'TextEncodeQwenImageEditPlus', 'QwenEditConfigPreparer',
                    'TextEncodeQwenImageEditPlusCustom_lrzjason', 'ModelSamplingAuraFlow'},
    'hunyuan_3d':  {'EmptyLatentHunyuan3Dv2', 'VAEDecodeHunyuan3D',
                    'Hunyuan3Dv2ConditioningMultiView', 'SaveGLB'},
    'training':    {'AnimaTrainingLauncher', 'AnimaModelDownloader', 'AnimaTrainingWizard',
                    'AnimaSDScriptsManager'},
    # Capability tags (can coexist with model tags)
    'video':       {'VHS_VideoCombine', 'VHS_LoadVideo', 'RIFE VFI', 'RIFEInterpolation',
                    'VideoToImages', 'SaveVideo', 'LoadVideo'},
    'audio':       {'ChatterBoxEngineNode', 'F5TTSEngineNode', 'UnifiedTTSSRTNode',
                    'CharacterVoicesNode', 'LoadAudio', 'SaveAudio', 'MMAudio',
                    'LTXVAddAudio', 'EmptyAudioLatent'},
    'inpainting':  {'InpaintModelConditioning', 'VAEEncodeForInpaint', 'LanPaintNode',
                    'DifferentialDiffusion', 'InpaintCropImproved'},
    'upscaling':   {'UltimateSDUpscale', 'UltimateSDUpscaleCustomSample',
                    'UltimateSDUpscaleNoUpscale', 'SeedVR2VideoUpscaler',
                    'ImageUpscaleWithModel'},
    'face':        {'FaceDetailer', 'FaceDetailerPipe', 'ReActorFaceSwap',
                    'NunchakuFluxPuLIDApplyV2', 'IPAdapterFaceID',
                    'InstantIDModelLoader', 'PulidModelLoader', 'ACE_Plus'},
    'controlnet':  {'ControlNetApplyAdvanced', 'ControlNetApplySD3',
                    'ControlNetLoader', 'AIO_Preprocessor',
                    'DepthAnythingV2Preprocessor', 'OpenposePreprocessor'},
    'captioning':  {'Florence2Run', 'DownloadAndLoadFlorence2Model',
                    'WD14Tagger', 'JoyCaptionAlpha', 'BLIPCaption',
                    'CLIPInterrogator', 'DeepDanbooru'},
    'segmentation':{'SAMLoader', 'SAMPredictor', 'GroundingDinoSAMSegment',
                    'DownloadAndLoadSAM2Model', 'SegmentAnything2',
                    'UltralyticsDetectorProvider', 'BboxDetectorSEGS'},
    'batch':       {'CR Prompt List', 'ImpactWildcardProcessor', 'LoadImageBatch',
                    'LoadImagesFromDirectory', 'VHS_LoadImages'},
    '3d':          {'SaveGLB', 'VoxelToMesh', 'EmptyLatentHunyuan3Dv2',
                    'Hunyuan3Dv2ConditioningMultiView'},
}

# Tags that imply "video" even if the direct video anchor isn't present
_VIDEO_IMPLIES = {'wan', 'ltx', 'hunyuan_vid', 'mochi', 'cogvideo', 'animatediff'}
# Tags that imply "image generation" as primary purpose
_IMAGE_GEN_IMPLIES = {'flux', 'nunchaku', 'zimage', 'sd3', 'qwen_edit'}

def tag_workflow(node_types: set) -> list[str]:
    """Return sorted list of ecosystem/capability tags for a workflow."""
    tags = []
    for tag, anchors in ECOSYSTEM_ANCHORS.items():
        if anchors & node_types:
            tags.append(tag)

    # Derived tags
    has_video_model = bool(set(tags) & _VIDEO_IMPLIES)
    has_sampler     = bool({'KSampler', 'KSamplerAdvanced', 'SamplerCustomAdvanced',
                            'WanVideoSampler', 'LTXVSampler', 'HunyuanVideoSampler'} & node_types)
    has_video_io    = bool({'VHS_VideoCombine', 'VHS_LoadVideo', 'SaveVideo'} & node_types)

    if has_video_model or (has_video_io and has_sampler):
        if 'video' not in tags:
            tags.append('video')
    if has_sampler and 'video' not in tags:
        if 'image_gen' not in tags:
            tags.append('image_gen')

    return sorted(set(tags))


# ═════════════════════════════════════════════════════════════════════════════
# DEDUPLICATION — deterministic fingerprinting
# ═════════════════════════════════════════════════════════════════════════════

def fingerprint(node_types: list) -> str:
    """Stable hash of sorted unique node types, excluding UI/routing noise."""
    ignore = SKIP_TYPES | REROUTE_TYPES | {
        'Note', 'MarkdownNote', 'NoteNode', 'Display Any (rgthree)',
        'GetNode', 'SetNode', 'easy getNode', 'easy setNode',
        'Any Switch (rgthree)', 'Fast Muter (rgthree)', 'Fast Bypasser (rgthree)',
        'Mute / Bypass Repeater (rgthree)', 'Seed (rgthree)',
        'SaveImage', 'PreviewImage',
    }
    meaningful = sorted({t for t in node_types
                         if t not in ignore and not t.startswith('[SUBGRAPH:')})
    return hashlib.md5(' '.join(meaningful).encode()).hexdigest()

def similarity(types_a: set, types_b: set) -> float:
    """Jaccard similarity between two node type sets."""
    if not types_a or not types_b:
        return 0.0
    return len(types_a & types_b) / len(types_a | types_b)

def cluster_duplicates(records: list, threshold: float = 0.85) -> dict:
    """
    Group workflows by similarity.
    Returns {cluster_id: [record, ...]} where cluster_id is the path of the
    'primary' workflow (first seen / largest).
    Exact duplicates (same fingerprint) always cluster together.
    """
    # Sort by node count descending so the most complex is the "primary"
    sorted_recs = sorted(records, key=lambda r: r['node_count'], reverse=True)

    clusters = {}   # primary_path -> [records]
    assigned = {}   # path -> primary_path

    for rec in sorted_recs:
        if rec['path'] in assigned:
            continue
        primary = rec['path']
        clusters[primary] = [rec]
        assigned[primary] = primary
        types_a = set(rec['node_types'])

        for other in sorted_recs:
            if other['path'] in assigned:
                continue
            types_b = set(other['node_types'])
            # Exact fingerprint match OR above threshold
            if (rec['fingerprint'] == other['fingerprint'] or
                    similarity(types_a, types_b) >= threshold):
                clusters[primary].append(other)
                assigned[other['path']] = primary

    return clusters


# ═════════════════════════════════════════════════════════════════════════════
# LLM BACKEND — Ollama and Claude API, swappable
# ═════════════════════════════════════════════════════════════════════════════

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

def ollama_enrich(workflow_data: dict, model: str, endpoint: str) -> Optional[dict]:
    """Send one workflow to Ollama for enrichment. Returns parsed dict or None."""
    if not _URLLIB_OK:
        return None

    notes = ' | '.join(
        nd['params'].get('text', '')[:100]
        for nd in workflow_data.get('nodes', [])
        if nd.get('is_text_node') and nd['params'].get('text', '').strip()
    )[:300]

    prompt = ENRICH_PROMPT.format(
        name   = workflow_data.get('path', '?'),
        path   = workflow_data.get('path', '?'),
        nodes  = ', '.join(sorted({nd['type'] for nd in workflow_data.get('nodes', [])
                                   if not nd['type'].startswith('[SUBGRAPH:')}))[:800],
        models = ', '.join(workflow_data.get('models', []))[:300],
        notes  = notes or 'none',
    )

    payload = json.dumps({
        'model': model,
        'prompt': prompt,
        'stream': False,
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
            raw = result.get('response', '').strip()
            # Strip markdown fences if present
            raw = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()
            return json.loads(raw)
    except Exception:
        return None


def claude_enrich(workflow_data: dict, api_key: str) -> Optional[dict]:
    """Send one workflow to Claude API for enrichment."""
    if not _ANTHROPIC_OK:
        return None

    notes = ' | '.join(
        nd['params'].get('text', '')[:100]
        for nd in workflow_data.get('nodes', [])
        if nd.get('is_text_node') and nd['params'].get('text', '').strip()
    )[:300]

    prompt = ENRICH_PROMPT.format(
        name   = workflow_data.get('path', '?'),
        path   = workflow_data.get('path', '?'),
        nodes  = ', '.join(sorted({nd['type'] for nd in workflow_data.get('nodes', [])
                                   if not nd['type'].startswith('[SUBGRAPH:')}))[:800],
        models = ', '.join(workflow_data.get('models', []))[:300],
        notes  = notes or 'none',
    )

    try:
        client = _anthropic_lib.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=500,
            messages=[{'role': 'user', 'content': prompt}],
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()
        return json.loads(raw)
    except Exception:
        return None


def list_ollama_models(endpoint: str) -> list[str]:
    """Fetch available models from Ollama. Returns empty list on failure."""
    if not _URLLIB_OK:
        return []
    try:
        req = urllib.request.Request(f'{endpoint.rstrip("/")}/api/tags', method='GET')
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m['name'] for m in data.get('models', [])]
    except Exception:
        return []


# ═════════════════════════════════════════════════════════════════════════════
# DATA INDEX — persisted sidecar file
# ═════════════════════════════════════════════════════════════════════════════

class WorkflowIndex:
    """
    Loads/saves wf_studio_index.json next to the workflow folder.
    Stores per-workflow metadata: tags, enrichment, fingerprint, etc.
    """
    def __init__(self, folder: Path):
        self.folder   = folder
        self.path     = folder.parent / 'wf_studio_index.json'
        self.records  = {}   # rel_path -> record dict
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, encoding='utf-8') as f:
                    self.records = json.load(f)
            except Exception:
                self.records = {}

    def save(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2)

    def get(self, rel_path: str) -> dict:
        return self.records.get(rel_path, {})

    def update(self, rel_path: str, data: dict):
        if rel_path not in self.records:
            self.records[rel_path] = {}
        self.records[rel_path].update(data)

    def mark_seen(self, rel_path: str, file_mtime: float):
        """Record file mod time so we can detect new/changed workflows."""
        if rel_path not in self.records:
            self.records[rel_path] = {}
        self.records[rel_path]['_last_mtime'] = file_mtime

    def is_new(self, rel_path: str) -> bool:
        return rel_path not in self.records

    def is_changed(self, rel_path: str, file_mtime: float) -> bool:
        saved = self.records.get(rel_path, {}).get('_last_mtime')
        return saved is not None and abs(file_mtime - saved) > 1.0

    def all_records(self) -> list[dict]:
        return [dict(v, path=k) for k, v in self.records.items()]


# ═════════════════════════════════════════════════════════════════════════════
# GUI
# ═════════════════════════════════════════════════════════════════════════════

BG    = '#111122'
PNL   = '#0c0c1e'
PNL2  = '#080818'
ACC   = '#6c72ff'
DIM   = '#3a3a66'
FG    = '#d4d4f0'
FG2   = '#8888bb'
GREEN = '#44cc88'
RED   = '#cc4455'
YEL   = '#ccaa44'
MONO  = ('Consolas', 10)
MONO_B = ('Consolas', 10, 'bold')
MONO_S = ('Consolas', 9)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('ComfyUI Workflow Studio')
        self.geometry('1200x800')
        self.minsize(900, 600)
        self.configure(bg=BG)

        self._folder: Optional[Path] = None
        self._index:  Optional[WorkflowIndex] = None
        self._records: list[dict] = []    # current full record list
        self._filtered: list[dict] = []   # after filter/search
        self._running = False

        self._build_style()
        self._build_menu()
        self._build_ui()
        self._load_config()
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ── Config ──────────────────────────────────────────────────────────────

    def _load_config(self):
        cfg = load_config()
        if cfg.get('folder') and Path(cfg['folder']).exists():
            self._folder = Path(cfg['folder'])
            self._folder_lbl.configure(text=str(self._folder))
            self._index = WorkflowIndex(self._folder)
            self._status(f'Restored folder: {self._folder}')
        if cfg.get('ollama_endpoint'):
            self._ollama_endpoint.set(cfg['ollama_endpoint'])
        if cfg.get('ollama_model'):
            self._ollama_model.set(cfg['ollama_model'])
        if cfg.get('claude_key'):
            self._claude_key.set(cfg['claude_key'])
        if cfg.get('llm_backend'):
            self._llm_backend.set(cfg['llm_backend'])
        if cfg.get('threshold') is not None:
            self._thresh_var.set(cfg['threshold'])
        if cfg.get('show_muted') is not None:
            self._show_muted.set(cfg['show_muted'])
        if cfg.get('include_notes') is not None:
            self._include_notes.set(cfg['include_notes'])

    def _on_close(self):
        cfg = {
            'folder':          str(self._folder) if self._folder else '',
            'ollama_endpoint': self._ollama_endpoint.get(),
            'ollama_model':    self._ollama_model.get(),
            'claude_key':      self._claude_key.get(),
            'llm_backend':     self._llm_backend.get(),
            'threshold':       self._thresh_var.get(),
            'show_muted':      self._show_muted.get(),
            'include_notes':   self._include_notes.get(),
        }
        save_config(cfg)
        if self._index:
            self._index.save()
        self.destroy()

    # ── Style ────────────────────────────────────────────────────────────────

    def _build_style(self):
        s = ttk.Style(self)
        s.theme_use('default')
        s.configure('TFrame',       background=BG)
        s.configure('Panel.TFrame', background=PNL)
        s.configure('TLabel',       background=BG, foreground=FG, font=MONO)
        s.configure('Dim.TLabel',   background=BG, foreground=FG2, font=MONO_S)
        s.configure('Head.TLabel',  background=BG, foreground=ACC, font=MONO_B)
        s.configure('TButton',      background=DIM, foreground=FG, font=MONO,
                    relief='flat', padding=(8, 4))
        s.map('TButton',
              background=[('active', ACC), ('disabled', '#222233')],
              foreground=[('disabled', DIM)])
        s.configure('Accent.TButton', background=ACC, foreground='#ffffff', font=MONO_B)
        s.map('Accent.TButton', background=[('active', '#8890ff')])
        s.configure('TNotebook',       background=BG, borderwidth=0)
        s.configure('TNotebook.Tab',   background=PNL, foreground=FG2, font=MONO,
                    padding=(12, 6))
        s.map('TNotebook.Tab',
              background=[('selected', BG)],
              foreground=[('selected', ACC)])
        s.configure('Treeview',        background=PNL2, foreground=FG, font=MONO_S,
                    fieldbackground=PNL2, borderwidth=0, rowheight=22)
        s.configure('Treeview.Heading', background=PNL, foreground=ACC, font=MONO_B)
        s.map('Treeview', background=[('selected', DIM)])
        s.configure('TScrollbar',  background=DIM, troughcolor=PNL2, borderwidth=0)
        s.configure('TProgressbar', troughcolor=PNL2, background=ACC, borderwidth=0)
        s.configure('TEntry',      fieldbackground=PNL, foreground=FG, font=MONO,
                    insertcolor=FG, borderwidth=1, relief='flat')
        s.configure('TCombobox',   fieldbackground=PNL, foreground=FG, font=MONO,
                    selectbackground=DIM)
        s.configure('TCheckbutton', background=BG, foreground=FG, font=MONO)
        s.configure('TSeparator',  background=DIM)

    # ── Menu ─────────────────────────────────────────────────────────────────

    def _build_menu(self):
        m = tk.Menu(self, bg=PNL, fg=FG, activebackground=DIM,
                    activeforeground=FG, font=MONO, tearoff=False)
        self.config(menu=m)

        fm = tk.Menu(m, bg=PNL, fg=FG, activebackground=DIM,
                     activeforeground=FG, tearoff=False)
        fm.add_command(label='Open folder…',          command=self._pick_folder)
        fm.add_command(label='Save index',            command=self._save_index)
        fm.add_separator()
        fm.add_command(label='Export combined text…', command=self._export_text)
        fm.add_separator()
        fm.add_command(label='Quit', command=self.destroy)
        m.add_cascade(label='File', menu=fm)

        am = tk.Menu(m, bg=PNL, fg=FG, activebackground=DIM,
                     activeforeground=FG, tearoff=False)
        am.add_command(label='Run extraction',        command=self._run_extraction)
        am.add_command(label='Run tagging',           command=self._run_tagging)
        am.add_command(label='Run deduplication',     command=self._run_dedup)
        am.add_separator()
        am.add_command(label='Run LLM enrichment',    command=self._run_llm_enrichment)
        m.add_cascade(label='Actions', menu=am)

    # ── Main UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self, style='Panel.TFrame')
        top.pack(fill='x', padx=0, pady=0)
        top.configure(padding=(12, 8))

        ttk.Label(top, text='ComfyUI Workflow Studio', style='Head.TLabel').pack(side='left')

        self._folder_lbl = ttk.Label(top, text='No folder selected', style='Dim.TLabel')
        self._folder_lbl.pack(side='left', padx=(16, 0))

        ttk.Button(top, text='Open Folder', command=self._pick_folder).pack(side='right', padx=4)
        ttk.Button(top, text='▶ Run All', style='Accent.TButton',
                   command=self._run_all).pack(side='right', padx=4)

        # Progress bar
        self._progress = ttk.Progressbar(self, mode='determinate')
        self._progress.pack(fill='x', padx=0)

        self._status_var = tk.StringVar(value='Ready.')
        ttk.Label(self, textvariable=self._status_var, style='Dim.TLabel',
                  padding=(12, 2)).pack(fill='x')

        ttk.Separator(self).pack(fill='x')

        # Notebook tabs
        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=0, pady=0)
        self._nb = nb

        self._tab_overview  = self._build_tab_overview(nb)
        self._tab_workflows = self._build_tab_workflows(nb)
        self._tab_dedup     = self._build_tab_dedup(nb)
        self._tab_settings  = self._build_tab_settings(nb)

        self._tab_log = self._build_tab_log(nb)

        nb.add(self._tab_overview,  text='  Overview  ')
        nb.add(self._tab_workflows, text='  Workflows  ')
        nb.add(self._tab_dedup,     text='  Duplicates  ')
        nb.add(self._tab_log,       text='  Log  ')
        nb.add(self._tab_settings,  text='  Settings  ')

    # ── Tab: Log ────────────────────────────────────────────────────────────

    def _build_tab_log(self, parent):
        f = ttk.Frame(parent)
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)

        self._log_text = tk.Text(f, bg=PNL2, fg=FG2, font=MONO_S,
                                  state='disabled', relief='flat',
                                  insertbackground=FG, wrap='word')
        self._log_text.grid(row=0, column=0, sticky='nsew')
        sb = ttk.Scrollbar(f, orient='vertical', command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky='ns')

        btn_f = ttk.Frame(f, style='Panel.TFrame', padding=(8,4))
        btn_f.grid(row=1, column=0, columnspan=2, sticky='ew')
        ttk.Button(btn_f, text='Clear log', command=self._clear_log).pack(side='left')

        return f

    # ── Tab: Overview ────────────────────────────────────────────────────────

    def _build_tab_overview(self, parent):
        f = ttk.Frame(parent)
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(1, weight=1)

        # Stats row
        stats_f = ttk.Frame(f, style='Panel.TFrame', padding=12)
        stats_f.grid(row=0, column=0, columnspan=2, sticky='ew', padx=8, pady=(8,4))

        self._stat_vars = {}
        for i, (key, label) in enumerate([
            ('total',    'Workflows'),
            ('tagged',   'Tagged'),
            ('enriched', 'LLM enriched'),
            ('dupes',    'Duplicates found'),
            ('models',   'Unique models'),
            ('nodes',    'Node types seen'),
        ]):
            col_f = ttk.Frame(stats_f)
            col_f.pack(side='left', expand=True, padx=8)
            v = tk.StringVar(value='—')
            self._stat_vars[key] = v
            ttk.Label(col_f, textvariable=v, font=('Consolas', 18, 'bold'),
                      foreground=ACC, background=PNL).pack()
            ttk.Label(col_f, text=label, style='Dim.TLabel',
                      background=PNL).pack()

        # Node frequency list (left)
        lf = ttk.Frame(f, style='Panel.TFrame')
        lf.grid(row=1, column=0, sticky='nsew', padx=(8,4), pady=(4,8))
        lf.rowconfigure(1, weight=1)
        lf.columnconfigure(0, weight=1)

        ttk.Label(lf, text='Top node types', style='Head.TLabel',
                  background=PNL, padding=(8,6)).grid(row=0, column=0, sticky='ew')

        self._node_tree = self._make_tree(lf, ('Type', 'Count'), widths=(300, 80))
        self._node_tree.grid(row=1, column=0, sticky='nsew')
        sb = ttk.Scrollbar(lf, orient='vertical', command=self._node_tree.yview)
        self._node_tree.configure(yscrollcommand=sb.set)
        sb.grid(row=1, column=1, sticky='ns')

        # Model list (right)
        rf = ttk.Frame(f, style='Panel.TFrame')
        rf.grid(row=1, column=1, sticky='nsew', padx=(4,8), pady=(4,8))
        rf.rowconfigure(1, weight=1)
        rf.columnconfigure(0, weight=1)

        ttk.Label(rf, text='Models referenced', style='Head.TLabel',
                  background=PNL, padding=(8,6)).grid(row=0, column=0, sticky='ew')

        self._model_tree = self._make_tree(rf, ('File', 'Count'), widths=(300, 80))
        self._model_tree.grid(row=1, column=0, sticky='nsew')
        sb2 = ttk.Scrollbar(rf, orient='vertical', command=self._model_tree.yview)
        self._model_tree.configure(yscrollcommand=sb2.set)
        sb2.grid(row=1, column=1, sticky='ns')

        return f

    # ── Tab: Workflows ───────────────────────────────────────────────────────

    def _build_tab_workflows(self, parent):
        f = ttk.Frame(parent)
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=1)

        # Filter bar
        bar = ttk.Frame(f, style='Panel.TFrame', padding=(8, 6))
        bar.grid(row=0, column=0, sticky='ew')

        ttk.Label(bar, text='Search:', background=PNL).pack(side='left')
        self._search_var = tk.StringVar()
        self._search_var.trace_add('write', lambda *_: self._apply_filter())
        ttk.Entry(bar, textvariable=self._search_var, width=30).pack(side='left', padx=4)

        ttk.Label(bar, text='Tag:', background=PNL).pack(side='left', padx=(12, 0))
        self._tag_filter_var = tk.StringVar(value='all')
        self._tag_combo = ttk.Combobox(bar, textvariable=self._tag_filter_var,
                                        state='readonly', width=20)
        self._tag_combo.pack(side='left', padx=4)
        self._tag_combo.bind('<<ComboboxSelected>>', lambda _: self._apply_filter())

        self._show_new_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text='New/Changed only',
                        variable=self._show_new_only,
                        command=self._apply_filter).pack(side='left', padx=(12,0))

        self._count_lbl = ttk.Label(bar, text='', style='Dim.TLabel', background=PNL)
        self._count_lbl.pack(side='right', padx=8)

        # Workflow tree
        cols = ('Name', 'Path', 'Tags', 'Nodes', 'Status', 'Enriched', 'Summary')
        self._wf_tree = self._make_tree(f, cols, widths=(220, 200, 140, 60, 65, 70, 300))
        self._wf_tree.grid(row=1, column=0, sticky='nsew')
        sb = ttk.Scrollbar(f, orient='vertical', command=self._wf_tree.yview)
        self._wf_tree.configure(yscrollcommand=sb.set)
        sb.grid(row=1, column=1, sticky='ns')

        sb2 = ttk.Scrollbar(f, orient='horizontal', command=self._wf_tree.xview)
        self._wf_tree.configure(xscrollcommand=sb2.set)
        sb2.grid(row=2, column=0, sticky='ew')

        # Detail panel
        det = ttk.Frame(f, style='Panel.TFrame', padding=8)
        det.grid(row=3, column=0, columnspan=2, sticky='ew')

        self._detail_var = tk.StringVar(value='Select a workflow to see details.')
        ttk.Label(det, textvariable=self._detail_var, style='Dim.TLabel',
                  background=PNL, wraplength=900, justify='left').pack(fill='x')
        self._wf_tree.tag_configure('new',     foreground=GREEN)
        self._wf_tree.tag_configure('changed', foreground=YEL)
        self._wf_tree.bind('<Button-3>', self._wf_context_menu)
        self._wf_tree.bind('<Button-2>', self._wf_context_menu)
        self._wf_tree.bind('<<TreeviewSelect>>', self._on_wf_select)

        return f

    # ── Tab: Duplicates ──────────────────────────────────────────────────────

    def _build_tab_dedup(self, parent):
        f = ttk.Frame(parent)
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=1)

        # Controls
        ctrl = ttk.Frame(f, style='Panel.TFrame', padding=(8,6))
        ctrl.grid(row=0, column=0, columnspan=2, sticky='ew')

        ttk.Label(ctrl, text='Similarity threshold:', background=PNL).pack(side='left')
        self._thresh_var = tk.DoubleVar(value=0.85)
        thresh_sl = tk.Scale(ctrl, from_=0.5, to=1.0, resolution=0.05, orient='horizontal',
                             variable=self._thresh_var, length=200,
                             bg=PNL, fg=FG, troughcolor=PNL2, highlightthickness=0,
                             command=lambda _: self._refresh_dedup_display())
        thresh_sl.pack(side='left', padx=8)

        self._thresh_lbl = ttk.Label(ctrl, text='85%', background=PNL, foreground=YEL)
        self._thresh_lbl.pack(side='left')
        self._thresh_var.trace_add('write',
            lambda *_: self._thresh_lbl.configure(
                text=f'{int(self._thresh_var.get()*100)}%'))

        ttk.Button(ctrl, text='Move duplicates to folder…',
                   command=self._move_dupes).pack(side='right', padx=4)
        ttk.Button(ctrl, text='Refresh', command=self._refresh_dedup_display
                   ).pack(side='right', padx=4)

        self._dupe_count_lbl = ttk.Label(ctrl, text='', foreground=YEL,
                                          background=PNL, font=MONO_S)
        self._dupe_count_lbl.pack(side='right', padx=12)

        # Duplicate tree
        cols = ('Cluster', 'File', 'Path', 'Similarity', 'Node count')
        self._dupe_tree = self._make_tree(f, cols, widths=(30, 220, 280, 80, 90))
        self._dupe_tree.grid(row=1, column=0, sticky='nsew')
        sb = ttk.Scrollbar(f, orient='vertical', command=self._dupe_tree.yview)
        self._dupe_tree.configure(yscrollcommand=sb.set)
        sb.grid(row=1, column=1, sticky='ns')

        # Tag colours
        self._dupe_tree.tag_configure('primary',   foreground=GREEN)
        self._dupe_tree.tag_configure('duplicate', foreground=YEL)
        self._dupe_tree.tag_configure('exact',     foreground=RED)

        self._dupe_tree.bind('<Button-3>', self._dupe_context_menu)
        self._dupe_tree.bind('<Button-2>', self._dupe_context_menu)  # macOS

        self._dedup_clusters = {}   # computed clusters cache

        return f

    # ── Tab: Settings ────────────────────────────────────────────────────────

    def _build_tab_settings(self, parent):
        f = ttk.Frame(parent, padding=20)

        row = 0
        def label(text):
            nonlocal row
            ttk.Label(f, text=text, style='Head.TLabel').grid(
                row=row, column=0, columnspan=2, sticky='w', pady=(14, 4))
            row += 1

        def field(text, var, width=40):
            nonlocal row
            ttk.Label(f, text=text).grid(row=row, column=0, sticky='w', pady=2)
            ttk.Entry(f, textvariable=var, width=width).grid(
                row=row, column=1, sticky='ew', padx=(8, 0))
            row += 1

        def combo(text, var, values, width=30):
            nonlocal row
            ttk.Label(f, text=text).grid(row=row, column=0, sticky='w', pady=2)
            c = ttk.Combobox(f, textvariable=var, values=values,
                             state='readonly', width=width)
            c.grid(row=row, column=1, sticky='w', padx=(8, 0))
            row += 1
            return c

        f.columnconfigure(1, weight=1)

        label('Ollama settings')
        self._ollama_endpoint = tk.StringVar(value='http://localhost:11434')
        field('Endpoint:', self._ollama_endpoint)
        self._ollama_model = tk.StringVar(value='')
        self._model_combo = combo('Model:', self._ollama_model, [])

        ttk.Button(f, text='Refresh model list',
                   command=self._refresh_ollama_models).grid(
            row=row, column=1, sticky='w', padx=(8,0), pady=4)
        row += 1

        self._ollama_status = ttk.Label(f, text='', style='Dim.TLabel')
        self._ollama_status.grid(row=row, column=0, columnspan=2, sticky='w')
        row += 1

        ttk.Separator(f).grid(row=row, column=0, columnspan=2, sticky='ew', pady=12)
        row += 1

        label('Claude API settings (optional)')
        self._claude_key = tk.StringVar(value='')
        field('API key:', self._claude_key, 50)

        self._llm_backend = tk.StringVar(value='ollama')
        ttk.Label(f, text='Backend:').grid(row=row, column=0, sticky='w')
        for val, txt in [('ollama', 'Ollama (local)'), ('claude', 'Claude API')]:
            ttk.Radiobutton(f, text=txt, variable=self._llm_backend,
                            value=val).grid(row=row, column=1, sticky='w', padx=(8+80*(['ollama','claude'].index(val)),0))
        row += 1

        ttk.Separator(f).grid(row=row, column=0, columnspan=2, sticky='ew', pady=12)
        row += 1

        label('Extraction settings')
        self._show_muted  = tk.BooleanVar(value=False)
        self._include_notes = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text='Include muted nodes',
                        variable=self._show_muted).grid(
            row=row, column=0, columnspan=2, sticky='w')
        row += 1
        ttk.Checkbutton(f, text='Include Note/Display Any text content',
                        variable=self._include_notes).grid(
            row=row, column=0, columnspan=2, sticky='w')
        row += 1

        ttk.Button(f, text='Test Ollama connection',
                   command=self._test_ollama).grid(
            row=row, column=0, columnspan=2, sticky='w', pady=(16,0))
        row += 1

        return f

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _make_tree(self, parent, cols, widths=None):
        t = ttk.Treeview(parent, columns=cols, show='headings', selectmode='browse')
        for i, col in enumerate(cols):
            w = (widths[i] if widths and i < len(widths) else 150)
            t.heading(col, text=col, anchor='w')
            t.column(col, width=w, anchor='w', stretch=(i == len(cols)-1))
        return t

    def _log(self, msg: str, level: str = 'info'):
        """Write a timestamped line to the log panel."""
        import time as _time
        ts = _time.strftime('%H:%M:%S')
        colours = {'info': FG2, 'ok': GREEN, 'warn': YEL, 'error': RED}
        colour = colours.get(level, FG2)
        try:
            self._log_text.configure(state='normal')
            self._log_text.tag_configure(level, foreground=colour)
            self._log_text.insert('end', f'[{ts}] {msg}\n', level)
            self._log_text.see('end')
            self._log_text.configure(state='disabled')
        except Exception:
            pass

    def _clear_log(self):
        self._log_text.configure(state='normal')
        self._log_text.delete('1.0', 'end')
        self._log_text.configure(state='disabled')

    def _status(self, msg: str, level: str = 'info'):
        self._status_var.set(msg)
        self._log(msg, level)
        self.update_idletasks()

    def _set_progress(self, val: float):
        self._progress['value'] = val * 100
        self.update_idletasks()

    # ── Actions ──────────────────────────────────────────────────────────────

    def _pick_folder(self):
        folder = filedialog.askdirectory(title='Select workflow folder')
        if not folder:
            return
        self._folder = Path(folder)
        self._folder_lbl.configure(text=str(self._folder))
        self._index = WorkflowIndex(self._folder)
        self._records = []
        self._status(f'Folder set: {self._folder}  |  '
                     f'{len(list(self._folder.rglob("*.json")))} JSON files found')
        self._refresh_tag_filter()

    def _save_index(self):
        if self._index:
            self._index.save()
            self._status('Index saved.')

    def _export_text(self):
        if not self._folder or not _extractor_available:
            messagebox.showwarning('No data', 'Run extraction first.')
            return
        out = filedialog.asksaveasfilename(
            defaultextension='.txt',
            filetypes=[('Text', '*.txt')],
            initialfile='workflows_combined.txt',
        )
        if out:
            self._run_in_thread(self._do_export_text, out)

    def _run_all(self):
        if not self._folder:
            messagebox.showwarning('No folder', 'Select a workflow folder first.')
            return
        self._run_in_thread(self._do_run_all)

    def _run_extraction(self):
        if not self._folder:
            messagebox.showwarning('No folder', 'Select a workflow folder first.')
            return
        self._run_in_thread(self._do_extraction)

    def _run_tagging(self):
        if not self._records:
            messagebox.showwarning('No data', 'Run extraction first.')
            return
        self._run_in_thread(self._do_tagging)

    def _run_dedup(self):
        if not self._records:
            messagebox.showwarning('No data', 'Run extraction first.')
            return
        self._run_in_thread(self._do_dedup)

    def _run_llm_enrichment(self):
        if not self._records:
            messagebox.showwarning('No data', 'Run extraction first.')
            return
        backend = self._llm_backend.get()
        if backend == 'ollama' and not self._ollama_model.get():
            messagebox.showwarning('No model', 'Select an Ollama model in Settings first.')
            return
        if backend == 'claude' and not self._claude_key.get():
            messagebox.showwarning('No API key', 'Enter Claude API key in Settings first.')
            return
        self._run_in_thread(self._do_llm_enrichment)

    def _run_in_thread(self, fn, *args):
        if self._running:
            self._status('Already running — please wait.')
            return
        self._running = True
        threading.Thread(target=self._wrapped_run, args=(fn, *args),
                         daemon=True).start()

    def _wrapped_run(self, fn, *args):
        try:
            fn(*args)
        except Exception as e:
            self.after(0, self._status, f'Error: {e}')
        finally:
            self._running = False
            self.after(0, self._set_progress, 0)

    # ── Core operations ──────────────────────────────────────────────────────

    def _do_run_all(self):
        self._do_extraction()
        self._do_tagging()
        self._do_dedup()
        self.after(0, self._status, 'All steps complete.')

    def _do_extraction(self):
        if not _extractor_available:
            self.after(0, self._status,
                       'Extractor not available — place wf_extract.py in the same folder.')
            return

        files = sorted(self._folder.glob('**/*.json'))
        files = [f for f in files if f.stat().st_size > 500 and is_comfyui_workflow(f)]
        if not files:
            self.after(0, self._status, 'No workflow JSON files found.')
            return

        self.after(0, self._status, f'Pass 1/2 — building schema registry ({len(files)} files)…')

        registry = SchemaRegistry()
        for fp in files:
            try:
                with open(fp, encoding='utf-8', errors='replace') as f:
                    data = json.load(f)
                for n in data.get('nodes', []):
                    registry.observe(n)
                for sg in data.get('definitions', {}).get('subgraphs', []):
                    for n in sg.get('nodes', []):
                        registry.observe(n)
            except Exception:
                pass
        registry.finalize()

        self.after(0, self._status, f'Pass 2/2 — extracting…')
        records = []
        type_counts  = defaultdict(int)
        model_counts = defaultdict(int)

        for i, fp in enumerate(files):
            self.after(0, self._set_progress, (i + 1) / len(files))
            try:
                rel = str(fp.relative_to(self._folder))
            except:
                rel = fp.name

            try:
                wf_data = extract(fp, registry,
                                  show_muted=self._show_muted.get(),
                                  include_notes=self._include_notes.get())
            except Exception:
                continue

            wf_data['path'] = rel
            node_types = [nd['type'] for nd in wf_data.get('nodes', [])]

            rec = {
                'path':       rel,
                'name':       fp.name,
                'node_count': wf_data['node_count'],
                'node_types': node_types,
                'models':     wf_data.get('models', []),
                'groups':     wf_data.get('groups', []),
                'meta':       wf_data.get('meta', {}),
                'fingerprint': fingerprint(node_types),
                'tags':       [],
                'enrichment': {},
                'wf_data':    wf_data,
            }

            # Merge saved index data
            saved = self._index.get(rel) if self._index else {}
            rec['tags']       = saved.get('tags', [])
            rec['enrichment'] = saved.get('enrichment', {})

            # Track new/changed status
            try:
                mtime = fp.stat().st_mtime
            except Exception:
                mtime = 0.0
            rec['is_new']     = self._index.is_new(rel) if self._index else False
            rec['is_changed'] = self._index.is_changed(rel, mtime) if self._index else False
            if self._index:
                self._index.mark_seen(rel, mtime)

            records.append(rec)

            for t in node_types:
                if not t.startswith('[SUBGRAPH:'):
                    type_counts[t] += 1
            for itype, cnt in wf_data.get('sg_inner_counts', {}).items():
                type_counts[itype] += cnt
            for m in wf_data.get('models', []):
                base = re.sub(r'[/\\]+', '/', m).strip('/').rsplit('/', 1)[-1]
                model_counts[base] += 1

        self._records = records
        self.after(0, self._update_overview, type_counts, model_counts)
        self.after(0, self._refresh_wf_table)
        self.after(0, self._refresh_tag_filter)
        self.after(0, self._status, f'Extraction complete — {len(records)} workflows.', 'ok')

    def _do_tagging(self):
        self.after(0, self._status, 'Tagging workflows…')
        for i, rec in enumerate(self._records):
            self.after(0, self._set_progress, (i + 1) / len(self._records))
            types = set(rec['node_types'])
            rec['tags'] = tag_workflow(types)
            if self._index:
                self._index.update(rec['path'], {'tags': rec['tags']})
        if self._index:
            self._index.save()
        self.after(0, self._refresh_wf_table)
        self.after(0, self._refresh_tag_filter)
        n_tagged = sum(1 for r in self._records if r['tags'])
        self.after(0, self._status, f'Tagging complete — {n_tagged} workflows tagged.', 'ok')

    def _do_dedup(self):
        self.after(0, self._status, 'Computing similarity clusters…')
        clusters = cluster_duplicates(self._records, threshold=self._thresh_var.get())
        self._dedup_clusters = clusters
        self.after(0, self._refresh_dedup_display)
        dupe_count = sum(len(v) - 1 for v in clusters.values() if len(v) > 1)
        self.after(0, self._status,
                   f'Deduplication complete — {dupe_count} potential duplicates '
                   f'across {sum(1 for v in clusters.values() if len(v)>1)} clusters.')

    def _do_llm_enrichment(self):
        backend  = self._llm_backend.get()
        to_enrich = [r for r in self._records if not r.get('enrichment')]
        if not to_enrich:
            self.after(0, self._status, 'All workflows already enriched.')
            return

        self.after(0, self._status,
                   f'LLM enrichment ({backend}) — {len(to_enrich)} workflows…')

        for i, rec in enumerate(to_enrich):
            self.after(0, self._set_progress, (i + 1) / len(to_enrich))
            self.after(0, self._status,
                       f'Enriching {i+1}/{len(to_enrich)}: {rec["name"]}')

            result = None
            if backend == 'ollama':
                result = ollama_enrich(rec['wf_data'], self._ollama_model.get(),
                                       self._ollama_endpoint.get())
            elif backend == 'claude':
                result = claude_enrich(rec['wf_data'], self._claude_key.get())

            if result:
                rec['enrichment'] = result
                if self._index:
                    self._index.update(rec['path'], {'enrichment': result})

        if self._index:
            self._index.save()
        self.after(0, self._refresh_wf_table)
        n_done = sum(1 for r in self._records if r.get('enrichment'))
        self.after(0, self._status, f'LLM enrichment complete — {n_done} enriched.', 'ok')

    def _do_export_text(self, outpath: str):
        if not _extractor_available:
            return
        self.after(0, self._status, 'Exporting combined text…')
        from collections import defaultdict as _dd
        registry = SchemaRegistry()
        files = sorted(self._folder.glob('**/*.json'))
        files = [f for f in files if f.stat().st_size > 500 and is_comfyui_workflow(f)]
        for fp in files:
            try:
                with open(fp, encoding='utf-8', errors='replace') as f:
                    data = json.load(f)
                for n in data.get('nodes', []):
                    registry.observe(n)
            except:
                pass
        registry.finalize()

        sections = []
        type_counts  = _dd(int)
        model_counts = _dd(int)

        for i, fp in enumerate(files):
            self.after(0, self._set_progress, (i+1)/len(files))
            try:
                rel = str(fp.relative_to(self._folder))
            except:
                rel = fp.name
            try:
                wf = extract(fp, registry,
                             show_muted=self._show_muted.get(),
                             include_notes=self._include_notes.get())
                wf['path'] = rel
                sections.append(render(fp.name, wf, compact=False))
                for nd in wf.get('nodes', []):
                    t = nd['type']
                    if not t.startswith('[SUBGRAPH:'):
                        type_counts[t] += 1
                for itype, cnt in wf.get('sg_inner_counts', {}).items():
                    type_counts[itype] += cnt
                for m in wf.get('models', []):
                    base = re.sub(r'[/\\]+', '/', m).strip('/').rsplit('/', 1)[-1]
                    model_counts[base] += 1
            except:
                pass

        # Build summary
        summary = ['='*66, 'COMBINED WORKFLOW ANALYSIS', '='*66,
                   f'Workflows: {len(files)}', '', 'Node type frequency:']
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            summary.append(f'  {c:4d}x  {t}')
        if model_counts:
            summary.append('\nModel files:')
            for base, total in sorted(model_counts.items(), key=lambda x: -x[1]):
                cnt = f' ({total}x)' if total > 1 else ''
                summary.append(f'  {base}{cnt}')

        full = '\n'.join(summary) + '\n\n' + '\n\n'.join(sections)
        with open(outpath, 'w', encoding='utf-8') as f:
            f.write(full)
        self.after(0, self._status, f'Exported → {outpath}')

    # ── UI Refresh ────────────────────────────────────────────────────────────

    def _update_overview(self, type_counts, model_counts):
        n_enriched = sum(1 for r in self._records if r.get('enrichment'))
        n_dupes    = sum(len(v)-1 for v in self._dedup_clusters.values()
                        if len(v) > 1) if self._dedup_clusters else 0

        self._stat_vars['total'].set(str(len(self._records)))
        self._stat_vars['tagged'].set(
            str(sum(1 for r in self._records if r.get('tags'))))
        self._stat_vars['enriched'].set(str(n_enriched))
        self._stat_vars['dupes'].set(str(n_dupes))
        self._stat_vars['models'].set(str(len(model_counts)))
        self._stat_vars['nodes'].set(str(len(type_counts)))

        # Node frequency tree
        self._node_tree.delete(*self._node_tree.get_children())
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1])[:200]:
            self._node_tree.insert('', 'end', values=(t, c))

        # Model tree
        self._model_tree.delete(*self._model_tree.get_children())
        for base, total in sorted(model_counts.items(), key=lambda x: -x[1])[:200]:
            self._model_tree.insert('', 'end', values=(base, total))

    def _refresh_wf_table(self):
        self._apply_filter()

    def _apply_filter(self):
        search = self._search_var.get().lower().strip()
        tag_f  = self._tag_filter_var.get()

        filtered = self._records
        if search:
            filtered = [r for r in filtered if
                        search in r['name'].lower() or
                        search in r['path'].lower() or
                        any(search in t.lower() for t in r.get('tags', [])) or
                        search in r.get('enrichment', {}).get('summary', '').lower() or
                        search in r.get('enrichment', {}).get('primary_purpose', '').lower()]
        if tag_f and tag_f != 'all':
            filtered = [r for r in filtered if tag_f in r.get('tags', [])]
        if self._show_new_only.get():
            filtered = [r for r in filtered
                        if r.get('is_new') or r.get('is_changed')]

        self._filtered = filtered
        self._count_lbl.configure(text=f'{len(filtered)} / {len(self._records)}')

        self._wf_tree.delete(*self._wf_tree.get_children())
        for rec in filtered:
            tags = ', '.join(rec.get('tags', [])) or '—'
            enr  = rec.get('enrichment', {})
            summary = enr.get('summary', '') or enr.get('primary_purpose', '') or '—'
            enriched = '✓' if enr else '—'
            status = ('new' if rec.get('is_new') else
                      'changed' if rec.get('is_changed') else '')
            row_tag = (status,) if status else ()
            self._wf_tree.insert('', 'end', iid=rec['path'], tags=row_tag, values=(
                rec['name'], rec['path'], tags,
                rec['node_count'], status or '—', enriched, summary[:120],
            ))

    def _refresh_tag_filter(self):
        all_tags = sorted({t for r in self._records for t in r.get('tags', [])})
        self._tag_combo['values'] = ['all'] + all_tags
        if not self._tag_filter_var.get():
            self._tag_filter_var.set('all')

    def _wf_context_menu(self, event):
        iid = self._wf_tree.identify_row(event.y)
        if not iid:
            return
        self._wf_tree.selection_set(iid)
        rec = next((r for r in self._records if r['path'] == iid), None)
        if not rec:
            return
        full_path = str(self._folder / rec['path']) if self._folder else rec['path']
        menu = tk.Menu(self, tearoff=False, bg=PNL, fg=FG,
                       activebackground=DIM, activeforeground=FG, font=MONO)
        menu.add_command(label='Copy path',
                         command=lambda: self._copy_to_clipboard(full_path))
        menu.add_command(label='Open containing folder',
                         command=lambda: self._open_folder(full_path))
        menu.add_command(label='Open file in default app',
                         command=lambda: self._open_file(full_path))
        menu.post(event.x_root, event.y_root)

    def _on_wf_select(self, _event=None):
        sel = self._wf_tree.selection()
        if not sel:
            return
        path = sel[0]
        rec  = next((r for r in self._records if r['path'] == path), None)
        if not rec:
            return

        enr   = rec.get('enrichment', {})
        meta  = rec.get('meta', {})
        parts = [
            f'File: {rec["name"]}',
            f'Path: {rec["path"]}',
            f'Nodes: {rec["node_count"]}',
        ]
        if rec.get('tags'):
            parts.append(f'Tags: {", ".join(rec["tags"])}')
        if meta.get('author'):
            parts.append(f'Author: {meta["author"]}')
        if meta.get('frontend_version'):
            parts.append(f'ComfyUI: {meta["frontend_version"]}')
        if enr.get('summary'):
            parts.append(f'Summary: {enr["summary"]}')
        if enr.get('primary_purpose'):
            parts.append(f'Purpose: {enr["primary_purpose"]}')
        if enr.get('notable') and enr['notable'] != 'nothing notable':
            parts.append(f'Notable: {enr["notable"]}')
        if enr.get('quality_signal'):
            parts.append(f'Quality: {enr["quality_signal"]}')
        if rec.get('models'):
            model_names = [re.sub(r'[/\\]+','/',m).rsplit('/',1)[-1]
                           for m in rec['models'][:6]]
            parts.append(f'Models: {", ".join(model_names)}')

        self._detail_var.set('  |  '.join(parts))

    def _dupe_context_menu(self, event):
        """Right-click menu on duplicate tree rows."""
        iid = self._dupe_tree.identify_row(event.y)
        if not iid:
            return
        self._dupe_tree.selection_set(iid)
        item = self._dupe_tree.item(iid)
        vals = item.get('values', [])
        if not vals or len(vals) < 3:
            return
        rel_path = vals[2]  # Path column

        menu = tk.Menu(self, tearoff=False, bg=PNL, fg=FG,
                       activebackground=DIM, activeforeground=FG,
                       font=MONO)
        full_path = str(self._folder / rel_path) if self._folder else rel_path

        menu.add_command(label='Copy path',
                         command=lambda: self._copy_to_clipboard(full_path))
        menu.add_command(label='Copy relative path',
                         command=lambda: self._copy_to_clipboard(rel_path))
        menu.add_separator()
        menu.add_command(label='Open containing folder',
                         command=lambda: self._open_folder(full_path))
        menu.add_command(label='Open file in default app',
                         command=lambda: self._open_file(full_path))
        menu.add_separator()
        menu.add_command(label='Compare with primary in text editor',
                         command=lambda: self._compare_with_primary(iid, full_path))
        menu.post(event.x_root, event.y_root)

    def _copy_to_clipboard(self, text: str):
        self.clipboard_clear()
        self.clipboard_append(text)
        self._status(f'Copied: {text}')

    def _open_folder(self, path: str):
        import subprocess
        folder = str(Path(path).parent)
        try:
            if sys.platform == 'win32':
                subprocess.Popen(['explorer', folder])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', folder])
            else:
                subprocess.Popen(['xdg-open', folder])
        except Exception as e:
            self._status(f'Could not open folder: {e}', 'error')

    def _open_file(self, path: str):
        import subprocess
        try:
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])
        except Exception as e:
            self._status(f'Could not open file: {e}', 'error')

    def _compare_with_primary(self, iid: str, full_path: str):
        """Open this file and its cluster primary in a text editor side by side."""
        import subprocess
        item = self._dupe_tree.item(iid)
        vals = item.get('values', [])
        cluster_id = vals[0] if vals else ''

        # Find the primary for this cluster
        primary_path = None
        for child_iid in self._dupe_tree.get_children():
            cv = self._dupe_tree.item(child_iid).get('values', [])
            if cv and cv[0] == cluster_id and cv[3] == 'primary':
                primary_path = str(self._folder / cv[2]) if self._folder else cv[2]
                break

        if not primary_path or primary_path == full_path:
            self._status('Could not find primary for this cluster.', 'warn')
            return

        try:
            if sys.platform == 'win32':
                # Try Notepad++ with compare plugin, fall back to separate notepad windows
                try:
                    subprocess.Popen([
                        'notepad++', '-multiInst', '-nosession',
                        primary_path, full_path
                    ])
                except FileNotFoundError:
                    subprocess.Popen(['notepad', primary_path])
                    subprocess.Popen(['notepad', full_path])
            else:
                subprocess.Popen(['diff', '--color', primary_path, full_path])
        except Exception as e:
            self._status(f'Could not open comparison: {e}', 'error')

    def _refresh_dedup_display(self, *_):
        if not self._records:
            return
        threshold = self._thresh_var.get()
        clusters = cluster_duplicates(self._records, threshold=threshold)
        self._dedup_clusters = clusters

        # Only show clusters with duplicates
        dup_clusters = {p: recs for p, recs in clusters.items() if len(recs) > 1}
        dupe_count   = sum(len(v)-1 for v in dup_clusters.values())

        self._dupe_count_lbl.configure(
            text=f'{dupe_count} duplicates in {len(dup_clusters)} clusters')

        self._dupe_tree.delete(*self._dupe_tree.get_children())
        for cluster_id, (primary, *dupes) in enumerate(
                sorted(dup_clusters.values(), key=len, reverse=True), 1):

            # Primary row
            self._dupe_tree.insert('', 'end', tags=('primary',), values=(
                f'#{cluster_id}', primary['name'], primary['path'],
                'primary', primary['node_count'],
            ))

            for dup in dupes:
                types_p = set(primary['node_types'])
                types_d = set(dup['node_types'])
                sim = similarity(types_p, types_d)
                is_exact = primary['fingerprint'] == dup['fingerprint']
                tag = 'exact' if is_exact else 'duplicate'
                self._dupe_tree.insert('', 'end', tags=(tag,), values=(
                    f'#{cluster_id}', dup['name'], dup['path'],
                    f'{"EXACT" if is_exact else f"{sim:.0%}"}',
                    dup['node_count'],
                ))

        # Update stat
        self._stat_vars['dupes'].set(str(dupe_count))

    def _move_dupes(self):
        if not self._dedup_clusters:
            messagebox.showinfo('No clusters', 'Run deduplication first.')
            return
        dup_clusters = {p: recs for p, recs in self._dedup_clusters.items()
                        if len(recs) > 1}
        if not dup_clusters:
            messagebox.showinfo('No duplicates', 'No duplicate clusters found.')
            return

        dest = filedialog.askdirectory(title='Move duplicates to folder…')
        if not dest:
            return
        dest_path = Path(dest)

        moved = 0
        errors = []
        for primary_path, recs in dup_clusters.items():
            for rec in recs[1:]:   # skip primary
                src = self._folder / rec['path']
                # Mirror folder structure
                try:
                    rel_parts = Path(rec['path']).parts
                    dst = dest_path.joinpath(*rel_parts)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    moved += 1
                except Exception as e:
                    errors.append(f'{rec["path"]}: {e}')

        msg = f'Moved {moved} duplicate(s) to {dest_path}.'
        if errors:
            msg += f'\n\nErrors ({len(errors)}):\n' + '\n'.join(errors[:5])
        messagebox.showinfo('Done', msg)
        # Re-run extraction to refresh
        self._run_extraction()

    # ── Ollama ───────────────────────────────────────────────────────────────

    def _refresh_ollama_models(self):
        models = list_ollama_models(self._ollama_endpoint.get())
        if models:
            self._model_combo['values'] = models
            if not self._ollama_model.get() or self._ollama_model.get() not in models:
                self._ollama_model.set(models[0])
            self._ollama_status.configure(
                text=f'✓ {len(models)} model(s) found', foreground=GREEN)
        else:
            self._ollama_status.configure(
                text='✗ Could not connect — is Ollama running?', foreground=RED)

    def _test_ollama(self):
        self._refresh_ollama_models()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = App()
    app.mainloop()
