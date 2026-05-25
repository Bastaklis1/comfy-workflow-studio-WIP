#!/usr/bin/env python3
"""
ComfyUI Workflow Extractor v3 — unified single-script solution.

Replaces wf_extract_v2.py + wf_audit.py + wf_autopatch.py.

On first pass, builds a dynamic schema registry from all workflows being scanned:
  - Every node type gets its widget params labeled using unconnected input names
    where available, falling back to positional param_0, param_1...
  - Static PARAM_NAMES (for well-known types) always take priority
  - New types are discovered automatically — no manual patching needed ever

Also captures data that v2 dropped:
  - extra.info: author, created date, description (when present)
  - extra.frontendVersion: ComfyUI version the workflow was built on
  - extra.node_versions: exact pack commit hashes
  - Display Any / easy showAnything: the text they're displaying
  - Note / MarkdownNote: text content (as workflow documentation)
  - Custom node titles (title != type) captured as 'label' field
  - Bypassed nodes included with [BYPASSED] flag (were dropped in v2)
  - Muted nodes optionally reportable via --show-muted flag

Usage:
    python wf_extract_v3.py <folder> [output.txt] [options]

Options:
    --compact       Summary only (models, groups, metadata) — no node details
    --show-muted    Include muted nodes in output (flagged [MUTED])
    --unknown       Append list of types that fell back to positional params
    --no-notes      Don't capture Note/MarkdownNote text content
"""

import json, sys, re, os
from pathlib import Path
from collections import defaultdict

def is_comfyui_workflow(filepath: Path) -> bool:
    """Return True for UI-exported ComfyUI workflow JSON files."""
    try:
        with open(filepath, encoding='utf-8', errors='replace') as f:
            raw = f.read(10)
            if not raw.strip().startswith('{'):
                return False
        with open(filepath, encoding='utf-8', errors='replace') as f:
            data = json.load(f)
        return (
            isinstance(data, dict)
            and isinstance(data.get('nodes'), list)
            and isinstance(data.get('links'), list)
        )
    except Exception:
        return False

# ── Node types that are pure UI with no extractable data ─────────────────────
# Note/MarkdownNote handled separately (they have text content worth keeping)
SKIP_TYPES = {
    'PreviewImage', 'PreviewImageMaskOverlay', 'PreviewAudio', 'PreviewAny',
    'PreviewBridge', 'MaskPreview', 'MaskPreview+', 'Preview_Mask_Plus',
    'ImageAndMaskPreview', 'SEGSPreview', 'AILab_ImagePreview',
    'Image Comparer (rgthree)', 'Label (rgthree)',
    'VHS_PruneOutputs', 'PlaySound|pysssss', 'AddLabel',
    'Bookmark (rgthree)', 'UmeAiRT_Signature', 'VRAM_Debug',
    'easy imageCount',
}

# ── Types whose content is text worth capturing ───────────────────────────────
TEXT_CONTENT_TYPES = {
    'Note', 'MarkdownNote', 'NoteNode', 'ComfyNote', 'Note Plus (mtb)',
    'Display Any (rgthree)', 'easy showAnything', 'Show Text [Eclipse]',
    'ShowText|pysssss',
}

REROUTE_TYPES = {'Reroute', 'Reroute (rgthree)'}
MODEL_EXTS = {'.safetensors', '.ckpt', '.pt', '.gguf', '.pth', '.bin'}

# ── Static param names for well-known types (always takes priority) ───────────
STATIC_PARAMS = {
    'CheckpointLoaderSimple':       ['checkpoint'],
    'CheckpointLoader|pysssss':     ['checkpoint', 'config'],
    'UNETLoader':                   ['model_file', 'weight_dtype'],
    'UnetLoaderGGUF':               ['model_file'],
    'UnetLoaderGGUFAdvanced':       ['model_file', 'dequant_dtype', 'patch_dtype'],
    'VAELoader':                    ['vae_name'],
    'VAEDecodeTiled':               ['tile_size', 'overlap', 'temporal_size', 'temporal_overlap'],
    'VAEEncodeTiled':               ['tile_size', 'overlap', 'temporal_size', 'temporal_overlap'],
    'CLIPLoader':                   ['clip_name', 'type', 'device', 'dtype'],
    'DualCLIPLoader':               ['clip_name1', 'clip_name2', 'type'],
    'DualCLIPLoaderGGUF':           ['clip_name1', 'clip_name2'],
    'CLIPLoaderGGUF':               ['clip_name'],
    'CLIPSetLastLayer':             ['stop_at_clip_layer'],
    'CLIPAttentionMultiply':        ['q', 'k', 'v', 'out'],
    'UpscaleModelLoader':           ['model_name'],
    'IPAdapterModelLoader':         ['ipadapter_file'],
    'CLIPVisionLoader':             ['clip_name'],
    'ControlNetLoader':             ['control_net_name'],
    'SAMLoader':                    ['model_name', 'device_mode'],
    'DownloadAndLoadSAM2Model':     ['model', 'segmentor', 'device', 'precision'],
    'UltralyticsDetectorProvider':  ['model_path'],
    'CLIPTextEncode':               ['text'],
    'CLIPTextEncodeFlux':           ['clip_l', 't5xxl', 'guidance'],
    'CLIPTextEncodeSDXL':           ['width', 'height', 'crop_w', 'crop_h', 'target_width', 'target_height', 'text_g', 'text_l'],
    'BNK_CLIPTextEncodeAdvanced':   ['text', 'token_normalization', 'weight_interpretation', 'balance'],
    'CLIPSetLastLayer':             ['stop_at_clip_layer'],
    'KSampler':                     ['seed', 'seed_mode', 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise'],
    'KSamplerAdvanced':             ['add_noise', 'noise_seed', 'seed_mode', 'steps', 'cfg', 'sampler_name', 'scheduler', 'start_at_step', 'end_at_step', 'return_leftover'],
    'KSamplerSelect':               ['sampler_name'],
    'BasicScheduler':               ['scheduler', 'steps', 'denoise'],
    'BasicGuider':                  [],
    'CFGGuider':                    ['cfg'],
    'RandomNoise':                  ['noise_seed', 'seed_mode'],
    'EmptyLatentImage':             ['width', 'height', 'batch_size'],
    'EmptySD3LatentImage':          ['width', 'height', 'batch_size'],
    'LatentUpscale':                ['upscale_method', 'width', 'height', 'crop'],
    'LatentUpscaleBy':              ['upscale_method', 'scale_by'],
    'FluxGuidance':                 ['guidance'],
    'ModelSamplingFlux':            ['max_shift', 'base_shift', 'width', 'height'],
    'ModelSamplingSD3':             ['shift'],
    'ModelSamplingAuraFlow':        ['shift'],
    'CFGNorm':                      ['scale'],
    'LyingSigmaSampler':            ['lying_sigma_multiplier', 'start_percent', 'end_percent'],
    'DetailDaemonSamplerNode':      ['detail_amount', 'start', 'end', 'bias', 'exponent', 'start_offset', 'end_offset', 'fade', 'smooth'],
    'LoraLoader':                   ['lora_name', 'strength_model', 'strength_clip'],
    'LoraLoader|pysssss':           ['lora_name', 'strength_model', 'strength_clip'],
    'LoraLoaderModelOnly':          ['lora_name', 'strength_model'],
    'NunchakuFluxDiTLoader':        ['model_path', 'cache_threshold', 'num_persistent_param_in_dit'],
    'NunchakuFluxLoraLoader':       ['lora_path', 'lora_strength'],
    'NunchakuTextEncoderLoaderV2':  ['text_encoder_path', 'dtype'],
    'NunchakuFluxPuLIDApplyV2':     ['weight', 'start_at', 'end_at'],
    'LoadImage':                    ['image', 'upload_type'],
    'SaveImage':                    ['filename_prefix'],
    'ImageScale':                   ['upscale_method', 'width', 'height', 'crop'],
    'ImageScaleBy':                 ['upscale_method', 'scale_by'],
    'ImageScaleToTotalPixels':      ['upscale_method', 'megapixels'],
    'ImageUpscaleWithModel':        [],
    'UltimateSDUpscale':            ['upscale_by', 'seed', 'seed_mode', 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise', 'mode_type', 'tile_width', 'tile_height', 'mask_blur'],
    'UltimateSDUpscaleCustomSample':['upscale_by', 'seed', 'seed_mode', 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise', 'mode_type', 'tile_width', 'tile_height', 'mask_blur'],
    'SeedVR2LoadDiTModel':          ['model_name', 'precision'],
    'SeedVR2LoadVAEModel':          ['model_name'],
    'SeedVR2VideoUpscaler':         ['seed', 'steps', 'cfg', 'tile_size', 'tile_overlap', 'scale_factor', 'enable_tiling'],
    'FaceDetailer':                 ['guide_size', 'guide_size_for', 'max_size', 'seed', 'seed_mode', 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise', 'feather', 'noise_mask', 'force_inpaint', 'bbox_threshold', 'bbox_dilation', 'bbox_crop_factor', 'sam_detection_hint', 'sam_dilation', 'sam_threshold', 'sam_bbox_expansion', 'sam_mask_hint_threshold', 'sam_mask_hint_use_negative', 'drop_size', 'wildcard', 'cycle'],
    'BboxDetectorSEGS':             ['threshold', 'dilation', 'crop_factor', 'drop_size', 'labels'],
    'SAMDetectorSegmented':         ['detection_hint', 'dilation', 'threshold', 'bbox_expansion', 'mask_hint_threshold', 'mask_hint_use_negative'],
    'ControlNetApplyAdvanced':      ['strength', 'start_percent', 'end_percent'],
    'AIO_Preprocessor':             ['preprocessor', 'resolution'],
    'DepthAnythingV2Preprocessor':  ['resolution'],
    'OpenposePreprocessor':         ['detect_hand', 'detect_body', 'detect_face', 'resolution'],
    'Any Switch (rgthree)':         [],
    'Context Switch Big (rgthree)': [],
    'Context Big (rgthree)':        [],
    'Context (rgthree)':            [],
    'GetNode':                      ['channel_name'],
    'SetNode':                      ['channel_name'],
    'easy getNode':                 ['channel_name'],
    'easy setNode':                 ['channel_name'],
    'Seed (rgthree)':               ['seed'],
    'mxSlider':                     ['min', 'value', 'max'],
    'mxSlider2D':                   ['Xi', 'Xf', 'Yi', 'Yf', 'isfloatX', 'isfloatY'],
    'mxStop':                       ['active'],
    'DownloadAndLoadFlorence2Model':['model_name', 'precision', 'attention'],
    'Florence2Run':                 ['text_input', 'task', 'fill_mask', 'keep_model_loaded', 'max_new_tokens', 'num_beams', 'do_sample', 'seed', 'seed_mode'],
    'VHS_VideoCombine':             ['frame_rate', 'loop_count', 'filename_prefix', 'format', 'pingpong', 'save_output', 'save_metadata'],
    'VHS_LoadVideo':                ['video', 'force_rate', 'force_size', 'custom_width', 'custom_height', 'frame_load_cap', 'skip_first_frames', 'select_every_nth'],
    'CR Prompt List':               ['prompt_1', 'prompt_2', 'prompt_3', 'prompt_4', 'prompt_5'],
    'ImpactWildcardProcessor':      ['populated_text', 'mode', 'seed', 'seed_mode', 'Select Wildcard'],
    'Text Prompt (JPS)':            ['text'],
    'easy positive':                ['text'],
    'easy negative':                ['text'],
    'easy seed':                    ['seed', 'seed_mode'],
    'easy int':                     ['value'],
    'easy float':                   ['value'],
    'easy string':                  ['value'],
    'easy boolean':                 ['value'],
    'easy hiresFix':                ['upscale_model', 'rescale_after_model', 'rescale_method', 'rescale_type', 'percent', 'width', 'height', 'longer_side'],
    'PrimitiveNode':                ['value', 'seed_mode'],
    'PrimitiveBoolean':             ['boolean'],
    'PrimitiveInt':                 ['int'],
    'PrimitiveFloat':               ['float'],
    'PrimitiveString':              ['string'],
    'PrimitiveStringMultiline':     ['string'],
    'Cfg Literal':                  ['float'],
    'Int Literal':                  ['int'],
    'Float':                        ['value'],
    'Int':                          ['value'],
    'String':                       ['value'],
    'String Literal':               ['string'],
    'INTConstant':                  ['value'],
    'FloatConstant':                ['value'],
    'Power Primitive (rgthree)':    ['type', 'value'],
    'StyleModelLoader':             ['style_model_name'],
    'StyleModelApplyAdvanced':      ['strength'],
    'CLIPVisionEncode':             ['crop'],
    'UmeAiRT_BundleLoader':         ['model_preset', 'dtype'],
    'UmeAiRT_GenerationSettings':   ['width', 'height', 'steps', 'cfg', 'sampler', 'scheduler', 'seed', 'seed_mode'],
    'UmeAiRT_ImageProcess_Inpaint': ['denoise', 'padding', 'use_mask'],
    'UmeAiRT_PipelineSeedVR2Upscale':['enabled', 'model_name', 'scale_factor', 'tile_width', 'tile_height', 'seed', 'steps', 'max_megapixels', 'mode_type', 'noise_augmentation', 'color_match', 'color_space'],
}

NOISE_RE = re.compile(r'^clipspace/|^/api/view\?|_temp_|\[input\]$|^example\.')

def is_noise(v):
    return isinstance(v, str) and bool(NOISE_RE.search(v))

def is_model_file(v):
    if not isinstance(v, str) or not v.strip(): return False
    vl = v.lower()
    return any(vl.endswith(e) for e in MODEL_EXTS)

def safe_widgets(raw):
    """Normalise widgets_values to a flat list regardless of format."""
    if isinstance(raw, dict):
        return list(raw.values())
    if isinstance(raw, list):
        return raw
    return []

def clean_widget_value(v):
    """Remove runtime noise, decode Power LoRA dicts."""
    if isinstance(v, dict):
        if v.get('type') == 'PowerLoraLoaderHeaderWidget' or v == {}:
            return None
        if 'lora' in v and 'strength' in v:
            lora = v.get('lora') or 'None'
            if lora not in ('None', None, ''):
                return {'lora': lora, 'strength': v.get('strength'), 'active': v.get('on', True)}
            return None
        return v
    if isinstance(v, list):
        if all(isinstance(x, dict) and 'url' in x for x in v):
            return None
        return v
    if is_noise(v):
        return None
    return v

# ── Dynamic schema registry ───────────────────────────────────────────────────

class SchemaRegistry:
    """
    Builds widget param name mappings dynamically from workflow data.
    For each node type, collects the most common (widget_count, unconnected_input_names)
    across all workflows, then uses that to label widget values.
    """
    def __init__(self):
        # type -> list of (widget_count, [unconnected_input_names])
        self._samples = defaultdict(list)
        # type -> chosen (widget_count, [names]) after finalization
        self._schema = {}

    def observe(self, node):
        """Record a node's widget/input pattern during the discovery pass."""
        ntype = node.get('type', '')
        if ntype in STATIC_PARAMS or ntype in SKIP_TYPES or ntype in REROUTE_TYPES:
            return
        if len(ntype) > 30 and '-' in ntype:  # UUID subgraph
            return

        raw_wv = safe_widgets(node.get('widgets_values', []))
        # Clean to get actual widget count
        cleaned = [v for v in (clean_widget_value(v) for v in raw_wv) if v is not None]
        wcount = len(cleaned)
        if wcount == 0:
            self._samples[ntype].append((0, []))
            return

        inputs = node.get('inputs', [])
        # Unconnected inputs (no link, not optional shape=7 required-but-missing)
        # In ComfyUI, widget slots often appear as inputs with link=None
        # Use their names as param name hints
        unconnected = [
            i.get('name', '') for i in inputs
            if i.get('link') is None and i.get('name')
        ]
        self._samples[ntype].append((wcount, unconnected))

    def finalize(self):
        """Choose the most representative schema for each type."""
        for ntype, samples in self._samples.items():
            if not samples:
                continue
            # Group by widget count, pick most common count
            count_groups = defaultdict(list)
            for wc, names in samples:
                count_groups[wc].append(names)
            # Most frequent widget count
            best_count = max(count_groups, key=lambda c: len(count_groups[c]))
            # Among samples with that count, pick the one with the most named inputs
            name_candidates = count_groups[best_count]
            best_names = max(name_candidates, key=len, default=[])
            self._schema[ntype] = (best_count, best_names)

    def label_params(self, ntype, cleaned_widgets):
        """Return a dict of {param_name: value} for a node's widgets."""
        if not cleaned_widgets:
            return {}

        # Static mapping takes priority
        if ntype in STATIC_PARAMS:
            names = STATIC_PARAMS[ntype]
            result = {}
            for i, v in enumerate(cleaned_widgets):
                k = names[i] if i < len(names) else f'param_{i}'
                if v not in (None, '', [], {}):
                    result[k] = v
            return result

        # Dynamic schema
        if ntype in self._schema:
            _, names = self._schema[ntype]
            result = {}
            for i, v in enumerate(cleaned_widgets):
                if v in (None, '', [], {}):
                    continue
                # Use unconnected input name if count roughly aligns
                # Heuristic: if we have N names and N widgets, zip them
                # Otherwise fall back to positional
                if i < len(names) and len(names) == len(cleaned_widgets):
                    k = names[i]
                else:
                    k = names[i] if i < len(names) else f'param_{i}'
                result[k] = v
            return result

        # Unknown — pure positional
        return {f'param_{i}': v for i, v in enumerate(cleaned_widgets)
                if v not in (None, '', [], {})}

    def is_dynamic(self, ntype):
        return ntype not in STATIC_PARAMS and ntype in self._schema

    def is_unknown(self, ntype):
        return ntype not in STATIC_PARAMS and ntype not in self._schema


# ── Connection resolution ─────────────────────────────────────────────────────

def build_link_map(links_raw):
    lm = {}
    for lk in links_raw:
        if len(lk) >= 6:
            lm[lk[0]] = {'from': lk[1], 'from_out': lk[2],
                          'to': lk[3], 'to_in': lk[4], 'type': lk[5]}
    return lm

def make_reroute_resolver(nodes_raw, link_map):
    id_to_node = {n['id']: n for n in nodes_raw}

    def follow_src(nid, oidx):
        nd = id_to_node.get(nid)
        if not nd or nd['type'] not in REROUTE_TYPES:
            return nid, oidx
        for inp in nd.get('inputs', []):
            lid = inp.get('link')
            if lid and lid in link_map:
                lk = link_map[lid]
                return follow_src(lk['from'], lk['from_out'])
        return nid, oidx

    def follow_dst(nid, iidx):
        nd = id_to_node.get(nid)
        if not nd or nd['type'] not in REROUTE_TYPES:
            return nid, iidx
        for out in nd.get('outputs', []):
            for lid in (out.get('links') or []):
                if lid in link_map:
                    lk = link_map[lid]
                    return follow_dst(lk['to'], lk['to_in'])
        return nid, iidx

    return follow_src, follow_dst

def get_node_group(pos, groups_raw):
    if isinstance(pos, list) and len(pos) >= 2:
        nx, ny = pos[0], pos[1]
    elif isinstance(pos, dict):
        nx, ny = pos.get('0', 0), pos.get('1', 0)
    else:
        return None
    for g in groups_raw:
        bx, by, bw, bh = g['bounding']
        if bx <= nx <= bx + bw and by <= ny <= by + bh:
            return g.get('title') or None
    return None


# ── Subgraph handling ─────────────────────────────────────────────────────────

def expand_subgraph(outer_node, sg_def, outer_group, registry, id_to_type, link_map, follow_src, follow_dst):
    sg_name = sg_def.get('name', 'subgraph')
    group_label = f'{outer_group} > {sg_name}' if outer_group else sg_name

    inner_link_map = {}
    for lk in sg_def.get('links', []):
        inner_link_map[lk['id']] = {
            'from': lk['origin_id'], 'from_out': lk['origin_slot'],
            'to': lk['target_id'], 'to_in': lk['target_slot'],
            'type': lk.get('type', '*'),
        }

    inner_id_to_type = {n['id']: n['type'] for n in sg_def.get('nodes', [])}

    sg_inputs  = sg_def.get('inputs', [])
    sg_outputs = sg_def.get('outputs', [])
    outer_inputs  = outer_node.get('inputs', [])
    outer_outputs = outer_node.get('outputs', [])

    # Boundary input link_id -> real source
    boundary_in = {}
    for i, bi in enumerate(sg_inputs):
        if i < len(outer_inputs):
            lid = outer_inputs[i].get('link')
            if lid and lid in link_map:
                lk = link_map[lid]
                real_src, _ = follow_src(lk['from'], lk['from_out'])
                real_type = id_to_type.get(real_src, '?')
                for blid in (bi.get('linkIds') or []):
                    boundary_in[blid] = {'from_type': real_type, 'from_id': real_src, 'data': lk['type']}

    # Boundary output link_id -> real destinations
    boundary_out = {}
    for i, bo in enumerate(sg_outputs):
        if i < len(outer_outputs):
            for lid in (outer_outputs[i].get('links') or []):
                if lid in link_map:
                    lk = link_map[lid]
                    real_dst, _ = follow_dst(lk['to'], lk['to_in'])
                    real_type = id_to_type.get(real_dst, '?')
                    for blid in (bo.get('linkIds') or []):
                        boundary_out.setdefault(blid, []).append(
                            {'to_type': real_type, 'to_id': real_dst, 'data': lk['type']})

    entries = []
    models = set()

    for inn in sg_def.get('nodes', []):
        itype = inn.get('type', '?')
        imode = inn.get('mode', 0)
        if itype in SKIP_TYPES or itype in REROUTE_TYPES:
            continue
        if imode == 4:
            continue

        raw_wv = safe_widgets(inn.get('widgets_values', []))
        cleaned = [x for x in (clean_widget_value(v) for v in raw_wv) if x is not None]
        params  = registry.label_params(itype, cleaned)

        for v in params.values():
            if isinstance(v, str) and is_model_file(v):
                models.add(v)

        title = inn.get('title', '')
        label = title if title and title != itype else None

        conns_in, conns_out = [], []
        for inp in inn.get('inputs', []):
            lid = inp.get('link')
            if lid is None: continue
            if lid in boundary_in:
                bi = boundary_in[lid]
                if bi['from_type'] not in SKIP_TYPES:
                    conns_in.append({'name': inp.get('name','?'), 'from_type': bi['from_type'],
                                     'from_id': bi['from_id'], 'data': bi['data']})
            elif lid in inner_link_map:
                ilk = inner_link_map[lid]
                if ilk['from'] != -10:
                    st = inner_id_to_type.get(ilk['from'], '?')
                    if st not in SKIP_TYPES and st not in REROUTE_TYPES:
                        conns_in.append({'name': inp.get('name','?'), 'from_type': st,
                                         'from_id': ilk['from'], 'data': ilk['type']})

        for out in inn.get('outputs', []):
            for lid in (out.get('links') or []):
                if lid in boundary_out:
                    for bo in boundary_out[lid]:
                        if bo['to_type'] not in SKIP_TYPES:
                            conns_out.append({'name': out.get('name','?'), 'to_type': bo['to_type'],
                                              'to_id': bo['to_id'], 'data': bo['data']})
                elif lid in inner_link_map:
                    ilk = inner_link_map[lid]
                    if ilk['to'] != -20:
                        dt = inner_id_to_type.get(ilk['to'], '?')
                        if dt not in SKIP_TYPES and dt not in REROUTE_TYPES:
                            conns_out.append({'name': out.get('name','?'), 'to_type': dt,
                                              'to_id': ilk['to'], 'data': ilk['type']})

        entries.append({
            'id': inn['id'], 'type': itype, 'label': label,
            'group': group_label, 'bypassed': imode == 2, 'muted': False,
            'params': params, 'in': conns_in, 'out': conns_out,
            'is_text_node': itype in TEXT_CONTENT_TYPES,
        })

    return entries, models


# ── Main extract function ─────────────────────────────────────────────────────

def extract(filepath, registry, show_muted=False, include_notes=True):
    with open(filepath, encoding='utf-8', errors='replace') as f:
        try:
            data = json.load(f)
        except Exception as e:
            return {'error': str(e)}

    nodes_raw  = data.get('nodes', [])
    links_raw  = data.get('links', [])
    groups_raw = data.get('groups', [])
    extra      = data.get('extra', {}) if isinstance(data.get('extra'), dict) else {}

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta = {}
    info = extra.get('info')
    if isinstance(info, dict):
        for k in ('name', 'author', 'description', 'version', 'created'):
            if info.get(k):
                meta[k] = info[k]
    fv = extra.get('frontendVersion')
    if fv:
        meta['frontend_version'] = fv
    nv = extra.get('node_versions')
    if isinstance(nv, dict) and nv:
        meta['node_versions'] = nv

    # Date extraction from output paths baked into widget values
    date_re = re.compile(r'20[0-9]{2}[-/][0-9]{2}[-/][0-9]{2}')
    dates = set()
    for n in nodes_raw:
        for v in safe_widgets(n.get('widgets_values', [])):
            if isinstance(v, str):
                dates.update(date_re.findall(v))
    if dates:
        meta['last_run_date'] = max(dates)

    # ── Subgraph definitions ──────────────────────────────────────────────────
    sub_defs = {}
    for sg in data.get('definitions', {}).get('subgraphs', []):
        sub_defs[sg['id']] = sg

    link_map = build_link_map(links_raw)
    id_to_type = {n['id']: n['type'] for n in nodes_raw}
    follow_src, follow_dst = make_reroute_resolver(nodes_raw, link_map)

    def outer_conns(node):
        ci, co = [], []
        for inp in node.get('inputs', []):
            lid = inp.get('link')
            if lid and lid in link_map:
                lk = link_map[lid]
                rs, _ = follow_src(lk['from'], lk['from_out'])
                rt = id_to_type.get(rs, '?')
                if rt not in SKIP_TYPES:
                    ci.append({'name': inp['name'], 'from_type': rt, 'from_id': rs, 'data': lk['type']})
        for out in node.get('outputs', []):
            for lid in (out.get('links') or []):
                if lid in link_map:
                    lk = link_map[lid]
                    rd, _ = follow_dst(lk['to'], lk['to_in'])
                    rt = id_to_type.get(rd, '?')
                    if rt not in SKIP_TYPES:
                        co.append({'name': out['name'], 'to_type': rt, 'to_id': rd, 'data': lk['type']})
        return ci, co

    models = set()
    nodes_out = []
    sg_inner_counts = {}
    unknown_types = set()

    for n in nodes_raw:
        ntype = n['type']
        mode  = n.get('mode', 0)

        # Pure skip
        if ntype in SKIP_TYPES:
            continue
        if ntype in REROUTE_TYPES:
            continue
        if mode == 4 and not show_muted:
            continue

        pos = n.get('pos', [0, 0])
        group = get_node_group(pos, groups_raw)
        title = n.get('title', '')
        label = title if title and title != ntype else None

        # ── Subgraph UUID ─────────────────────────────────────────────────────
        is_sub = len(ntype) > 30 and '-' in ntype and ntype in sub_defs
        if is_sub:
            sg_def = sub_defs[ntype]
            sg_name = sg_def.get('name', 'subgraph')

            # Count inner types for summary
            for inn in sg_def.get('nodes', []):
                it = inn.get('type', '?')
                if it not in SKIP_TYPES and it not in REROUTE_TYPES and not (len(it) > 30 and '-' in it):
                    sg_inner_counts[it] = sg_inner_counts.get(it, 0) + 1

            # Collect models from inner nodes
            for inn in sg_def.get('nodes', []):
                for v in safe_widgets(inn.get('widgets_values', [])):
                    if isinstance(v, str) and is_model_file(v):
                        models.add(v)

            inner_types = list(dict.fromkeys(
                n2['type'] for n2 in sg_def.get('nodes', [])
                if n2['type'] not in SKIP_TYPES and n2['type'] not in REROUTE_TYPES
                and not (len(n2['type']) > 30 and '-' in n2['type'])
            ))
            ci, co = outer_conns(n)
            nodes_out.append({
                'id': n['id'], 'type': f'[SUBGRAPH: {sg_name}]',
                'label': label, 'group': group,
                'bypassed': mode == 2, 'muted': mode == 4,
                'params': {}, 'in': ci, 'out': co,
                'inner_types': inner_types[:12],
                'is_text_node': False,
            })
            continue

        # ── Text content nodes ────────────────────────────────────────────────
        if ntype in TEXT_CONTENT_TYPES:
            if not include_notes:
                continue
            raw_wv = safe_widgets(n.get('widgets_values', []))
            # Grab first non-empty string widget as the text content
            text_content = next(
                (v for v in raw_wv if isinstance(v, str) and v.strip()), None
            )
            if not text_content:
                continue
            ci, co = outer_conns(n)
            nodes_out.append({
                'id': n['id'], 'type': ntype, 'label': label, 'group': group,
                'bypassed': mode == 2, 'muted': mode == 4,
                'params': {'text': text_content[:500]},  # cap at 500 chars
                'in': ci, 'out': co, 'is_text_node': True,
            })
            continue

        # ── Regular node ──────────────────────────────────────────────────────
        raw_wv  = safe_widgets(n.get('widgets_values', []))
        cleaned = [x for x in (clean_widget_value(v) for v in raw_wv) if x is not None]
        params  = registry.label_params(ntype, cleaned)

        for v in params.values():
            if isinstance(v, str) and is_model_file(v):
                models.add(v)

        if registry.is_unknown(ntype) and cleaned:
            unknown_types.add(ntype)

        ci, co = outer_conns(n)

        nodes_out.append({
            'id': n['id'], 'type': ntype, 'label': label, 'group': group,
            'bypassed': mode == 2, 'muted': mode == 4,
            'params': params, 'in': ci, 'out': co, 'is_text_node': False,
        })

    return {
        'node_count':       len(nodes_out),
        'groups':           [g.get('title','') for g in groups_raw if g.get('title')],
        'models':           sorted(models),
        'meta':             meta,
        'nodes':            nodes_out,
        'sg_inner_counts':  sg_inner_counts,
        'unknown_types':    unknown_types,
        'path':             '',
    }


# ── Rendering ─────────────────────────────────────────────────────────────────

def render(name, data, compact=False):
    L = []
    sep = '=' * 66
    L.append(sep)
    L.append(f'WORKFLOW: {name}')
    L.append(sep)

    if 'error' in data:
        L.append(f'ERROR: {data["error"]}')
        return '\n'.join(L)

    if data.get('path'):
        L.append(f'Path: {data["path"]}')

    meta = data.get('meta', {})
    if meta.get('author'):      L.append(f'Author: {meta["author"]}')
    if meta.get('created'):     L.append(f'Created: {meta["created"]}')
    if meta.get('name'):        L.append(f'Name: {meta["name"]}')
    if meta.get('description'): L.append(f'Description: {meta["description"][:200]}')
    if meta.get('frontend_version'): L.append(f'ComfyUI frontend: {meta["frontend_version"]}')
    if meta.get('last_run_date'):    L.append(f'Last run date: {meta["last_run_date"]}')
    if meta.get('node_versions'):
        packs = list(meta['node_versions'].keys())
        L.append(f'Node packs ({len(packs)}): {", ".join(packs[:8])}{"..." if len(packs) > 8 else ""}')

    L.append(f'Nodes: {data["node_count"]}')
    if data['groups']:
        L.append(f'Sections: {" | ".join(data["groups"])}')
    if data['models']:
        L.append('Models:')
        for m in data['models']:
            L.append(f'  {m}')

    if compact:
        return '\n'.join(L)

    L.append('\n--- NODES ---')
    for nd in data['nodes']:
        flags = ''
        if nd.get('bypassed'): flags += ' [BYPASSED]'
        if nd.get('muted'):    flags += ' [MUTED]'
        grp  = f' [{nd["group"]}]' if nd.get('group') else ''
        lbl  = f' "{nd["label"]}"' if nd.get('label') else ''
        line = f'\n[{nd["id"]}] {nd["type"]}{lbl}{grp}{flags}'

        inner = nd.get('inner_types', [])
        if inner:
            line += f'\n  contains: {", ".join(inner)}'
        L.append(line)

        if nd.get('is_text_node'):
            text = nd['params'].get('text', '')
            L.append(f'  text: {text[:300]}{"..." if len(text) > 300 else ""}')
        else:
            for k, v in nd.get('params', {}).items():
                if v not in (None, '', [], {}):
                    L.append(f'  param  {k}: {v}')

        for c in nd.get('in', []):
            L.append(f'  in     {c["name"]} <- {c["from_type"]} (id:{c["from_id"]}) [{c["data"]}]')
        for c in nd.get('out', []):
            L.append(f'  out    {c["name"]} -> {c["to_type"]} (id:{c["to_id"]}) [{c["data"]}]')

    return '\n'.join(L)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='ComfyUI Workflow Extractor v3')
    parser.add_argument('target',   help='File or folder of workflow JSONs')
    parser.add_argument('output',   nargs='?', default='workflows_combined.txt')
    parser.add_argument('--compact',    action='store_true', help='Summary only, no node details')
    parser.add_argument('--show-muted', action='store_true', help='Include muted nodes (flagged [MUTED])')
    parser.add_argument('--unknown',    action='store_true', help='Report types that fell back to positional params')
    parser.add_argument('--no-notes',   action='store_true', help='Exclude Note/Display Any text content')
    args = parser.parse_args()

    target = Path(args.target)
    files  = sorted(target.glob('**/*.json')) if target.is_dir() else [target]
    files  = [f for f in files if f.stat().st_size > 500 and is_comfyui_workflow(f)]

    print(f'Pass 1/2 — building schema registry from {len(files)} workflows...')
    registry = SchemaRegistry()
    for fp in files:
        try:
            with open(fp, encoding='utf-8', errors='replace') as f:
                data = json.load(f)
            for n in data.get('nodes', []):
                registry.observe(n)
            # Also observe inner subgraph nodes
            for sg in data.get('definitions', {}).get('subgraphs', []):
                for n in sg.get('nodes', []):
                    registry.observe(n)
        except Exception:
            pass
    registry.finalize()
    print(f'  Registry: {len(STATIC_PARAMS)} static + {len(registry._schema)} dynamic schemas')

    print(f'Pass 2/2 — extracting...')
    all_sections      = []
    type_counts       = defaultdict(int)
    sg_type_counts    = {}
    model_counts      = defaultdict(int)
    all_unknown_types = set()

    for fp in files:
        try:
            rel = str(fp.relative_to(target)) if target.is_dir() else fp.name
        except Exception:
            rel = fp.name

        data = extract(fp, registry,
                       show_muted=args.show_muted,
                       include_notes=not args.no_notes)
        data['path'] = rel
        all_sections.append(render(fp.name, data, compact=args.compact))

        for nd in data.get('nodes', []):
            t = nd['type']
            type_counts[t] += 1
        for itype, cnt in data.get('sg_inner_counts', {}).items():
            type_counts[itype] = type_counts.get(itype, 0) + cnt
            sg_type_counts[itype] = sg_type_counts.get(itype, 0) + cnt
        for m in data.get('models', []):
            model_counts[m] += 1
        all_unknown_types.update(data.get('unknown_types', set()))

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = [
        '=' * 66,
        'COMBINED WORKFLOW ANALYSIS',
        '=' * 66,
        f'Workflows: {len(files)}',
        f'Static param mappings: {len(STATIC_PARAMS)}',
        f'Dynamic param mappings: {len(registry._schema)}',
        f'Mode: {"compact" if args.compact else "full"}',
        '',
        'Node type frequency:',
    ]
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        if t.startswith('[SUBGRAPH:'):
            continue
        summary.append(f'  {c:4d}x  {t}')

    if model_counts:
        summary.append('\nModel files:')
        def norm(p):
            return re.sub(r'[/\\]+', '/', p).strip('/')
        by_name = defaultdict(int)
        for m, c in model_counts.items():
            base = norm(m).rsplit('/', 1)[-1]
            by_name[base] += c

        for base, total in sorted(by_name.items(), key=lambda x: -x[1]):
            cnt = f' ({total}x)' if total > 1 else ''
            summary.append(f'  {base}{cnt}')

    if args.unknown and all_unknown_types:
        summary.append(f'\nTypes with positional fallback ({len(all_unknown_types)}):')
        for t in sorted(all_unknown_types):
            summary.append(f'  {t}')

    full = '\n'.join(summary) + '\n\n' + '\n\n'.join(all_sections)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(full)

    orig = sum(fp.stat().st_size for fp in files)
    out  = Path(args.output).stat().st_size
    pct  = 100 * out // orig if orig else 0
    print(f'\nDone -> {args.output}')
    print(f'{orig//1024}KB in -> {out//1024}KB out ({pct}% of original)')
    if all_unknown_types:
        print(f'Note: {len(all_unknown_types)} types used positional fallback (run --unknown to list)')

if __name__ == '__main__':
    main()
