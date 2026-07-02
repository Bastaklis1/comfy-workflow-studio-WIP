"""
Deduplication logic for ComfyUI Workflow Studio.
Deterministic graph-aware fingerprinting and cluster detection.
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


# These are populated at import time by wfs4.py's extractor loader,
# but we need safe fallbacks in case this module is imported standalone.
try:
    from wf_extractor import SKIP_TYPES, REROUTE_TYPES  # type: ignore
except Exception:
    SKIP_TYPES    = set()
    REROUTE_TYPES = set()

DEDUPE_IGNORE_TYPES = SKIP_TYPES | REROUTE_TYPES | {
    'Note', 'MarkdownNote', 'NoteNode', 'Display Any (rgthree)',
    'GetNode', 'SetNode', 'easy getNode', 'easy setNode',
    'Any Switch (rgthree)', 'Fast Muter (rgthree)', 'Fast Bypasser (rgthree)',
    'Mute / Bypass Repeater (rgthree)', 'Seed (rgthree)',
    'SaveImage', 'PreviewImage',
}


def _stable_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.md5(raw.encode('utf-8')).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _model_basenames(models: list) -> list[str]:
    out = []
    for m in models or []:
        norm = re.sub(r'[/\\]+', '/', str(m)).strip('/')
        base = norm.rsplit('/', 1)[-1] if norm else ''
        if base:
            out.append(base.lower())
    return sorted(out)


def _counter_jaccard(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return inter / union if union else 0.0


def _dedupe_profile(rec: dict) -> dict:
    """Build cached signatures for duplicate and near-duplicate review."""
    cached = rec.get('_dedupe_profile')
    if cached:
        return cached

    wf_data = rec.get('wf_data', {}) or {}
    type_counts   = defaultdict(int)
    degree_counts = defaultdict(int)
    edge_counts   = defaultdict(int)

    for nd in wf_data.get('nodes', []):
        ntype = nd.get('type', '?')
        if ntype in DEDUPE_IGNORE_TYPES or ntype.startswith('[SUBGRAPH:'):
            continue
        type_counts[ntype] += 1
        in_deg  = len(nd.get('in',  []) or [])
        out_deg = len(nd.get('out', []) or [])
        degree_counts[(ntype, in_deg, out_deg)] += 1
        for conn in nd.get('out', []) or []:
            to_type = conn.get('to_type') or '?'
            if to_type in DEDUPE_IGNORE_TYPES or str(to_type).startswith('[SUBGRAPH:'):
                continue
            data = conn.get('data') or ''
            if isinstance(data, (list, dict)):
                data = json.dumps(data, sort_keys=True, ensure_ascii=False)
            edge_counts[(str(ntype), str(conn.get('name') or ''), str(to_type), str(data))] += 1

    # Inner subgraph nodes are known as counts, even when inner topology is not
    # available in the flattened GUI record. Include them so subgraph-heavy
    # workflows do not collapse into only their outer wrapper node.
    for itype, cnt in (wf_data.get('sg_inner_counts') or {}).items():
        if itype not in DEDUPE_IGNORE_TYPES:
            type_counts[itype] += cnt

    model_counts = defaultdict(int)
    for base in _model_basenames(rec.get('models', [])):
        model_counts[base] += 1

    type_items   = sorted(type_counts.items())
    degree_items = sorted((f'{t}|{i}|{o}', c) for (t, i, o), c in degree_counts.items())
    edge_items   = sorted((f'{src}|{slot}|{dst}|{data}', c)
                          for (src, slot, dst, data), c in edge_counts.items())
    model_items  = sorted(model_counts.items())

    profile = {
        'types':           dict(type_counts),
        'degrees':         dict(degree_counts),
        'edges':           dict(edge_counts),
        'models':          dict(model_counts),
        'meaningful_count': sum(type_counts.values()),
        'type_hash':       _stable_hash(type_items),
        'degree_hash':     _stable_hash(degree_items),
        'edge_hash':       _stable_hash(edge_items),
        'model_hash':      _stable_hash(model_items),
    }
    profile['structure_hash'] = _stable_hash([
        profile['type_hash'],
        profile['degree_hash'],
        profile['edge_hash'],
        profile['model_hash'],
    ])
    rec['_dedupe_profile'] = profile
    return profile


def fingerprint(record_or_node_types) -> str:
    """Compatibility helper: graph-aware for records, node-set for old callers."""
    if isinstance(record_or_node_types, dict):
        return _dedupe_profile(record_or_node_types)['structure_hash']
    meaningful = sorted({
        t for t in record_or_node_types
        if t not in DEDUPE_IGNORE_TYPES and not str(t).startswith('[SUBGRAPH:')
    })
    return _stable_hash(meaningful)


def similarity(types_a: set, types_b: set) -> float:
    """Legacy node-toolset Jaccard similarity used as one review signal."""
    if not types_a or not types_b:
        return 0.0
    return len(types_a & types_b) / len(types_a | types_b)


def compare_records(a: dict, b: dict) -> dict:
    """Return a multi-signal similarity profile for two workflow records."""
    pa = _dedupe_profile(a)
    pb = _dedupe_profile(b)
    if a.get('file_hash') and a.get('file_hash') == b.get('file_hash'):
        return {
            'score': 1.0, 'reason': 'exact file',
            'toolset': 1.0, 'structure': 1.0, 'degree': 1.0, 'models': 1.0,
        }
    structure_exact = (
        pa['structure_hash'] == pb['structure_hash']
        and pa.get('meaningful_count', 0) >= 4
        and pb.get('meaningful_count', 0) >= 4
    )
    edge    = _counter_jaccard(pa['edges'],   pb['edges'])
    degree  = _counter_jaccard(pa['degrees'], pb['degrees'])
    toolset = _counter_jaccard(pa['types'],   pb['types'])
    models  = _counter_jaccard(pa['models'],  pb['models'])
    score   = (edge * 0.45) + (degree * 0.25) + (toolset * 0.20) + (models * 0.10)
    if min(pa.get('meaningful_count', 0), pb.get('meaningful_count', 0)) < 4:
        score = min(score, 0.70)
    if structure_exact:
        score = max(score, 0.99)
    if structure_exact:
        reason = 'same structure'
    elif min(pa.get('meaningful_count', 0), pb.get('meaningful_count', 0)) < 4:
        reason = 'tiny workflow'
    elif edge >= 0.95 and degree >= 0.90:
        reason = 'near structure'
    elif score >= 0.85:
        reason = 'strong variant'
    else:
        reason = 'similar toolbox'
    return {
        'score': score, 'reason': reason,
        'toolset': toolset, 'structure': edge, 'degree': degree, 'models': models,
    }


def cluster_duplicates(records: list, threshold: float = 0.85) -> dict:
    """
    Group workflows by exact file match, graph-aware structure match, or a
    combined near-duplicate score. The old node-set score is retained as one
    signal, but it is no longer the whole decision.
    """
    for rec in records:
        rec.pop('_dupe_score',     None)
        rec.pop('_dupe_reason',    None)
        rec.pop('_dupe_breakdown', None)
        rec['fingerprint'] = fingerprint(rec)

    sorted_recs = sorted(records, key=lambda r: r['node_count'], reverse=True)

    clusters: dict = {}   # primary_path -> [records]
    assigned: set  = set()

    for rec in sorted_recs:
        if rec['path'] in assigned:
            continue
        primary = rec['path']
        clusters[primary] = [rec]
        assigned.add(primary)
        rec['_dupe_reason'] = 'primary'
        rec['_dupe_score']  = 1.0

        for other in sorted_recs:
            if other['path'] in assigned:
                continue
            cmp     = compare_records(rec, other)
            exactish = cmp['reason'] in {'exact file', 'same structure'}
            if exactish or cmp['score'] >= threshold:
                clusters[primary].append(other)
                assigned.add(other['path'])
                other['_dupe_score']     = cmp['score']
                other['_dupe_reason']    = cmp['reason']
                other['_dupe_breakdown'] = cmp

    return clusters
