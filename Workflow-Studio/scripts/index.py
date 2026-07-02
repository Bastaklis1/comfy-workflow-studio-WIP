"""
Workflow index persistence for ComfyUI Workflow Studio.
Saves/loads wf_studio_index.json as a sidecar to the project root.
"""

import json
from pathlib import Path


class WorkflowIndex:
    """
    Loads/saves wf_studio_index.json in the project root (parent of scripts folder).
    Stores per-workflow metadata: tags, enrichment, fingerprint, mtime, etc.

    root_dir must be passed explicitly (the parent of the scripts/ folder).
    This avoids the module depending on a global _root_dir from wfs4.py.
    """

    def __init__(self, folder: Path, root_dir: Path):
        self.folder  = folder
        self.path    = root_dir / 'wf_studio_index.json'
        self.records: dict = {}   # rel_path -> record dict
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
