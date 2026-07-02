#!/usr/bin/env python3
"""
ComfyUI Workflow Studio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unified GUI for workflow analysis, tagging, deduplication, and LLM enrichment.

Requires: wf_extract.py in the same folder (or on sys.path)
Optional: Ollama / LM Studio / any OpenAI-compatible local endpoint for LLM enrichment
Optional: anthropic package for Claude API enrichment

Run:  python wfs4.py
"""

import json
import os
import re
import sys
import threading
import shutil
import time
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
_script_dir = Path(__file__).parent   # …/Workflow-Studio/scripts/
_root_dir   = _script_dir.parent      # …/Workflow-Studio/

# ── Sibling-module imports ─────────────────────────────────────────────────────
from config        import load_config, save_config           # noqa: E402
from validation    import is_comfyui_workflow                 # noqa: E402
from tagging       import ECOSYSTEM_ANCHORS, tag_workflow     # noqa: E402
from deduplication import (                                   # noqa: E402
    DEDUPE_IGNORE_TYPES, fingerprint, similarity,
    compare_records, cluster_duplicates,
    _file_sha256, _model_basenames,
)
from llm_backends  import (                                   # noqa: E402
    ENRICH_PROMPT, ollama_enrich, claude_enrich,
    openai_compat_enrich, list_ollama_models,
)
from index         import WorkflowIndex                       # noqa: E402
from graph_canvas  import GraphCanvas                         # noqa: E402

_graph_viewer_available = True  # always available — built-in

# ── Try to import the extractor ───────────────────────────────────────────────
_extractor_path      = _script_dir / 'wf_extract.py'
_extractor_available = False
try:
    import importlib.util
    spec     = importlib.util.spec_from_file_location('wf_extractor', _extractor_path)
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


# ═════════════════════════════════════════════════════════════════════════════
# GUI
# ═════════════════════════════════════════════════════════════════════════════

from theme import BG, PNL, PNL2, ACC, DIM, FG, FG2, GREEN, RED, YEL, MONO, MONO_B, MONO_S  # noqa: E402


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
        self._selected_tags_filter: set[str] = set()
        self._running = False

        self._dupes_folder = tk.StringVar(value='00_duplicates')  # name or abs path; created in script dir
        self._trash_folder  = tk.StringVar(value='01_trash')       # name or abs path; created in script dir

        self._build_style()
        self._build_menu()
        self._build_ui()
        self._load_config()
        self._init_managed_folders()
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ── Config ──────────────────────────────────────────────────────────────

    def _load_config(self):
        cfg = load_config()
        if cfg.get('folder') and Path(cfg['folder']).exists():
            self._folder = Path(cfg['folder'])
            self._folder_lbl.configure(text=str(self._folder))
            self._index = WorkflowIndex(self._folder, _root_dir)
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
        if cfg.get('dupes_folder'):
            self._dupes_folder.set(cfg['dupes_folder'])
        if cfg.get('trash_folder'):
            self._trash_folder.set(cfg['trash_folder'])
        if cfg.get('selected_tags_filter'):
            self._selected_tags_filter = set(cfg.get('selected_tags_filter') or [])
            self._refresh_tag_filter()

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
            'selected_tags_filter': sorted(self._selected_tags_filter),
            'dupes_folder':    self._dupes_folder.get(),
            'trash_folder':    self._trash_folder.get(),
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
        fm.add_command(label='Export analysis JSON…',  command=self._export_json)
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

        self._tag_filter_btn = ttk.Button(bar, text='Tag filter',
                                          command=self._open_tag_filter_dialog)
        self._tag_filter_btn.pack(side='left', padx=(12, 4))
        ttk.Button(bar, text='Clear tags', command=self._clear_tag_filter
                   ).pack(side='left', padx=4)

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
        self._wf_tree.bind('<Double-1>', self._wf_open_folder_from_event)
        self._wf_tree.bind('<<TreeviewSelect>>', self._on_wf_select)

        return f

    def _open_tag_filter_dialog(self):
        all_tags = sorted({t for r in self._records for t in r.get('tags', [])})
        if not all_tags:
            messagebox.showinfo('No tags', 'Run tagging first.')
            return

        dlg = tk.Toplevel(self)
        dlg.title('Tag filter')
        dlg.geometry('320x420')
        dlg.configure(bg=BG)
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text='Workflows must include all selected tags.',
                  style='Dim.TLabel').pack(anchor='w', padx=12, pady=(12, 6))

        frame = ttk.Frame(dlg)
        frame.pack(fill='both', expand=True, padx=12, pady=6)
        lb = tk.Listbox(frame, selectmode='extended', bg=PNL2, fg=FG,
                        selectbackground=DIM, relief='flat',
                        font=MONO, exportselection=False)
        sb = ttk.Scrollbar(frame, orient='vertical', command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        for tag in all_tags:
            lb.insert('end', tag)
        for idx, tag in enumerate(all_tags):
            if tag in self._selected_tags_filter:
                lb.selection_set(idx)

        def apply():
            self._selected_tags_filter = {all_tags[i] for i in lb.curselection()}
            self._refresh_tag_filter()
            self._apply_filter()
            dlg.destroy()

        def clear():
            self._selected_tags_filter = set()
            self._refresh_tag_filter()
            self._apply_filter()
            dlg.destroy()

        btns = ttk.Frame(dlg, style='Panel.TFrame', padding=8)
        btns.pack(fill='x')
        ttk.Button(btns, text='Clear', command=clear).pack(side='left')
        ttk.Button(btns, text='Apply', style='Accent.TButton',
                   command=apply).pack(side='right')

    def _clear_tag_filter(self):
        self._selected_tags_filter = set()
        self._refresh_tag_filter()
        self._apply_filter()

    # ── Tab: Duplicates ──────────────────────────────────────────────────────

    def _build_tab_dedup(self, parent):
        f = ttk.Frame(parent)
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=1)

        # Controls
        ctrl = ttk.Frame(f, style='Panel.TFrame', padding=(8, 6))
        ctrl.grid(row=0, column=0, columnspan=2, sticky='ew')

        ttk.Label(ctrl, text='Similarity threshold:', background=PNL).pack(side='left')
        self._thresh_var = tk.DoubleVar(value=0.85)
        thresh_sl = tk.Scale(ctrl, from_=0.5, to=1.0, resolution=0.05, orient='horizontal',
                             variable=self._thresh_var, length=200,
                             bg=PNL, fg=FG, troughcolor=PNL2, highlightthickness=0)
        thresh_sl.pack(side='left', padx=8)
        # Only recalculate when the user *releases* the slider, not on every tick
        thresh_sl.bind('<ButtonRelease-1>', lambda e: self._refresh_dedup_display())

        self._thresh_lbl = ttk.Label(ctrl, text='85%', background=PNL, foreground=YEL)
        self._thresh_lbl.pack(side='left')
        self._thresh_var.trace_add('write',
            lambda *_: self._thresh_lbl.configure(
                text=f'{int(self._thresh_var.get()*100)}%'))

        ttk.Button(ctrl, text='Move ALL non-primaries to Duplicates folder',
                   command=self._move_dupes).pack(side='right', padx=4)
        ttk.Button(ctrl, text='Refresh', command=self._refresh_dedup_display
                   ).pack(side='right', padx=4)

        self._dupe_count_lbl = ttk.Label(ctrl, text='', foreground=YEL,
                                          background=PNL, font=MONO_S)
        self._dupe_count_lbl.pack(side='right', padx=12)

        # Duplicate tree
        cols = ('Cluster', 'File', 'Path', 'Match', 'Score', 'Node count')
        self._dupe_tree = self._make_tree(
            f, cols, widths=(55, 220, 280, 115, 70, 90), selectmode='extended')
        self._dupe_tree.grid(row=1, column=0, sticky='nsew')
        sb = ttk.Scrollbar(f, orient='vertical', command=self._dupe_tree.yview)
        self._dupe_tree.configure(yscrollcommand=sb.set)
        sb.grid(row=1, column=1, sticky='ns')

        # Tag colours
        self._dupe_tree.tag_configure('primary',   foreground=GREEN)
        self._dupe_tree.tag_configure('duplicate', foreground=YEL)
        self._dupe_tree.tag_configure('exact',     foreground=RED)
        self._dupe_tree.tag_configure(
            'cluster_hdr', foreground='#4a4a6a', background='#0e0e1c')

        # Prevent header rows from staying selected
        def _deselect_headers(event):
            for iid in self._dupe_tree.selection():
                if 'cluster_hdr' in self._dupe_tree.item(iid, 'tags'):
                    self._dupe_tree.selection_remove(iid)
        self._dupe_tree.bind('<<TreeviewSelect>>', _deselect_headers)

        self._dupe_tree.bind('<Button-3>', self._dupe_context_menu)
        self._dupe_tree.bind('<Button-2>', self._dupe_context_menu)  # macOS
        self._dupe_tree.bind('<Double-1>', self._dupe_open_folder_from_event)

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

        ttk.Separator(f).grid(row=row, column=0, columnspan=2, sticky='ew', pady=12)
        row += 1

        label('File management folders')
        ttk.Label(f, text='Folder name or full path. Plain names are created\n'
                           'automatically in the Workflow Studio folder.',
                  style='Dim.TLabel').grid(
            row=row, column=0, columnspan=2, sticky='w', pady=(0, 6))
        row += 1

        def folder_field(text, var):
            nonlocal row
            ttk.Label(f, text=text).grid(row=row, column=0, sticky='w', pady=2)
            ef = ttk.Frame(f)
            ef.grid(row=row, column=1, sticky='ew', padx=(8, 0))
            ef.columnconfigure(0, weight=1)
            ttk.Entry(ef, textvariable=var, width=34).grid(
                row=0, column=0, sticky='ew')
            ttk.Button(ef, text='Browse…',
                       command=lambda v=var: self._pick_managed_folder(v)).grid(
                row=0, column=1, padx=(4, 0))
            row += 1

        folder_field('Duplicates folder:', self._dupes_folder)
        folder_field('Trash folder:',      self._trash_folder)

        return f

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _make_tree(self, parent, cols, widths=None, selectmode='browse'):
        t = ttk.Treeview(parent, columns=cols, show='headings', selectmode=selectmode)
        for i, col in enumerate(cols):
            w = (widths[i] if widths and i < len(widths) else 150)
            t.heading(col, text=col, anchor='w',
                      command=lambda c=col, tree=t: self._sort_tree(tree, c, False))
            t.column(col, width=w, anchor='w', stretch=(i == len(cols)-1))
        return t

    def _sort_tree(self, tree, col, reverse):
        def key_for(iid):
            val = tree.set(iid, col)
            if isinstance(val, str):
                cleaned = val.strip().replace('%', '')
                if cleaned in {'—', ''}:
                    return (1, '')
                try:
                    return (0, float(cleaned))
                except ValueError:
                    return (0, cleaned.lower())
            return (0, val)
        rows = [(key_for(iid), iid) for iid in tree.get_children('')]
        rows.sort(reverse=reverse)
        for idx, (_, iid) in enumerate(rows):
            tree.move(iid, '', idx)
        tree.heading(col, text=col, anchor='w',
                     command=lambda c=col, tr=tree: self._sort_tree(tr, c, not reverse))

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
        self._index = WorkflowIndex(self._folder, _root_dir)
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

        # Snapshot existing index state for change detection
        _prev_paths = set(self._index.records.keys()) if self._index else set()

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
                'file_hash':  '',
                'tags':       [],
                'enrichment': {},
                'wf_data':    wf_data,
            }
            try:
                rec['file_hash'] = _file_sha256(fp)
            except Exception:
                rec['file_hash'] = ''
            rec['fingerprint'] = fingerprint(rec)

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

        # Generate what's-new summary
        current_paths = {rec['path'] for rec in records}
        new_wfs      = [r for r in records if r.get('is_new')]
        changed_wfs  = [r for r in records if r.get('is_changed')]
        removed_paths = _prev_paths - current_paths if '_prev_paths' in dir() else set()
        if new_wfs or changed_wfs or removed_paths:
            parts = []
            if new_wfs:
                parts.append(f'{len(new_wfs)} new')
            if changed_wfs:
                parts.append(f'{len(changed_wfs)} changed')
            if removed_paths:
                parts.append(f'{len(removed_paths)} removed')
            summary_msg = 'Since last scan: ' + ', '.join(parts) + '.'
            self.after(0, self._log, summary_msg, 'ok')
            if new_wfs:
                for r in new_wfs[:5]:
                    self.after(0, self._log, f'  + {r["name"]}', 'ok')
                if len(new_wfs) > 5:
                    self.after(0, self._log, f'  … and {len(new_wfs)-5} more', 'ok')
            if changed_wfs:
                for r in changed_wfs[:5]:
                    self.after(0, self._log, f'  ~ {r["name"]}', 'warn')
        self._last_type_counts  = dict(type_counts)
        self._last_model_counts = dict(model_counts)
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
                for sg in data.get('definitions', {}).get('subgraphs', []):
                    for n in sg.get('nodes', []):
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

    def _export_json(self):
        if not self._records:
            messagebox.showwarning('No data', 'Run extraction first.')
            return
        out = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('JSON', '*.json')],
            initialfile='wf_analysis.json',
        )
        if not out:
            return
        self._run_in_thread(self._do_export_json, out)

    def _do_export_json(self, outpath: str):
        self.after(0, self._status, 'Exporting analysis JSON…')
        export = {
            'meta': {
                'folder':          str(self._folder) if self._folder else '',
                'workflow_count':  len(self._records),
                'exported_at':     time.strftime('%Y-%m-%dT%H:%M:%S'),
            },
            'summary': {
                'node_type_frequency': self._last_type_counts if hasattr(self, '_last_type_counts') else {},
                'model_frequency':     self._last_model_counts if hasattr(self, '_last_model_counts') else {},
            },
            'workflows': [],
        }
        for rec in self._records:
            wf = {
                'name':       rec.get('name', ''),
                'path':       rec.get('path', ''),
                'node_count': rec.get('node_count', 0),
                'fingerprint':rec.get('fingerprint', ''),
                'tags':       rec.get('tags', []),
                'is_new':     rec.get('is_new', False),
                'is_changed': rec.get('is_changed', False),
                'models':     rec.get('models', []),
                'groups':     rec.get('groups', []),
                'meta':       rec.get('meta', {}),
                'enrichment': rec.get('enrichment', {}),
            }
            # Node type list (deduplicated)
            wf['node_types'] = sorted(set(
                nd.get('type', '') for nd in
                (rec.get('wf_data') or {}).get('nodes', [])
                if nd.get('type') and not nd['type'].startswith('[SUBGRAPH:')
            ))
            export['workflows'].append(wf)

        # Dedup cluster info if available
        if self._dedup_clusters:
            clusters_out = []
            for primary_path, members in self._dedup_clusters.items():
                if len(members) < 2:
                    continue
                clusters_out.append({
                    'primary': primary_path,
                    'members': [
                        {
                            'path':   m['path'],
                            'score':  round(m.get('_dupe_score', 0), 3),
                            'reason': m.get('_dupe_reason', ''),
                        }
                        for m in members[1:]
                    ],
                })
            export['duplicate_clusters'] = clusters_out

        with open(outpath, 'w', encoding='utf-8') as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        self.after(0, self._status, f'Exported JSON → {outpath}', 'ok')

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
        tag_f  = self._selected_tags_filter

        filtered = self._records
        if search:
            filtered = [r for r in filtered if
                        search in r['name'].lower() or
                        search in r['path'].lower() or
                        any(search in t.lower() for t in r.get('tags', [])) or
                        search in r.get('enrichment', {}).get('summary', '').lower() or
                        search in r.get('enrichment', {}).get('primary_purpose', '').lower()]
        if tag_f:
            filtered = [
                r for r in filtered
                if all(t in set(r.get('tags', [])) for t in tag_f)
            ]
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
        if not hasattr(self, '_tag_filter_btn'):
            return
        if self._selected_tags_filter:
            label = ', '.join(sorted(self._selected_tags_filter))
            if len(label) > 34:
                label = f'{len(self._selected_tags_filter)} tags selected'
            self._tag_filter_btn.configure(text=f'Tags: {label}')
        else:
            self._tag_filter_btn.configure(text='Tag filter')

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
        menu.add_separator()
        menu.add_command(label='Show graph preview',
                         command=lambda: self._show_graph(full_path))
        menu.post(event.x_root, event.y_root)

    def _wf_open_folder_from_event(self, event):
        iid = self._wf_tree.identify_row(event.y)
        if not iid:
            return
        self._wf_tree.selection_set(iid)
        rec = next((r for r in self._records if r['path'] == iid), None)
        if not rec:
            return
        full_path = str(self._folder / rec['path']) if self._folder else rec['path']
        self._open_folder(full_path)

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
        """Right-click menu on duplicate tree rows, including multi-select."""
        iid = self._dupe_tree.identify_row(event.y)
        if not iid:
            return
        # Don't show a menu for cluster header rows
        if 'cluster_hdr' in self._dupe_tree.item(iid, 'tags'):
            return
        if iid not in self._dupe_tree.selection():
            self._dupe_tree.selection_set(iid)

        selected = []
        for sid in self._dupe_tree.selection():
            if 'cluster_hdr' in self._dupe_tree.item(sid, 'tags'):
                continue
            vals = self._dupe_tree.item(sid).get('values', [])
            if vals and len(vals) >= 3:
                selected.append((sid, vals[2]))
        if not selected:
            return

        rel_path = selected[0][1]
        full_path = str(self._folder / rel_path) if self._folder else rel_path
        full_paths = [
            str(self._folder / rel) if self._folder else rel
            for _, rel in selected
        ]
        rel_paths = [rel for _, rel in selected]

        menu = tk.Menu(self, tearoff=False, bg=PNL, fg=FG,
                       activebackground=DIM, activeforeground=FG,
                       font=MONO)
        if len(selected) == 1:
            menu.add_command(label='Copy path',
                             command=lambda: self._copy_to_clipboard(full_path))
            menu.add_command(label='Copy relative path',
                             command=lambda: self._copy_to_clipboard(rel_path))
        else:
            menu.add_command(label=f'Copy {len(selected)} paths',
                             command=lambda: self._copy_to_clipboard('\n'.join(full_paths)))
        menu.add_separator()
        if len(selected) == 1:
            menu.add_command(label='Open containing folder',
                             command=lambda: self._open_folder(full_path))
            menu.add_command(label='Open file in default app',
                             command=lambda: self._open_file(full_path))
            menu.add_command(label='Show graph preview',
                             command=lambda: self._show_graph(full_path))
        else:
            menu.add_command(label='Open selected folders',
                             command=lambda: self._open_many_folders(full_paths))
            menu.add_command(label='Open selected files',
                             command=lambda: self._open_many_files(full_paths))
        menu.add_separator()
        if len(selected) == 1:
            menu.add_command(label='Compare with primary',
                             command=lambda: self._compare_with_primary(iid, full_path))
        if len(selected) == 2:
            menu.add_command(label='Compare selected pair',
                             command=lambda: self._compare_side_by_side(full_paths[0], full_paths[1]))
        menu.add_separator()
        menu.add_command(
            label=f'Move {len(selected)} selected to Duplicates folder',
            command=lambda: self._move_selected_dupes(list(rel_paths)))
        menu.add_command(
            label=f'Move {len(selected)} selected to Trash folder',
            command=lambda: self._delete_selected_dupes(list(rel_paths)))
        menu.post(event.x_root, event.y_root)

    def _dupe_open_folder_from_event(self, event):
        iid = self._dupe_tree.identify_row(event.y)
        if not iid or 'cluster_hdr' in self._dupe_tree.item(iid, 'tags'):
            return
        self._dupe_tree.selection_set(iid)
        vals = self._dupe_tree.item(iid).get('values', [])
        if len(vals) < 3:
            return
        rel_path = vals[2]
        full_path = str(self._folder / rel_path) if self._folder else rel_path
        self._open_folder(full_path)

    def _copy_to_clipboard(self, text: str):
        self.clipboard_clear()
        self.clipboard_append(text)
        self._status(f'Copied: {text}')

    def _open_folder(self, path: str):
        import subprocess
        try:
            if sys.platform == 'win32':
                # /select highlights the specific file in Explorer
                subprocess.Popen(['explorer', '/select,', Path(path).resolve()])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', '-R', path])   # -R reveals in Finder
            else:
                subprocess.Popen(['xdg-open', str(Path(path).parent)])
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

    def _open_many_folders(self, paths: list[str]):
        seen = set()
        for path in paths:
            folder = str(Path(path).parent)
            if folder in seen:
                continue
            seen.add(folder)
            self._open_folder(path)

    def _open_many_files(self, paths: list[str]):
        for path in paths:
            self._open_file(path)

    def _compare_with_primary(self, iid: str, full_path: str):
        """Open this workflow and its cluster primary in an in-window summary."""
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

        self._compare_side_by_side(primary_path, full_path)

    def _record_for_full_path(self, full_path: str) -> Optional[dict]:
        if not self._folder:
            return None
        try:
            rel = str(Path(full_path).resolve().relative_to(self._folder.resolve()))
        except Exception:
            rel = str(full_path)
        rel = rel.replace('/', '\\')
        return next((r for r in self._records if r['path'].replace('/', '\\') == rel), None)

    def _comparison_lines(self, rec: dict) -> list[str]:
        lines = [
            rec.get('name', ''),
            rec.get('path', ''),
            '',
            f"Nodes: {rec.get('node_count', 0)}",
            f"Tags: {', '.join(rec.get('tags', [])) or '—'}",
        ]
        enr = rec.get('enrichment') or {}
        if enr:
            lines += [
                f"Purpose: {enr.get('primary_purpose', '—')}",
                f"Summary: {enr.get('summary', '—')}",
                f"Notable: {enr.get('notable', '—')}",
            ]
        models = _model_basenames(rec.get('models', []))
        lines += ['', 'Models:']
        lines += [f'  {m}' for m in models[:80]] or ['  —']
        if len(models) > 80:
            lines.append(f'  ... {len(models) - 80} more')
        counts = defaultdict(int)
        for nd in (rec.get('wf_data') or {}).get('nodes', []):
            t = nd.get('type', '?')
            if t not in DEDUPE_IGNORE_TYPES and not t.startswith('[SUBGRAPH:'):
                counts[t] += 1
        for itype, cnt in (rec.get('wf_data') or {}).get('sg_inner_counts', {}).items():
            if itype not in DEDUPE_IGNORE_TYPES:
                counts[itype] += cnt
        lines += ['', 'Meaningful node types:']
        for t, c in sorted(counts.items(), key=lambda x: (-x[1], x[0].lower()))[:140]:
            lines.append(f'  {c:3d}x {t}')
        if len(counts) > 140:
            lines.append(f'  ... {len(counts) - 140} more')
        return lines

    def _compare_side_by_side(self, path1: str, path2: str):
        rec1 = self._record_for_full_path(path1)
        rec2 = self._record_for_full_path(path2)
        if not rec1 or not rec2:
            self._status('Could not find workflow data for comparison.', 'error')
            return
        cmp = compare_records(rec1, rec2)
        win = tk.Toplevel(self)
        win.title('Workflow comparison')
        win.geometry('1180x760')
        win.configure(bg=BG)

        head = ttk.Frame(win, style='Panel.TFrame', padding=8)
        head.pack(fill='x')
        ttk.Label(
            head,
            text=(f"Match: {cmp['reason']}  |  score {cmp['score']:.0%}  |  "
                  f"structure {cmp['structure']:.0%}, shape {cmp['degree']:.0%}, "
                  f"toolset {cmp['toolset']:.0%}, models {cmp['models']:.0%}"),
            background=PNL, foreground=YEL, font=MONO_B
        ).pack(side='left')

        body = ttk.Frame(win)
        body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        def make_panel(col, rec, path):
            frame = ttk.Frame(body, padding=6)
            frame.grid(row=0, column=col, sticky='nsew')
            frame.rowconfigure(1, weight=1)
            frame.columnconfigure(0, weight=1)
            ttk.Label(frame, text=rec['name'], style='Head.TLabel').grid(row=0, column=0, sticky='w')
            text = tk.Text(frame, bg=PNL2, fg=FG, insertbackground=FG,
                           font=MONO_S, wrap='none', relief='flat')
            ys = ttk.Scrollbar(frame, orient='vertical', command=text.yview)
            text.configure(yscrollcommand=ys.set)
            text.grid(row=1, column=0, sticky='nsew')
            ys.grid(row=1, column=1, sticky='ns')
            text.insert('end', '\n'.join(self._comparison_lines(rec)))
            text.configure(state='disabled')
            btns = ttk.Frame(frame, style='Panel.TFrame', padding=4)
            btns.grid(row=2, column=0, columnspan=2, sticky='ew')
            ttk.Button(btns, text='Open folder',
                       command=lambda: self._open_folder(path)).pack(side='left')
            ttk.Button(btns, text='Graph',
                       command=lambda: self._show_graph(path)).pack(side='left', padx=4)

        make_panel(0, rec1, path1)
        make_panel(1, rec2, path2)

    def _show_graph(self, full_path: str):
        win = tk.Toplevel(self)
        win.title(f'Graph — {Path(full_path).name}')
        win.geometry('1100x760')
        win.configure(bg=BG)
        # Title bar with file info
        bar = ttk.Frame(win, style='Panel.TFrame', padding=(8, 6))
        bar.pack(fill='x')
        ttk.Label(bar, text=str(full_path), style='Dim.TLabel',
                  background=PNL).pack(side='left')
        ttk.Button(bar, text='Open folder',
                   command=lambda: self._open_folder(full_path)).pack(side='right')
        canvas = GraphCanvas(win)
        canvas.pack(fill='both', expand=True)
        canvas.load_workflow(full_path)

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

            total = 1 + len(dupes)
            self._dupe_tree.insert('', 'end', tags=('cluster_hdr',), values=(
                f'Cluster #{cluster_id}',
                f'{total} files  ──────────────────────────────────────────',
                '', '', '', '',
            ))

            # Primary row
            self._dupe_tree.insert('', 'end', tags=('primary',), values=(
                f'#{cluster_id}', primary['name'], primary['path'],
                'primary', '—', primary['node_count'],
            ))

            for dup in dupes:
                cmp = dup.get('_dupe_breakdown') or compare_records(primary, dup)
                reason = dup.get('_dupe_reason') or cmp['reason']
                score = dup.get('_dupe_score', cmp['score'])
                is_exact = reason in {'exact file', 'same structure'}
                tag = 'exact' if is_exact else 'duplicate'
                self._dupe_tree.insert('', 'end', tags=(tag,), values=(
                    f'#{cluster_id}', dup['name'], dup['path'],
                    reason,
                    f'{score:.0%}',
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

        dest_path = self._resolve_move_dest()
        if not dest_path:
            return

        moved = 0
        errors = []
        for primary_path, recs in dup_clusters.items():
            for rec in recs[1:]:   # skip primary
                src = self._folder / rec['path']
                # Mirror folder structure
                try:
                    rel_parts = Path(rec['path']).parts
                    dst = self._unique_destination(dest_path.joinpath(*rel_parts))
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

    def _pick_managed_folder(self, var: tk.StringVar):
        d = filedialog.askdirectory(title='Select folder…')
        if d:
            var.set(d)

    def _init_managed_folders(self):
        """Create 00_duplicates and 01_trash in the project root on startup."""
        for var in (self._dupes_folder, self._trash_folder):
            raw = var.get().strip()
            if not raw:
                continue
            p = Path(raw)
            dest = p if p.is_absolute() else _root_dir / raw
            try:
                dest.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass   # non-fatal; will surface properly on first use

    def _resolve_managed_folder(self, name_var: tk.StringVar, purpose: str) -> Optional[Path]:
        """
        Resolve a managed folder (move-to or trash).
        Plain names resolve to the project root (Workflow-Studio/).
        Full absolute paths are used as-is.
        Auto-creates the folder if it doesn't exist.
        """
        raw = name_var.get().strip() or name_var.get()
        p   = Path(raw)
        dest = p if p.is_absolute() else _root_dir / raw
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror('Folder error',
                                  f'Could not create {purpose} folder:\n{dest}\n\n{e}')
            return None
        return dest

    def _resolve_move_dest(self) -> Optional[Path]:
        return self._resolve_managed_folder(self._dupes_folder, 'move-to')

    def _resolve_trash_dest(self) -> Optional[Path]:
        return self._resolve_managed_folder(self._trash_folder, 'trash')

    def _unique_destination(self, dst: Path) -> Path:
        """Return a non-overwriting destination path for moved workflow files."""
        if not dst.exists():
            return dst
        stem, suffix = dst.stem, dst.suffix
        for i in range(2, 1000):
            candidate = dst.with_name(f'{stem} ({i}){suffix}')
            if not candidate.exists():
                return candidate
        return dst.with_name(f'{stem} ({int(time.time())}){suffix}')

    def _source_and_relative_path(self, path: str) -> tuple[Path, Path]:
        """
        Convert a duplicate-tree path into source and preserved relative path.
        The tree normally stores paths relative to the scanned workflow folder.
        """
        raw = Path(path)
        if raw.is_absolute():
            src = raw
            try:
                rel = raw.relative_to(self._folder) if self._folder else Path(raw.name)
            except ValueError:
                rel = Path(raw.name)
        else:
            rel = raw
            src = self._folder / raw if self._folder else raw
        return src, rel

    def _move_selected_dupes(self, paths: list[str]):
        if not paths:
            return
        dest_path = self._resolve_move_dest()
        if not dest_path:
            return
        moved, errors = 0, []
        for path in paths:
            src, rel = self._source_and_relative_path(path)
            if not src.exists():
                errors.append(f'{src.name}: not found')
                continue
            try:
                dst = self._unique_destination(dest_path.joinpath(*rel.parts))
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved += 1
            except Exception as e:
                errors.append(f'{rel}: {e}')
        msg = f'Moved {moved} file(s) to {dest_path}.'
        if errors:
            msg += '\n\nErrors:\n' + '\n'.join(errors[:5])
        messagebox.showinfo('Done', msg)
        self._run_extraction()

    def _delete_selected_dupes(self, paths: list[str]):
        if not paths:
            return
        trash = self._resolve_trash_dest()
        if not trash:
            return
        rels = [self._source_and_relative_path(p)[1] for p in paths]
        names = '\n'.join(str(p) for p in rels[:8])
        if len(paths) > 8:
            names += f'\n… and {len(paths)-8} more'
        if not messagebox.askyesno(
                'Confirm move to trash',
                f'Move {len(paths)} file(s) to trash folder?\n\n{names}'):
            return
        moved, errors = 0, []
        for path in paths:
            src, rel = self._source_and_relative_path(path)
            try:
                dst = self._unique_destination(trash.joinpath(*rel.parts))
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved += 1
            except Exception as e:
                errors.append(f'{rel}: {e}')
        msg = f'Moved {moved} file(s) to trash folder.'
        if errors:
            msg += '\n\nErrors:\n' + '\n'.join(errors[:5])
        messagebox.showinfo('Done', msg)
        self._run_extraction()

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
