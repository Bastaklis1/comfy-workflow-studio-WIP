"""
Graph canvas for ComfyUI Workflow Studio.
Self-contained ComfyUI workflow graph renderer — no external dependencies.

Layout:  Topological depth columns (data flows left → right).
         Source nodes are pulled adjacent to the node they feed so they
         don't all pile into column 0.
Pan:     Left-drag on empty canvas space.
Zoom:    Scroll wheel.
Click:   Click a node to highlight its incoming (red) and outgoing (blue)
         connections.  Click empty space to clear.
"""

import json
import tkinter as tk
from collections import defaultdict, deque
from tkinter import ttk

# ── Local colour constants (mirror the main theme for the graph renderer) ─────
# Keeping these here avoids a circular import from the theme module.
_FG2   = '#8888bb'
_YEL   = '#ccaa44'
_MONO  = ('Consolas', 10)
_MONO_S = ('Consolas', 9)


# ── Node colour palette ───────────────────────────────────────────────────────

def _node_colour(ntype: str):
    """Returns (fill, text_colour) for a node type based on category."""
    t = ntype.lower()
    if any(x in t for x in ('loader', 'unet', 'vae', 'checkpoint', 'clip')):
        return '#1a2a4a', '#7ab4ff'
    if any(x in t for x in ('ksampler', 'sampler', 'scheduler', 'guider', 'noise')):
        return '#2a1a3a', '#c090ff'
    if 'lora' in t:
        return '#1a3a2a', '#70d090'
    if any(x in t for x in ('controlnet', 'preprocessor', 'openpose', 'depth', 'canny')):
        return '#3a2a1a', '#d09050'
    if any(x in t for x in ('detailer', 'upscale', 'esrgan', 'seedvr', 'adetailer')):
        return '#3a1a1a', '#ff7070'
    if any(x in t for x in ('save', 'preview', 'output')):
        return '#1a3a3a', '#50d0d0'
    if any(x in t for x in ('text', 'prompt', 'wildcard', 'clip text')):
        return '#2a2a1a', '#d0d060'
    if any(x in t for x in ('video', 'wan', 'ltx', 'vhs', 'rife', 'mochi')):
        return '#1a2a3a', '#60a0d0'
    if any(x in t for x in ('florence', 'caption', 'vision', 'qwen')):
        return '#2a1a2a', '#d060d0'
    if any(x in t for x in ('face', 'reactor', 'pulid', 'instantid', 'reacter')):
        return '#3a2a3a', '#d090c0'
    if any(x in t for x in ('sam', 'segment', 'mask', 'rmbg')):
        return '#1a2a1a', '#80c080'
    if 'subgraph' in t:
        return '#222222', '#aaaaaa'
    return '#1e1e2e', '#ccccdd'


# ── Graph canvas ──────────────────────────────────────────────────────────────

class GraphCanvas(tk.Frame):
    _NODE_W = 160
    _NODE_H = 36
    _H_GAP  = 54
    _V_GAP  = 12
    _PAD    = 30

    _SKIP = {'Reroute', 'Reroute (rgthree)'}

    # Edge colours
    _COL_EDGE_DIM  = '#1e2a3e'
    _COL_EDGE_OUT  = '#4488ff'   # outgoing from selected node
    _COL_EDGE_IN   = '#ff5533'   # incoming to selected node

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg='#0a0a18', **kwargs)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._cv = tk.Canvas(self, bg='#0a0a18', highlightthickness=0)
        self._cv.grid(row=0, column=0, sticky='nsew')
        sb_y = ttk.Scrollbar(self, orient='vertical',   command=self._cv.yview)
        sb_x = ttk.Scrollbar(self, orient='horizontal', command=self._cv.xview)
        self._cv.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        sb_y.grid(row=0, column=1, sticky='ns')
        sb_x.grid(row=1, column=0, sticky='ew')

        # Pan + click (differentiated by drag distance)
        self._drag_occurred = False
        self._press_xy = (0, 0)
        self._cv.bind('<ButtonPress-1>',   self._on_press)
        self._cv.bind('<B1-Motion>',       self._on_motion)
        self._cv.bind('<ButtonRelease-1>', self._on_release)
        self._cv.bind('<MouseWheel>',      self._on_zoom)
        self._cv.bind('<Button-4>',        lambda e: self._zoom(1.1, e))
        self._cv.bind('<Button-5>',        lambda e: self._zoom(0.9, e))

        # Connectivity index populated by _render
        # nid -> {'out': [(from,to),...], 'in': [(from,to),...]}
        self._conn: dict = {}
        self._highlighted: int | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    def load_workflow(self, filepath: str):
        try:
            with open(filepath, encoding='utf-8', errors='replace') as f:
                data = json.load(f)
        except Exception as e:
            self._cv.delete('all')
            self._cv.create_text(100, 60, text=f'Load error: {e}',
                                  fill='#ff7070', anchor='w', font=_MONO_S)
            return
        self._render(data.get('nodes', []), data.get('links', []),
                     data.get('groups', []))

    # ── Pan / click ───────────────────────────────────────────────────────────

    def _on_press(self, event):
        self._cv.scan_mark(event.x, event.y)
        self._press_xy = (event.x, event.y)
        self._drag_occurred = False

    def _on_motion(self, event):
        self._cv.scan_dragto(event.x, event.y, gain=1)
        px, py = self._press_xy
        if abs(event.x - px) + abs(event.y - py) > 5:
            self._drag_occurred = True

    def _on_release(self, event):
        if not self._drag_occurred:
            self._handle_click(event.x, event.y)

    def _handle_click(self, ex, ey):
        cx = self._cv.canvasx(ex)
        cy = self._cv.canvasy(ey)
        items = self._cv.find_overlapping(cx - 5, cy - 5, cx + 5, cy + 5)
        clicked = None
        for item in reversed(items):
            for tag in self._cv.gettags(item):
                if tag.startswith('node_'):
                    try:
                        clicked = int(tag[5:])
                    except ValueError:
                        pass
                    break
            if clicked is not None:
                break

        if clicked is not None and clicked != self._highlighted:
            self._highlight(clicked)
        else:
            self._clear_highlight()

    def _highlight(self, nid: int):
        self._highlighted = nid
        # Dim all edges and reset all node outlines
        self._cv.itemconfig('edge',      fill=self._COL_EDGE_DIM, width=1)
        self._cv.itemconfig('node_rect', outline='#334466',        width=1)

        # Walk full upstream chain (BFS through incoming edges)
        upstream: set[int] = set()
        frontier = {nid}
        while frontier:
            nxt: set[int] = set()
            for n in frontier:
                for frm, to in self._conn.get(n, {}).get('in', []):
                    self._cv.itemconfig(f'edge_{frm}_{to}',
                                         fill=self._COL_EDGE_IN, width=2)
                    if frm not in upstream:
                        upstream.add(frm)
                        nxt.add(frm)
            frontier = nxt

        # Walk full downstream chain (BFS through outgoing edges)
        downstream: set[int] = set()
        frontier = {nid}
        while frontier:
            nxt = set()
            for n in frontier:
                for frm, to in self._conn.get(n, {}).get('out', []):
                    self._cv.itemconfig(f'edge_{frm}_{to}',
                                         fill=self._COL_EDGE_OUT, width=2)
                    if to not in downstream:
                        downstream.add(to)
                        nxt.add(to)
            frontier = nxt

        # Highlight selected node rect only (text has no -outline option)
        self._cv.itemconfig(f'rect_{nid}', outline='#ffffff', width=2)

    def _clear_highlight(self):
        self._highlighted = None
        self._cv.itemconfig('edge',      fill=self._COL_EDGE_DIM, width=1)
        self._cv.itemconfig('node_rect', outline='#334466',        width=1)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _orig_pos(self, node):
        pos = node.get('pos', [0, 0])
        if isinstance(pos, list) and len(pos) >= 2:
            return pos[0], pos[1]
        if isinstance(pos, dict):
            return pos.get('0', 0), pos.get('1', 0)
        return 0, 0

    def _group_membership(self, nodes, groups):
        membership = {}
        for n in nodes:
            ox, oy = self._orig_pos(n)
            for g in groups:
                bx, by, bw, bh = g.get('bounding', [0, 0, 0, 0])
                if bx <= ox <= bx + bw and by <= oy <= by + bh:
                    t = g.get('title', '')
                    if t:
                        membership[n['id']] = t
                    break
        return membership

    def _topological_layout(self, nodes, links):
        """
        Left-to-right topological layout.

        Two-pass depth assignment:
        1. Standard Kahn BFS (longest path from sources → depth).
        2. "Pull" pass: source nodes with a single outgoing edge are placed
           one column to the left of their target, so they don't all pile into
           column 0.  Repeated up to 4 times to propagate cascades.

        Returns (positions dict, edge_set for link drawing).
        """
        id_to_node = {n['id']: n for n in nodes}
        visible = {
            n['id'] for n in nodes
            if n.get('type') not in self._SKIP and n.get('mode', 0) != 4
        }

        link_src = {lk[0]: lk[1] for lk in links if len(lk) >= 4}

        def resolve_reroute(nid, seen=None):
            if seen is None: seen = set()
            if nid in seen or nid not in id_to_node: return None
            seen.add(nid)
            n = id_to_node[nid]
            if n.get('type') not in self._SKIP: return nid
            for inp in n.get('inputs', []):
                lid = inp.get('link')
                if lid is not None and lid in link_src:
                    return resolve_reroute(link_src[lid], seen)
            return None

        children  = defaultdict(set)
        in_degree = {nid: 0 for nid in visible}

        for lk in links:
            if len(lk) < 4: continue
            from_id, to_id = lk[1], lk[3]
            if id_to_node.get(from_id, {}).get('type') in self._SKIP:
                from_id = resolve_reroute(from_id)
            if id_to_node.get(to_id, {}).get('type') in self._SKIP: continue
            if from_id is None or from_id not in visible or to_id not in visible: continue
            if to_id not in children[from_id]:
                children[from_id].add(to_id)
                in_degree[to_id] += 1

        # Pass 1: BFS depth
        depth: dict[int, int] = {}
        queue: deque = deque()
        for nid in visible:
            if in_degree.get(nid, 0) == 0:
                depth[nid] = 0
                queue.append(nid)
        while queue:
            nid = queue.popleft()
            d = depth.get(nid, 0)
            for child in children[nid]:
                nd = d + 1
                if nd > depth.get(child, -1):
                    depth[child] = nd
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        max_d = max(depth.values(), default=0) if depth else 0
        for nid in visible:
            if nid not in depth:
                depth[nid] = max_d + 1   # cycles

        # Pass 2: pull every node rightward toward its first consumer.
        parents: dict[int, set] = defaultdict(set)
        for frm, targets in children.items():
            for to in targets:
                parents[to].add(frm)

        topo_order = sorted(visible, key=lambda n: depth.get(n, 0), reverse=True)
        for nid in topo_order:
            if not children.get(nid):
                continue   # leaf/sink node — stays put
            child_depths = [depth[c] for c in children[nid] if c in depth]
            if not child_depths:
                continue
            new_d = min(child_depths) - 1
            if new_d > depth.get(nid, 0):
                if all(new_d < depth[c] for c in children[nid] if c in depth):
                    depth[nid] = new_d

        # Compact to consecutive integers (gaps may appear after pull)
        unique = sorted(set(depth.values()))
        remap  = {d: i for i, d in enumerate(unique)}
        depth  = {nid: remap[d] for nid, d in depth.items()}

        # Build columns, sort within column by original y
        columns: dict[int, list] = defaultdict(list)
        for nid in visible:
            _, oy = self._orig_pos(id_to_node.get(nid, {}))
            columns[depth[nid]].append((oy, nid))
        for col in columns.values():
            col.sort()

        NW, NH = self._NODE_W, self._NODE_H
        positions: dict[int, tuple] = {}
        cx = self._PAD
        for d in sorted(columns.keys()):
            cy = self._PAD
            for _, nid in columns[d]:
                positions[nid] = (cx, cy)
                cy += NH + self._V_GAP
            cx += NW + self._H_GAP

        return positions, children

    # ── Render ────────────────────────────────────────────────────────────────

    def _render(self, nodes, links, groups):
        cv = self._cv
        cv.delete('all')
        self._conn = {}
        self._highlighted = None

        visible_nodes = [
            n for n in nodes
            if n.get('type') not in self._SKIP and n.get('mode', 0) != 4
        ]
        if not visible_nodes:
            cv.create_text(100, 60, text='No renderable nodes',
                            fill=_FG2, anchor='w', font=_MONO)
            return

        membership = self._group_membership(nodes, groups)
        positions, _ = self._topological_layout(nodes, links)

        id_to_node = {n['id']: n for n in nodes}
        NW, NH = self._NODE_W, self._NODE_H

        # ── Group bounding boxes ──────────────────────────────────────────────
        group_bounds: dict[str, list] = {}
        for nid, grp in membership.items():
            if nid not in positions: continue
            x, y = positions[nid]
            if grp not in group_bounds:
                group_bounds[grp] = [x, y, x + NW, y + NH]
            else:
                b = group_bounds[grp]
                b[0] = min(b[0], x);  b[1] = min(b[1], y)
                b[2] = max(b[2], x + NW); b[3] = max(b[3], y + NH)

        P = 6
        for grp, (x1, y1, x2, y2) in group_bounds.items():
            cv.create_rectangle(x1-P, y1-P, x2+P, y2+P,
                                  fill='#0d0d1a', outline='#2a2a44',
                                  width=1, dash=(4, 4))
            cv.create_text(x1-P+4, y1-P+2, text=grp,
                            fill='#555577', font=_MONO_S, anchor='nw')

        # ── Links ─────────────────────────────────────────────────────────────
        link_src = {lk[0]: lk[1] for lk in links if len(lk) >= 4}

        def resolve_reroute(nid, seen=None):
            if seen is None: seen = set()
            if nid in seen or nid not in id_to_node: return None
            seen.add(nid)
            n = id_to_node[nid]
            if n.get('type') not in self._SKIP: return nid
            for inp in n.get('inputs', []):
                lid = inp.get('link')
                if lid is not None and lid in link_src:
                    return resolve_reroute(link_src[lid], seen)
            return None

        drawn: set = set()
        for lk in links:
            if len(lk) < 4: continue
            from_id, to_id = lk[1], lk[3]
            if id_to_node.get(from_id, {}).get('type') in self._SKIP:
                from_id = resolve_reroute(from_id)
            if id_to_node.get(to_id, {}).get('type') in self._SKIP: continue
            if from_id is None or from_id == to_id: continue
            key = (from_id, to_id)
            if key in drawn: continue
            drawn.add(key)
            if from_id not in positions or to_id not in positions: continue

            # Build connectivity index for click-highlighting
            self._conn.setdefault(from_id, {'out': [], 'in': []})['out'].append(key)
            self._conn.setdefault(to_id,   {'out': [], 'in': []})['in'].append(key)

            sx, sy = positions[from_id]
            dx, dy = positions[to_id]
            x1 = sx + NW;  y1 = sy + NH // 2
            x2 = dx;       y2 = dy + NH // 2
            mid = (x1 + x2) / 2
            tag = f'edge_{from_id}_{to_id}'
            cv.create_line(x1, y1, mid, y1, mid, y2, x2, y2,
                            fill=self._COL_EDGE_DIM, width=1,
                            smooth=True, splinesteps=10,
                            tags=(tag, 'edge'))

        # ── Nodes ─────────────────────────────────────────────────────────────
        for n in visible_nodes:
            nid   = n['id']
            ntype = n.get('type', '?')
            mode  = n.get('mode', 0)
            if nid not in positions: continue

            x, y = positions[nid]
            fill, text_col = _node_colour(ntype)
            if mode == 2:
                fill = '#1a1a1a';  text_col = '#555555'

            node_tag = f'node_{nid}'
            rect_tag = f'rect_{nid}'
            cv.create_rectangle(x, y, x+NW, y+NH,
                                  fill=fill, outline='#334466', width=1,
                                  tags=(node_tag, rect_tag, 'node_rect', 'node'))
            if mode == 2:
                cv.create_rectangle(x, y, x+3, y+NH, fill=_YEL, outline='',
                                     tags=('node',))

            label = n.get('title') or ntype
            if len(label) > 22: label = label[:20] + '…'
            cv.create_text(x+NW//2, y+NH//2, text=label,
                            fill=text_col, font=_MONO_S, width=NW-8,
                            tags=(node_tag, 'node'))

        # ── Scroll region ─────────────────────────────────────────────────────
        bbox = cv.bbox('all')
        if bbox:
            cv.configure(scrollregion=(bbox[0]-30, bbox[1]-30,
                                        bbox[2]+30, bbox[3]+30))

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def _on_zoom(self, event):
        self._zoom(1.1 if event.delta > 0 else 0.9, event)

    def _zoom(self, factor, event=None):
        if event:
            cx = self._cv.canvasx(event.x)
            cy = self._cv.canvasy(event.y)
        else:
            cx = self._cv.canvasx(self._cv.winfo_width() // 2)
            cy = self._cv.canvasy(self._cv.winfo_height() // 2)
        self._cv.scale('all', cx, cy, factor, factor)
        bbox = self._cv.bbox('all')
        if bbox:
            self._cv.configure(scrollregion=(bbox[0]-30, bbox[1]-30,
                                              bbox[2]+30, bbox[3]+30))
