"""Object-layer combinators (M5) — the composable core that covers the
structural stratum, registered into the SAME `clarc.dsl.core.REGISTRY`.

Abstraction convention (the key idea that keeps the SMT layer unchanged): the
abstract state σ always describes **the grid the current value would render to**.
`objects()` and `render()` are therefore σ-IDENTITY (re-viewing the same content),
and the SELECTION→SELECTION transforms carry the abstract effect — they change
what `render()` will produce, so they change σ. The z3 `encode`s below relate the
rendered-σ before/after; `clarc/smt.py` threads σ exactly as for whole-grid prims.

`objects()` is 4-connected, matching σ's `n_obj`/`osz`/`ocol` (so select/remove-by-
color get EXACT object-count contracts via `ocol`). 8-connectivity would need σ to
track it too — deferred.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import z3
from scipy import ndimage

from clarc.dsl.absdomain import K_OBJ, N_COLORS, N_SHAPE, ZState
from clarc.contracts.vocab import bg as bg_of
from clarc.dsl.core import _COLORS, Param, Primitive, _case, _reg
from clarc.dsl.types import Obj, Selection, Ty

_CONN4 = ndimage.generate_binary_structure(2, 1)


# --------------------------------------------------------------------------- #
# Runtime: grid <-> selection
# --------------------------------------------------------------------------- #

def grid_to_selection(grid: np.ndarray) -> Selection:
    g = np.asarray(grid, dtype=int)
    bgc = int(bg_of(g))
    lab, n = ndimage.label(g != bgc, structure=_CONN4)
    objs: list[Obj] = []
    for k in range(1, n + 1):
        m = lab == k
        vals, counts = np.unique(g[m], return_counts=True)
        dom = int(vals[np.flatnonzero(counts == counts.max()).min()])
        objs.append(Obj(mask=m, color=dom, size=int(m.sum())))
    return Selection(objs=objs, grid=g, bg=bgc)


def render(sel: Selection) -> np.ndarray:
    out = np.full(sel.shape, sel.bg, dtype=int)
    for o in sel.objs:
        out[o.mask] = o.recolor if o.recolor is not None else sel.grid[o.mask]
    return out


def to_grid(val) -> np.ndarray:
    """Render if it's a Selection, else assume it's already a grid."""
    return render(val) if isinstance(val, Selection) else np.asarray(val, dtype=int)


# --------------------------------------------------------------------------- #
# z3 encode helpers (σ = rendered-grid state)
# --------------------------------------------------------------------------- #

def _eq_sigma(si: ZState, so: ZState) -> list[z3.BoolRef]:
    cs = [so.h == si.h, so.w == si.w, so.n_obj == si.n_obj,
          so.bbox_h == si.bbox_h, so.bbox_w == si.bbox_w]
    cs += [so.cnt[c] == si.cnt[c] for c in range(N_COLORS)]
    cs += [so.sym[i] == si.sym[i] for i in range(5)]
    cs += [so.osz[j] == si.osz[j] for j in range(K_OBJ)]
    cs += [so.ocol[c] == si.ocol[c] for c in range(N_COLORS)]
    return cs


def _subset(si: ZState, so: ZState) -> list[z3.BoolRef]:
    """Common bounds when a filter REMOVES objects (renders a subset on bg).
    Non-bg cells can only be lost; the BACKGROUND count grows (removed object
    cells become bg), so bg is excluded from the per-color upper bound."""
    cs = [so.h == si.h, so.w == si.w, so.nonbg <= si.nonbg]
    cs += [z3.Implies(si.bg != c, so.cnt[c] <= si.cnt[c]) for c in range(N_COLORS)]
    cs += [so.osz[j] <= si.osz[j] for j in range(K_OBJ)]   # j-th largest kept ≤ overall
    # a subset of objects: every shape-class count, holed and border count drop
    cs += [so.oshape[k] <= si.oshape[k] for k in range(N_SHAPE)]
    cs += [so.n_holed <= si.n_holed, so.n_border <= si.n_border]
    return cs


# --------------------------------------------------------------------------- #
# Bridge: GRID <-> SELECTION (σ-identity)
# --------------------------------------------------------------------------- #

def _enc_identity(si, so, P):
    return _eq_sigma(si, so)


_reg(Primitive("objects", "bridge", (), "view the grid as its 4-connected non-background objects",
               lambda g, P: grid_to_selection(g), _enc_identity,
               in_type=Ty.GRID, out_type=Ty.SELECTION))

_reg(Primitive("render", "bridge", (), "paint the current objects back onto a background grid",
               lambda s, P: render(s), _enc_identity,
               in_type=Ty.SELECTION, out_type=Ty.GRID))


# --------------------------------------------------------------------------- #
# Selection -> Selection: size selectors
# --------------------------------------------------------------------------- #

def _apply_select_largest(sel, P):
    if not sel.objs:
        return sel
    mx = max(o.size for o in sel.objs)
    return replace(sel, objs=[o for o in sel.objs if o.size == mx])


def _enc_select_largest(si, so, P):
    # (>=1 only when there WAS an object; empty selection renders to empty)
    return _subset(si, so) + [z3.Implies(si.n_obj >= 1, so.n_obj >= 1),
                              so.n_obj <= si.n_obj,
                              so.osz[0] == si.osz[0]]   # largest survives, exact


_reg(Primitive("select_largest", "select", (), "keep only the largest object(s)",
               _apply_select_largest, _enc_select_largest,
               in_type=Ty.SELECTION, out_type=Ty.SELECTION,
               havoc=("which objects below the max",)))


def _apply_select_smallest(sel, P):
    if not sel.objs:
        return sel
    mn = min(o.size for o in sel.objs)
    return replace(sel, objs=[o for o in sel.objs if o.size == mn])


def _enc_select_smallest(si, so, P):
    return _subset(si, so) + [z3.Implies(si.n_obj >= 1, so.n_obj >= 1),
                              so.n_obj <= si.n_obj]


_reg(Primitive("select_smallest", "select", (), "keep only the smallest object(s)",
               _apply_select_smallest, _enc_select_smallest,
               in_type=Ty.SELECTION, out_type=Ty.SELECTION, havoc=("sizes",)))


def _apply_remove_largest(sel, P):
    if not sel.objs:
        return sel
    mx = max(o.size for o in sel.objs)
    return replace(sel, objs=[o for o in sel.objs if o.size != mx])


def _enc_remove_largest(si, so, P):
    return _subset(si, so) + [so.n_obj <= si.n_obj]


_reg(Primitive("remove_largest", "select", (), "drop the largest object(s)",
               _apply_remove_largest, _enc_remove_largest,
               in_type=Ty.SELECTION, out_type=Ty.SELECTION, havoc=("sizes",)))


# --------------------------------------------------------------------------- #
# Selection -> Selection: color selectors (EXACT via ocol)
# --------------------------------------------------------------------------- #

def _apply_select_color(sel, P):
    c = P["c"]
    return replace(sel, objs=[o for o in sel.objs if o.color == c])


def _enc_select_color(si, so, P):
    def branch(v):
        cs = [so.n_obj == si.ocol[v], so.ocol[v] == si.ocol[v]]
        cs += [so.ocol[d] == 0 for d in range(N_COLORS) if d != v]
        return cs
    return _subset(si, so) + _case(P["c"], _COLORS, branch)


_reg(Primitive("select_color", "select", (Param("c", _COLORS),),
               "keep only objects whose dominant color is c",
               _apply_select_color, _enc_select_color,
               in_type=Ty.SELECTION, out_type=Ty.SELECTION))


def _apply_remove_color(sel, P):
    c = P["c"]
    return replace(sel, objs=[o for o in sel.objs if o.color != c])


def _enc_remove_color(si, so, P):
    def branch(v):
        cs = [so.n_obj == si.n_obj - si.ocol[v], so.ocol[v] == 0]
        cs += [so.ocol[d] == si.ocol[d] for d in range(N_COLORS) if d != v]
        return cs
    return _subset(si, so) + _case(P["c"], _COLORS, branch)


_reg(Primitive("remove_color", "select", (Param("c", _COLORS),),
               "drop every object whose dominant color is c",
               _apply_remove_color, _enc_remove_color,
               in_type=Ty.SELECTION, out_type=Ty.SELECTION))


# --------------------------------------------------------------------------- #
# Selection -> Selection: recolor
# --------------------------------------------------------------------------- #

def _apply_recolor_all(sel, P):
    c = P["c"]
    return replace(sel, objs=[replace(o, recolor=c) for o in sel.objs])


def _enc_recolor_all(si, so, P):
    cc = P["c"]   # _COLORS[i]==i, so the param value IS the color
    # Every cell ends up either recolored (cc) or untouched background (bg_in),
    # so the palette collapses to {cc, bg_in} — sound regardless of which color
    # becomes the new most-frequent "background". n_obj / nonbg / osz are HAVOC:
    # recoloring can flip the bg designation, making them unpredictable.
    cs = [so.h == si.h, so.w == si.w]
    for d in range(N_COLORS):
        cs.append(z3.Implies(z3.And(cc != d, si.bg != d), so.cnt[d] == 0))
    return cs


_reg(Primitive("recolor_all", "recolor", (Param("c", _COLORS),),
               "recolor every object to color c",
               _apply_recolor_all, _enc_recolor_all,
               in_type=Ty.SELECTION, out_type=Ty.SELECTION,
               havoc=("object merges",)))


def _apply_recolor_largest(sel, P):
    if not sel.objs:
        return sel
    mx = max(o.size for o in sel.objs)
    return replace(sel, objs=[replace(o, recolor=P["c"]) if o.size == mx else o
                              for o in sel.objs])


def _enc_recolor_largest(si, so, P):
    # Only dims are safe: recoloring one object can flip the most-frequent-color
    # background, so n_obj / nonbg / hist / osz are all havoc (sound = weak here).
    return [so.h == si.h, so.w == si.w]


_reg(Primitive("recolor_largest", "recolor", (Param("c", _COLORS),),
               "recolor the largest object(s) to color c",
               _apply_recolor_largest, _enc_recolor_largest,
               in_type=Ty.SELECTION, out_type=Ty.SELECTION,
               havoc=("hist detail",)))
