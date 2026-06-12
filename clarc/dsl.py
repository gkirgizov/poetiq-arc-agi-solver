"""The typed transformation DSL — the dual's concrete side and its z3 contracts.

Each primitive carries BOTH semantics of the dual:
  - `apply(grid, params)`  — concrete numpy implementation (raises DslRuntimeError
    on precondition failure: the typed C1 of `f : T1[C1] -> T2[C2]`);
  - `encode(si, so, P)`    — the relational contract R_f over abstract states
    (clarc.absdomain.ZState), z3 QF_LIA+Bool. Components NOT mentioned are havoc
    (over-approximated) BY OMISSION; `havoc` documents them. Every claim must be
    SOUND w.r.t. `apply` — enforced by the fuzz suite in tests/test_dsl.py, which
    asserts R_f(σ(g), σ(f(g)), params) is satisfiable for random grids/params.

Param encoding convention: every param is a z3 Int ranging over the INDEX of its
finite `values` domain; `encode` case-splits over concrete values, so all
arithmetic stays linear. Shared-across-pairs param consts (allocated by clarc.smt)
are what make cross-pair refutations possible.

A candidate is a linear `Pipeline` of steps. `compile_pipeline` emits a tiny
self-contained-via-import python `transform` so candidates flow through the
EXISTING sandbox/eval path unchanged.

D4 symmetry bookkeeping: for f in the dihedral group, out has symmetry s iff the
input has f⁻¹sf — an exact biconditional, so all 8 geometry primitives transform
the 5 symmetry bits exactly (zero havoc). For pointwise color ops symmetries are
monotone (sym_in => sym_out).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import z3
from scipy import ndimage

from clarc.absdomain import MAX_DIM, N_COLORS, ZState
from clarc.contracts import bg as bg_of
from clarc.contracts import nonbg_mask


class DslRuntimeError(RuntimeError):
    """A primitive's typed precondition failed on a concrete grid."""


# --------------------------------------------------------------------------- #
# Pipeline data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Param:
    name: str
    values: tuple  # finite domain (ints or strs); z3 var ranges over the INDEX


@dataclass(frozen=True)
class Step:
    name: str
    params: dict = field(default_factory=dict)

    def pretty(self) -> str:
        prim = REGISTRY[self.name]
        if prim.name == "recolor":
            pairs = [f"{c}->{self.params[f'pi{c}']}" for c in range(N_COLORS)
                     if self.params.get(f"pi{c}", c) != c]
            return f"recolor({', '.join(pairs)})"
        args = ", ".join(str(self.params[p.name]) for p in prim.params)
        return f"{self.name}({args})"


@dataclass(frozen=True)
class Pipeline:
    steps: tuple[Step, ...]

    def pretty(self) -> str:
        return "; ".join(s.pretty() for s in self.steps)


# --------------------------------------------------------------------------- #
# z3 encode helpers
# --------------------------------------------------------------------------- #

def _case(p: z3.ArithRef, values: tuple, branch: Callable) -> list[z3.BoolRef]:
    """OR over the param's finite domain; `branch(value) -> list[BoolRef]`."""
    return [z3.Or(*[z3.And(p == i, *branch(v)) for i, v in enumerate(values)])]


def _case2(pa, va, pb, vb, branch) -> list[z3.BoolRef]:
    return [z3.Or(*[z3.And(pa == i, pb == j, *branch(a, b))
                    for i, a in enumerate(va) for j, b in enumerate(vb)])]


def _eq_dims(si: ZState, so: ZState) -> list[z3.BoolRef]:
    return [so.h == si.h, so.w == si.w]


def _eq_cnt(si: ZState, so: ZState) -> list[z3.BoolRef]:
    return [so.cnt[c] == si.cnt[c] for c in range(N_COLORS)]


def _eq_bbox(si: ZState, so: ZState) -> list[z3.BoolRef]:
    return [so.bbox_h == si.bbox_h, so.bbox_w == si.bbox_w]


def _sym_perm(si: ZState, so: ZState, perm: tuple[int, ...]) -> list[z3.BoolRef]:
    """Exact: out bit i == in bit perm[i] (D4 conjugation)."""
    return [so.sym[i] == si.sym[perm[i]] for i in range(5)]


def _sym_monotone(si: ZState, so: ZState, bits=range(5)) -> list[z3.BoolRef]:
    return [z3.Implies(si.sym[i], so.sym[i]) for i in bits]


_ID = (0, 1, 2, 3, 4)            # mirror_h, mirror_v, rot180, transpose, anti
_SWAP_MIRRORS_DIAGS = (1, 0, 2, 4, 3)   # rot90 / rot270: mh<->mv, t<->at
_SWAP_DIAGS = (0, 1, 2, 4, 3)           # flip_h / flip_v: t<->at
_SWAP_MIRRORS = (1, 0, 2, 3, 4)         # transpose / anti_transpose: mh<->mv


# --------------------------------------------------------------------------- #
# Primitive definition
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Primitive:
    name: str
    family: str
    params: tuple[Param, ...]
    doc: str
    apply: Callable[[np.ndarray, dict], np.ndarray]
    encode: Callable[[ZState, ZState, dict], list[z3.BoolRef]]
    havoc: tuple[str, ...] = ()


REGISTRY: dict[str, Primitive] = {}


def _reg(p: Primitive) -> None:
    REGISTRY[p.name] = p


# --------------------------------------------------------------------------- #
# Geometry family — D4, all exact, zero havoc
# --------------------------------------------------------------------------- #

def _geo(name: str, fn: Callable[[np.ndarray], np.ndarray], swap_dims: bool,
         perm: tuple[int, ...], doc: str) -> None:
    def apply(g, params):
        return fn(g)

    def encode(si, so, P):
        cs = []
        if swap_dims:
            cs += [so.h == si.w, so.w == si.h,
                   so.bbox_h == si.bbox_w, so.bbox_w == si.bbox_h]
        else:
            cs += _eq_dims(si, so) + _eq_bbox(si, so)
        cs += _eq_cnt(si, so) + [so.n_obj == si.n_obj]
        cs += _sym_perm(si, so, perm)
        return cs

    _reg(Primitive(name, "geometry", (), doc, apply, encode))


_geo("identity", lambda g: g.copy(), False, _ID, "no-op")
_geo("rot90", lambda g: np.rot90(g, -1), True, _SWAP_MIRRORS_DIAGS,
     "rotate 90° clockwise")
_geo("rot180", lambda g: np.rot90(g, 2), False, _ID, "rotate 180°")
_geo("rot270", lambda g: np.rot90(g, 1), True, _SWAP_MIRRORS_DIAGS,
     "rotate 90° counter-clockwise")
_geo("flip_h", lambda g: g[:, ::-1], False, _SWAP_DIAGS,
     "mirror left-right")
_geo("flip_v", lambda g: g[::-1, :], False, _SWAP_DIAGS,
     "mirror top-bottom")
_geo("transpose", lambda g: g.T.copy(), True, _SWAP_MIRRORS,
     "transpose rows/columns")
_geo("anti_transpose", lambda g: g[::-1, ::-1].T.copy(), True, _SWAP_MIRRORS,
     "transpose across the anti-diagonal")


# --------------------------------------------------------------------------- #
# Size-changing structural family
# --------------------------------------------------------------------------- #

def _apply_crop_bbox(g, params):
    m = nonbg_mask(g)
    if not m.any():
        raise DslRuntimeError("crop_bbox: blank grid (no non-background content)")
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    return g[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].copy()


def _enc_crop_bbox(si, so, P):
    cs = [si.nonbg >= 1,                       # typed precondition C1
          so.h == si.bbox_h, so.w == si.bbox_w]
    for c in range(N_COLORS):
        cs.append(so.cnt[c] <= si.cnt[c])
        cs.append(z3.Implies(si.bg != c, so.cnt[c] == si.cnt[c]))
    cs.append(z3.Implies(so.bg == si.bg,
                         z3.And(so.bbox_h == so.h, so.bbox_w == so.w,
                                so.n_obj == si.n_obj)))
    return cs


_reg(Primitive("crop_bbox", "crop", (), "crop to the bounding box of non-background content",
               _apply_crop_bbox, _enc_crop_bbox, havoc=("sym",)))


_K4 = (1, 2, 3, 4)


def _apply_tile(g, params):
    return np.tile(g, (params["kh"], params["kw"]))


def _enc_tile(si, so, P):
    def branch(a, b):
        cs = [so.h == a * si.h, so.w == b * si.w]
        cs += [so.cnt[c] == a * b * si.cnt[c] for c in range(N_COLORS)]
        cs += [z3.If(si.bbox_h == 0, so.bbox_h == 0,
                     so.bbox_h == (a - 1) * si.h + si.bbox_h),
               z3.If(si.bbox_w == 0, so.bbox_w == 0,
                     so.bbox_w == (b - 1) * si.w + si.bbox_w),
               so.n_obj <= a * b * si.n_obj]
        cs += _sym_monotone(si, so, (0, 1, 2))
        if a == b:
            cs += _sym_monotone(si, so, (3, 4))
        return cs
    return _case2(P["kh"], _K4, P["kw"], _K4, branch)


_reg(Primitive("tile", "tile", (Param("kh", _K4), Param("kw", _K4)),
               "repeat the grid kh x kw times", _apply_tile, _enc_tile,
               havoc=("n_obj-exact", "sym-diag-when-kh!=kw")))


def _apply_scale(g, params):
    return np.kron(g, np.ones((params["kh"], params["kw"]), dtype=int))


def _enc_scale(si, so, P):
    def branch(a, b):
        cs = [so.h == a * si.h, so.w == b * si.w,
              so.n_obj == si.n_obj,
              so.bbox_h == a * si.bbox_h, so.bbox_w == b * si.bbox_w]
        cs += [so.cnt[c] == a * b * si.cnt[c] for c in range(N_COLORS)]
        cs += _sym_monotone(si, so, (0, 1, 2))
        if a == b:
            cs += _sym_monotone(si, so, (3, 4))
        return cs
    return _case2(P["kh"], _K4, P["kw"], _K4, branch)


_reg(Primitive("scale", "scale", (Param("kh", _K4), Param("kw", _K4)),
               "scale up: each cell becomes a kh x kw block", _apply_scale, _enc_scale,
               havoc=("sym-diag-when-kh!=kw",)))


_K24 = (2, 3, 4)


def _apply_downscale(g, params):
    a, b = params["kh"], params["kw"]
    h, w = g.shape
    if h % a or w % b:
        raise DslRuntimeError(f"downscale: dims {h}x{w} not divisible by {a}x{b}")
    out = np.zeros((h // a, w // b), dtype=int)
    for i in range(h // a):
        for j in range(w // b):
            block = g[i * a:(i + 1) * a, j * b:(j + 1) * b]
            vals, counts = np.unique(block, return_counts=True)
            m = counts.max()
            out[i, j] = int(vals[np.flatnonzero(counts == m).min()])
    return out


def _enc_downscale(si, so, P):
    def branch(a, b):
        cs = [si.h == a * so.h, si.w == b * so.w]   # divisibility as typing
        cs += [z3.Implies(so.cnt[c] > 0, si.cnt[c] > 0) for c in range(N_COLORS)]
        return cs
    return _case2(P["kh"], _K24, P["kw"], _K24, branch)


_reg(Primitive("downscale", "downscale", (Param("kh", _K24), Param("kw", _K24)),
               "shrink by block-mode: each kh x kw block becomes its dominant color",
               _apply_downscale, _enc_downscale,
               havoc=("hist", "objects", "bbox", "sym")))


_SIDES = ("top", "bottom", "left", "right")


def _apply_half(g, params):
    which = params["which"]
    h, w = g.shape
    if which in ("top", "bottom") and h < 2:
        raise DslRuntimeError("half: height < 2")
    if which in ("left", "right") and w < 2:
        raise DslRuntimeError("half: width < 2")
    return {"top": g[:h // 2], "bottom": g[h - h // 2:],
            "left": g[:, :w // 2], "right": g[:, w - w // 2:]}[which].copy()


def _enc_half(si, so, P):
    def branch(which):
        if which in ("top", "bottom"):
            dims = [z3.Or(si.h == 2 * so.h, si.h == 2 * so.h + 1), so.w == si.w]
        else:
            dims = [z3.Or(si.w == 2 * so.w, si.w == 2 * so.w + 1), so.h == si.h]
        return dims + [so.cnt[c] <= si.cnt[c] for c in range(N_COLORS)]
    return _case(P["which"], _SIDES, branch)


_reg(Primitive("half", "half", (Param("which", _SIDES),),
               "keep one half of the grid (floor split)", _apply_half, _enc_half,
               havoc=("objects", "bbox", "sym")))


_AXES = ("h", "v")


def _apply_concat_flip(g, params):
    if params["axis"] == "v":
        return np.concatenate([g, g[::-1, :]], axis=0)
    return np.concatenate([g, g[:, ::-1]], axis=1)


def _enc_concat_flip(si, so, P):
    def branch(axis):
        cs = [so.cnt[c] == 2 * si.cnt[c] for c in range(N_COLORS)]
        cs.append(so.n_obj <= 2 * si.n_obj)
        if axis == "v":
            cs += [so.h == 2 * si.h, so.w == si.w,
                   so.sym[1],                                   # gained mirror_v
                   z3.Implies(si.sym[0], so.sym[0]),            # mirror_h monotone
                   so.bbox_w == si.bbox_w,
                   z3.If(si.bbox_h == 0, so.bbox_h == 0, so.bbox_h >= 2 * si.bbox_h)]
        else:
            cs += [so.w == 2 * si.w, so.h == si.h,
                   so.sym[0],                                   # gained mirror_h
                   z3.Implies(si.sym[1], so.sym[1]),
                   so.bbox_h == si.bbox_h,
                   z3.If(si.bbox_w == 0, so.bbox_w == 0, so.bbox_w >= 2 * si.bbox_w)]
        return cs
    return _case(P["axis"], _AXES, branch)


_reg(Primitive("concat_flip", "concat", (Param("axis", _AXES),),
               "append the mirrored grid (axis=v: below; axis=h: to the right)",
               _apply_concat_flip, _enc_concat_flip,
               havoc=("rot180/diag sym", "objects-exact")))


_COLORS = tuple(range(N_COLORS))


def _apply_border(g, params):
    h, w = g.shape
    if h > MAX_DIM - 2 or w > MAX_DIM - 2:
        raise DslRuntimeError("border: grid too large to pad")
    return np.pad(g, 1, constant_values=params["c"])


def _enc_border(si, so, P):
    def branch(v):
        cs = [so.h == si.h + 2, so.w == si.w + 2]
        for d in range(N_COLORS):
            if d == v:
                cs.append(so.cnt[d] == si.cnt[d] + 2 * si.h + 2 * si.w + 4)
            else:
                cs.append(so.cnt[d] == si.cnt[d])
        return cs
    return _case(P["c"], _COLORS, branch)


_reg(Primitive("border", "border", (Param("c", _COLORS),),
               "add a 1-cell border of color c", _apply_border, _enc_border,
               havoc=("objects", "bbox", "sym")))


def _apply_compress_uniform(g, params):
    keep_r = [0] + [i for i in range(1, g.shape[0])
                    if not np.array_equal(g[i], g[i - 1])]
    g2 = g[keep_r]
    keep_c = [0] + [j for j in range(1, g2.shape[1])
                    if not np.array_equal(g2[:, j], g2[:, j - 1])]
    return g2[:, keep_c].copy()


def _enc_compress_uniform(si, so, P):
    cs = [so.h <= si.h, so.w <= si.w, so.h >= 1, so.w >= 1,
          so.n_obj == si.n_obj]
    cs += [(so.cnt[c] > 0) == (si.cnt[c] > 0) for c in range(N_COLORS)]
    cs += [so.cnt[c] <= si.cnt[c] for c in range(N_COLORS)]
    return cs


_reg(Primitive("compress_uniform", "compress", (),
               "drop consecutive duplicate rows and columns",
               _apply_compress_uniform, _enc_compress_uniform,
               havoc=("bbox", "sym", "hist-exact")))


# --------------------------------------------------------------------------- #
# Color family
# --------------------------------------------------------------------------- #

def _apply_recolor(g, params):
    lut = np.array([params.get(f"pi{c}", c) for c in range(N_COLORS)], dtype=int)
    return lut[g]


def _enc_recolor(si, so, P):
    pi = [P[f"pi{c}"] for c in range(N_COLORS)]
    cs = _eq_dims(si, so)
    for d in range(N_COLORS):
        cs.append(so.cnt[d] == z3.Sum(*[z3.If(pi[c] == d, si.cnt[c], 0)
                                        for c in range(N_COLORS)]))
    cs += _sym_monotone(si, so)
    inj = z3.And(*[z3.Implies(z3.And(si.cnt[c] > 0, si.cnt[d] > 0), pi[c] != pi[d])
                   for c in range(N_COLORS) for d in range(c + 1, N_COLORS)])
    bg_fixed = z3.Or(*[z3.And(si.bg == b, pi[b] == b) for b in range(N_COLORS)])
    cs.append(z3.Implies(z3.And(inj, bg_fixed, so.bg == si.bg),
                         z3.And(so.n_obj == si.n_obj,
                                so.bbox_h == si.bbox_h, so.bbox_w == si.bbox_w)))
    return cs


_reg(Primitive("recolor", "color",
               tuple(Param(f"pi{c}", _COLORS) for c in range(N_COLORS)),
               "apply a color map (args like 1->2, 5->0; unmapped colors unchanged)",
               _apply_recolor, _enc_recolor,
               havoc=("objects/bbox when map non-injective or moves bg",)))


def _apply_replace_color(g, params):
    out = g.copy()
    out[g == params["a"]] = params["b"]
    return out


def _enc_replace_color(si, so, P):
    a, b = P["a"], P["b"]
    cs = _eq_dims(si, so)
    cnt_a = z3.Sum(*[z3.If(a == c, si.cnt[c], 0) for c in range(N_COLORS)])
    for d in range(N_COLORS):
        cs.append(so.cnt[d] == z3.If(
            b == d,
            z3.If(a == d, si.cnt[d], si.cnt[d] + cnt_a),
            z3.If(a == d, 0, si.cnt[d]),
        ))
    cs += _sym_monotone(si, so)
    return cs


_reg(Primitive("replace_color", "color", (Param("a", _COLORS), Param("b", _COLORS)),
               "replace every cell of color a with color b",
               _apply_replace_color, _enc_replace_color,
               havoc=("objects", "bbox")))


def _apply_keep_only_color(g, params):
    out = np.full_like(g, bg_of(g))
    out[g == params["c"]] = params["c"]
    return out


def _enc_keep_only_color(si, so, P):
    def branch(v):
        uniform = z3.And(so.cnt[v] == so.area,
                         *[so.cnt[d] == 0 for d in range(N_COLORS) if d != v])
        keep = [so.cnt[v] == si.cnt[v]]
        for d in range(N_COLORS):
            if d != v:
                keep.append(z3.Implies(si.bg == d, so.cnt[d] == so.area - si.cnt[v]))
                keep.append(z3.Implies(si.bg != d, so.cnt[d] == 0))
        guarded = z3.Implies(z3.And(si.bg != v, so.bg == si.bg),
                             z3.And(so.bbox_h <= si.bbox_h, so.bbox_w <= si.bbox_w))
        return ([z3.If(si.bg == v, uniform, z3.And(*keep)), guarded]
                + _eq_dims(si, so) + _sym_monotone(si, so))
    return _case(P["c"], _COLORS, branch)


_reg(Primitive("keep_only_color", "color", (Param("c", _COLORS),),
               "keep only cells of color c; everything else becomes background",
               _apply_keep_only_color, _enc_keep_only_color,
               havoc=("objects",)))


def _apply_mask_nonbg(g, params):
    out = g.copy()
    out[nonbg_mask(g)] = params["c"]
    return out


def _enc_mask_nonbg(si, so, P):
    def branch(v):
        uniform = z3.And(so.cnt[v] == so.area,
                         *[so.cnt[d] == 0 for d in range(N_COLORS) if d != v])
        paint = [so.cnt[v] == si.nonbg]
        for d in range(N_COLORS):
            if d != v:
                paint.append(z3.Implies(si.bg == d, so.cnt[d] == si.cnt[d]))
                paint.append(z3.Implies(si.bg != d, so.cnt[d] == 0))
        guarded = z3.Implies(z3.And(si.bg != v, so.bg == si.bg),
                             z3.And(so.n_obj == si.n_obj,
                                    so.bbox_h == si.bbox_h, so.bbox_w == si.bbox_w))
        return ([z3.If(si.bg == v, uniform, z3.And(*paint)), guarded]
                + _eq_dims(si, so) + _sym_monotone(si, so))
    return _case(P["c"], _COLORS, branch)


_reg(Primitive("mask_nonbg", "color", (Param("c", _COLORS),),
               "paint all non-background cells with color c",
               _apply_mask_nonbg, _enc_mask_nonbg, havoc=()))


# --------------------------------------------------------------------------- #
# Object / content family
# --------------------------------------------------------------------------- #

def _apply_extract_largest_object(g, params):
    m = nonbg_mask(g)
    if not m.any():
        raise DslRuntimeError("extract_largest_object: blank grid")
    structure = ndimage.generate_binary_structure(2, 1)
    lab, n = ndimage.label(m, structure=structure)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    best = int(sizes.argmax())
    keep = lab == best
    rows = np.flatnonzero(keep.any(axis=1))
    cols = np.flatnonzero(keep.any(axis=0))
    out = g[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].copy()
    sub = keep[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    out[~sub] = bg_of(g)
    return out


def _enc_extract_largest_object(si, so, P):
    cs = [si.n_obj >= 1,
          so.h >= 1, so.w >= 1, so.h <= si.bbox_h, so.w <= si.bbox_w]
    for c in range(N_COLORS):
        cs.append(z3.Implies(si.bg != c, so.cnt[c] <= si.cnt[c]))
    cs.append(z3.Implies(so.bg == si.bg,
                         z3.And(so.n_obj == 1, so.bbox_h == so.h, so.bbox_w == so.w)))
    return cs


_reg(Primitive("extract_largest_object", "object", (),
               "crop to the largest connected object; erase the others",
               _apply_extract_largest_object, _enc_extract_largest_object,
               havoc=("hist-exact", "sym")))


def _apply_fill_holes(g, params):
    m = nonbg_mask(g)
    filled = ndimage.binary_fill_holes(m)
    out = g.copy()
    out[filled & ~m] = params["c"]
    return out


def _enc_fill_holes(si, so, P):
    def branch(v):
        cs = list(_eq_dims(si, so))
        cs.append(so.cnt[v] >= si.cnt[v])
        for d in range(N_COLORS):
            if d != v:
                cs.append(z3.Implies(si.bg != d, so.cnt[d] == si.cnt[d]))
                cs.append(z3.Implies(si.bg == d, so.cnt[d] <= si.cnt[d]))
        cs.append(z3.Implies(so.bg == si.bg,
                             z3.And(so.n_obj <= si.n_obj,
                                    so.bbox_h == si.bbox_h, so.bbox_w == si.bbox_w)))
        return cs + _sym_monotone(si, so)
    return _case(P["c"], _COLORS, branch)


_reg(Primitive("fill_holes", "object", (Param("c", _COLORS),),
               "fill enclosed background holes with color c",
               _apply_fill_holes, _enc_fill_holes, havoc=()))


_DIRS = ("up", "down", "left", "right")


def _apply_gravity(g, params):
    d = params["dir"]
    b = bg_of(g)
    out = np.full_like(g, b)
    if d in ("up", "down"):
        for j in range(g.shape[1]):
            vals = g[:, j][g[:, j] != b]
            if d == "down":
                out[g.shape[0] - len(vals):, j] = vals
            else:
                out[:len(vals), j] = vals
    else:
        for i in range(g.shape[0]):
            vals = g[i][g[i] != b]
            if d == "right":
                out[i, g.shape[1] - len(vals):] = vals
            else:
                out[i, :len(vals)] = vals
    return out


def _enc_gravity(si, so, P):
    def branch(d):
        cs = list(_eq_dims(si, so)) + _eq_cnt(si, so)
        if d in ("up", "down"):
            cs += [so.bbox_w == si.bbox_w, so.bbox_h <= si.bbox_h]
        else:
            cs += [so.bbox_h == si.bbox_h, so.bbox_w <= si.bbox_w]
        return cs
    return _case(P["dir"], _DIRS, branch)


_reg(Primitive("gravity", "object", (Param("dir", _DIRS),),
               "slide non-background cells to one side (per row/column)",
               _apply_gravity, _enc_gravity, havoc=("objects", "sym")))


# --------------------------------------------------------------------------- #
# Logic-stratum macros: split + cellwise boolean combine
# --------------------------------------------------------------------------- #

_OPS = ("and", "or", "xor", "diff")


def _combine(a: np.ndarray, b: np.ndarray, op: str, out_color: int) -> np.ndarray:
    ma, mb = a != 0, b != 0
    m = {"and": ma & mb, "or": ma | mb, "xor": ma ^ mb, "diff": ma & ~mb}[op]
    return np.where(m, out_color, 0).astype(int)


def _apply_split_binop(g, params, vertical: bool):
    h, w = g.shape
    if vertical:
        if h < 2:
            raise DslRuntimeError("split_binop_v: height < 2")
        k = h // 2
        a, b = g[:k], g[h - k:]
    else:
        if w < 2:
            raise DslRuntimeError("split_binop_h: width < 2")
        k = w // 2
        a, b = g[:, :k], g[:, w - k:]
    return _combine(a, b, params["op"], params["out_color"])


def _enc_split_binop(si, so, P, vertical: bool):
    if vertical:
        dims = [z3.Or(si.h == 2 * so.h, si.h == 2 * so.h + 1), so.w == si.w,
                si.h >= 2]
    else:
        dims = [z3.Or(si.w == 2 * so.w, si.w == 2 * so.w + 1), so.h == si.h,
                si.w >= 2]

    def branch(v):
        if v == 0:
            return [so.cnt[0] == so.area] + [so.cnt[d] == 0 for d in range(1, N_COLORS)]
        return [so.cnt[d] == 0 for d in range(N_COLORS) if d not in (0, v)]
    return dims + _case(P["out_color"], _COLORS, branch)


_reg(Primitive("split_binop_h", "split",
               (Param("op", _OPS), Param("out_color", _COLORS)),
               "split into left|right halves (drop odd middle), combine non-zero "
               "masks with op, paint out_color on black",
               lambda g, p: _apply_split_binop(g, p, vertical=False),
               lambda si, so, P: _enc_split_binop(si, so, P, vertical=False),
               havoc=("objects", "bbox", "sym")))

_reg(Primitive("split_binop_v", "split",
               (Param("op", _OPS), Param("out_color", _COLORS)),
               "split into top/bottom halves (drop odd middle), combine non-zero "
               "masks with op, paint out_color on black",
               lambda g, p: _apply_split_binop(g, p, vertical=True),
               lambda si, so, P: _enc_split_binop(si, so, P, vertical=True),
               havoc=("objects", "bbox", "sym")))


# --------------------------------------------------------------------------- #
# Pipeline execution / compilation
# --------------------------------------------------------------------------- #

def run_pipeline(p: Pipeline, grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    for step in p.steps:
        g = np.asarray(REGISTRY[step.name].apply(g, step.params), dtype=int)
        if g.ndim != 2 or g.shape[0] < 1 or g.shape[1] < 1:
            raise DslRuntimeError(f"{step.name}: produced an invalid grid")
        if g.shape[0] > MAX_DIM or g.shape[1] > MAX_DIM:
            raise DslRuntimeError(f"{step.name}: grid exceeds {MAX_DIM}px")
    return g


def apply_steps(steps: list[dict], grid: np.ndarray) -> np.ndarray:
    """Entry point for compiled candidates (runs inside the sandbox subprocess)."""
    p = Pipeline(tuple(Step(s["name"], dict(s["params"])) for s in steps))
    return run_pipeline(p, grid)


def compile_pipeline(p: Pipeline) -> str:
    """Emit a `transform` that replays the pipeline via clarc.dsl (the project is
    installed in the venv, so the sandbox subprocess can import it)."""
    steps = json.dumps([{"name": s.name, "params": s.params} for s in p.steps])
    return (
        "import json as _json\n"
        "from clarc.dsl import apply_steps as _apply_steps\n"
        f"_STEPS = _json.loads({steps!r})\n"
        "def transform(grid):\n"
        "    return _apply_steps(_STEPS, grid)\n"
    )


# --------------------------------------------------------------------------- #
# Param plumbing for the SMT layer
# --------------------------------------------------------------------------- #

def alloc_params(prim: Primitive, prefix: str) -> dict[str, z3.ArithRef]:
    return {p.name: z3.Int(f"{prefix}_{p.name}") for p in prim.params}


def param_domain(prim: Primitive, P: dict[str, z3.ArithRef]) -> list[z3.BoolRef]:
    return [z3.And(P[p.name] >= 0, P[p.name] < len(p.values)) for p in prim.params]


def param_index(prim: Primitive, name: str, value) -> int:
    spec = next(p for p in prim.params if p.name == name)
    if value not in spec.values:
        raise DslRuntimeError(f"{prim.name}: {name}={value!r} outside {spec.values}")
    return spec.values.index(value)


# --------------------------------------------------------------------------- #
# Prompt catalog (single source of truth for the LLM-facing reference)
# --------------------------------------------------------------------------- #

def render_catalog() -> str:
    lines = []
    for prim in REGISTRY.values():
        if prim.name == "recolor":
            sig = "recolor(a->b, ...)"
        elif prim.params:
            sig = f"{prim.name}({', '.join(p.name + '∈' + '|'.join(map(str, p.values)) for p in prim.params)})"
        else:
            sig = f"{prim.name}()"
        lines.append(f"  {sig:58s} {prim.doc}")
    return "\n".join(lines)
