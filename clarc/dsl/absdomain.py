"""The abstract grid domain σ — the dual's shared vocabulary.

σ(grid) is a small record of decidable-logic-friendly observations (QF_LIA + Bool):
dims, the full color histogram (subsumes palette / nonbg / bg), connected-object
count, content-bbox dims, and five symmetry bits (the 4 classic ones + anti-
transpose, which closes the D4 orbit so rotations transform symmetry bits exactly).

Two sides of the dual:
  - `sigma_of(grid)`  — CONCRETE: evaluate σ on a real grid (reuses clarc.contracts.vocab
    extractors). Task facts are concrete observations, never inductions.
  - `ZState`          — SYMBOLIC: a bundle of z3 variables for one abstract state,
    with `wf()` well-formedness axioms (free theorems about any grid) and
    `eq_concrete()` fact-group equalities used as labeled assumptions so unsat
    cores can name exactly which observation a refutation rests on.

bg and nonbg are DERIVED symbolic variables (constrained by the most-frequent/
smallest-tie law) — primitives that transform `cnt` exactly get bg/nonbg
consistency for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import z3
from scipy import ndimage

from clarc.contracts.vocab import bg as bg_of
from clarc.contracts.vocab import content_bbox_shape, n_objects, symmetries
from clarc.common.geometry import label_components

N_COLORS = 10
MAX_DIM = 30
K_OBJ = 6   # number of largest objects tracked in the per-object summary
SYM_NAMES = ("mirror_h", "mirror_v", "rot180", "transpose", "anti_transpose")
FACT_GROUPS = ("dims", "hist", "objects", "bbox", "sym", "osz", "ocol",
               "oshape", "ohole", "oborder")


@dataclass(frozen=True)
class Sigma:
    """Concrete abstract state of one grid."""

    h: int
    w: int
    cnt: tuple[int, ...]        # color histogram, len 10
    n_obj: int
    bbox_h: int                 # (0, 0) for an all-background grid
    bbox_w: int
    sym: tuple[bool, ...]       # len 5, order = SYM_NAMES
    osz: tuple[int, ...]        # K_OBJ largest object sizes, sorted desc, 0-padded
    ocol: tuple[int, ...]       # objects-by-dominant-color count, len 10
    oshape: tuple[int, ...]     # objects by shape class, len N_SHAPE (disjoint, sums to n_obj)
    n_holed: int                # objects with >=1 enclosed background hole
    n_border: int               # objects touching the grid edge


# Disjoint per-object shape partition (every object lands in exactly one).
SHAPE_NAMES = ("dot", "hline", "vline", "rect", "other")
N_SHAPE = len(SHAPE_NAMES)


def _shape_class(cells: np.ndarray, bh: int, bw: int, size: int) -> int:
    """Classify one component's shape into a SHAPE_NAMES index."""
    if size == 1:
        return 0                                   # dot
    if bh == 1:
        return 1                                   # hline
    if bw == 1:
        return 2                                   # vline
    if bh >= 2 and bw >= 2 and size == bh * bw:
        return 3                                   # solid rectangle
    return 4                                        # other (L, cross, hollow, …)


def _object_features(g: np.ndarray):
    """All per-object summaries in one labeling pass."""
    bgc = bg_of(g)
    mask = g != bgc
    lab, n = label_components(mask, 1)
    H, W = g.shape
    sizes: list[int] = []
    ocol = [0] * N_COLORS
    oshape = [0] * N_SHAPE
    n_holed = n_border = 0
    for k in range(1, n + 1):
        cells = lab == k
        size = int(cells.sum())
        sizes.append(size)
        vals, counts = np.unique(g[cells], return_counts=True)
        ocol[int(vals[np.flatnonzero(counts == counts.max()).min()])] += 1
        rows = np.flatnonzero(cells.any(axis=1))
        cols = np.flatnonzero(cells.any(axis=0))
        bh, bw = int(rows[-1] - rows[0] + 1), int(cols[-1] - cols[0] + 1)
        oshape[_shape_class(cells, bh, bw, size)] += 1
        if ndimage.binary_fill_holes(cells).sum() > size:
            n_holed += 1
        if rows[0] == 0 or cols[0] == 0 or rows[-1] == H - 1 or cols[-1] == W - 1:
            n_border += 1
    sizes.sort(reverse=True)
    osz = tuple((sizes + [0] * K_OBJ)[:K_OBJ])
    return osz, tuple(ocol), tuple(oshape), n_holed, n_border


def sigma_of(g: np.ndarray) -> Sigma:
    g = np.asarray(g, dtype=int)
    h, w = g.shape
    cnt = tuple(int(np.sum(g == c)) for c in range(N_COLORS))
    bbox = content_bbox_shape(g) or (0, 0)
    syms = symmetries(g)
    anti = bool(g.shape[0] == g.shape[1] and np.array_equal(g, g[::-1, ::-1].T))
    sym = tuple([*("mirror_h" in syms, "mirror_v" in syms, "rot180" in syms,
                   "transpose" in syms), anti])
    osz, ocol, oshape, n_holed, n_border = _object_features(g)
    return Sigma(h=h, w=w, cnt=cnt, n_obj=n_objects(g),
                 bbox_h=int(bbox[0]), bbox_w=int(bbox[1]), sym=sym,
                 osz=osz, ocol=ocol, oshape=oshape,
                 n_holed=n_holed, n_border=n_border)


class ZState:
    """z3 variables for one abstract state (one position in one pair's trace)."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.h = z3.Int(f"{prefix}_h")
        self.w = z3.Int(f"{prefix}_w")
        self.area = z3.Int(f"{prefix}_area")
        self.cnt = [z3.Int(f"{prefix}_cnt{c}") for c in range(N_COLORS)]
        self.n_obj = z3.Int(f"{prefix}_nobj")
        self.bbox_h = z3.Int(f"{prefix}_bbh")
        self.bbox_w = z3.Int(f"{prefix}_bbw")
        self.sym = [z3.Bool(f"{prefix}_sym_{n}") for n in SYM_NAMES]
        self.osz = [z3.Int(f"{prefix}_osz{j}") for j in range(K_OBJ)]
        self.ocol = [z3.Int(f"{prefix}_ocol{c}") for c in range(N_COLORS)]
        self.oshape = [z3.Int(f"{prefix}_osh{k}") for k in range(N_SHAPE)]
        self.n_holed = z3.Int(f"{prefix}_nholed")
        self.n_border = z3.Int(f"{prefix}_nborder")
        # Derived (constrained in wf(), not free):
        self.bg = z3.Int(f"{prefix}_bg")
        self.nonbg = z3.Int(f"{prefix}_nonbg")

    # --- well-formedness: free theorems about ANY grid's σ -------------------
    def wf(self) -> list[z3.BoolRef]:
        cs: list[z3.BoolRef] = [
            self.h >= 1, self.h <= MAX_DIM, self.w >= 1, self.w <= MAX_DIM,
        ]
        # area == h*w, linearized by case-split on h (h is bounded).
        cs.append(z3.Or(*[z3.And(self.h == v, self.area == v * self.w)
                          for v in range(1, MAX_DIM + 1)]))
        cs += [c >= 0 for c in self.cnt]
        cs.append(z3.Sum(self.cnt) == self.area)
        # bg law: bg == b  <=>  cnt[b] maximal AND no smaller index ties it.
        cs.append(z3.And(self.bg >= 0, self.bg < N_COLORS))
        for b in range(N_COLORS):
            is_bg = z3.And(
                *[self.cnt[b] >= self.cnt[c] for c in range(N_COLORS) if c != b],
                *[self.cnt[c] < self.cnt[b] for c in range(b)],
            )
            cs.append((self.bg == b) == is_bg)
        cs.append(z3.Or(*[z3.And(self.bg == b, self.nonbg == self.area - self.cnt[b])
                          for b in range(N_COLORS)]))
        # bbox / object-count laws.
        cs += [
            self.bbox_h >= 0, self.bbox_h <= self.h,
            self.bbox_w >= 0, self.bbox_w <= self.w,
            (self.nonbg == 0) == (self.bbox_h == 0),
            (self.bbox_h == 0) == (self.bbox_w == 0),
            (self.nonbg == 0) == (self.n_obj == 0),
            self.n_obj >= 0, self.n_obj <= self.nonbg,
            # (nonbg <= bbox_h*bbox_w is true but nonlinear — deliberately omitted)
        ]
        # Symmetry-set closure under the D4 group (invariance under a and b
        # implies invariance under a∘b) + squareness for the diagonal axes.
        mh, mv, r180, tr, at = self.sym
        cs += [
            z3.Implies(z3.Or(tr, at), self.h == self.w),
            z3.Implies(z3.And(mh, mv), r180),
            z3.Implies(z3.And(mh, r180), mv),
            z3.Implies(z3.And(mv, r180), mh),
            z3.Implies(z3.And(tr, at), r180),
            z3.Implies(z3.And(tr, r180), at),
            z3.Implies(z3.And(at, r180), tr),
            # mirror + diagonal => full D4 orbit (rot90 invariance).
            z3.Implies(z3.And(mh, tr), z3.And(mv, r180, at)),
            z3.Implies(z3.And(mh, at), z3.And(mv, r180, tr)),
            z3.Implies(z3.And(mv, tr), z3.And(mh, r180, at)),
            z3.Implies(z3.And(mv, at), z3.And(mh, r180, tr)),
        ]
        # Per-object summary: sorted-descending, sizes positive iff enough objects,
        # bounded by total non-bg; one dominant color per object (all non-bg).
        cs += [o >= 0 for o in self.osz]
        cs += [self.osz[j] >= self.osz[j + 1] for j in range(K_OBJ - 1)]
        cs.append(z3.Sum(self.osz) <= self.nonbg)
        cs += [(self.osz[j] > 0) == (self.n_obj > j) for j in range(K_OBJ)]
        cs += [o >= 0 for o in self.ocol]
        cs.append(z3.Sum(self.ocol) == self.n_obj)
        for b in range(N_COLORS):
            cs.append(z3.Implies(self.bg == b, self.ocol[b] == 0))
        # Shape partition (disjoint, covers every object) + independent flags.
        cs += [o >= 0 for o in self.oshape]
        cs.append(z3.Sum(self.oshape) == self.n_obj)
        cs += [self.n_holed >= 0, self.n_holed <= self.n_obj,
               self.n_border >= 0, self.n_border <= self.n_obj]
        return cs

    # --- fact groups: named equality bundles for labeled assumptions ---------
    def eq_concrete(self, s: Sigma) -> dict[str, z3.BoolRef]:
        return {
            "dims": z3.And(self.h == s.h, self.w == s.w),
            "hist": z3.And(*[self.cnt[c] == s.cnt[c] for c in range(N_COLORS)]),
            "objects": self.n_obj == s.n_obj,
            "bbox": z3.And(self.bbox_h == s.bbox_h, self.bbox_w == s.bbox_w),
            "sym": z3.And(*[self.sym[i] == bool(s.sym[i]) for i in range(5)]),
            "osz": z3.And(*[self.osz[j] == s.osz[j] for j in range(K_OBJ)]),
            "ocol": z3.And(*[self.ocol[c] == s.ocol[c] for c in range(N_COLORS)]),
            "oshape": z3.And(*[self.oshape[k] == s.oshape[k] for k in range(N_SHAPE)]),
            "ohole": self.n_holed == s.n_holed,
            "oborder": self.n_border == s.n_border,
        }
