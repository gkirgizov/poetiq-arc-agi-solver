"""The abstract grid domain σ — the dual's shared vocabulary.

σ(grid) is a small record of decidable-logic-friendly observations (QF_LIA + Bool):
dims, the full color histogram (subsumes palette / nonbg / bg), connected-object
count, content-bbox dims, and five symmetry bits (the 4 classic ones + anti-
transpose, which closes the D4 orbit so rotations transform symmetry bits exactly).

Two sides of the dual:
  - `sigma_of(grid)`  — CONCRETE: evaluate σ on a real grid (reuses clarc.contracts
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

from clarc.contracts import bg as bg_of
from clarc.contracts import content_bbox_shape, n_objects, symmetries

N_COLORS = 10
MAX_DIM = 30
K_OBJ = 6   # number of largest objects tracked in the per-object summary
SYM_NAMES = ("mirror_h", "mirror_v", "rot180", "transpose", "anti_transpose")
FACT_GROUPS = ("dims", "hist", "objects", "bbox", "sym", "osz", "ocol")


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


def _object_summary(g: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(osz, ocol): sizes of the K_OBJ largest 4-connected non-bg components
    (sorted desc, zero-padded) and a count of components by dominant color."""
    bgc = bg_of(g)
    mask = g != bgc
    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 1))
    sizes: list[int] = []
    ocol = [0] * N_COLORS
    for k in range(1, n + 1):
        cells = lab == k
        sizes.append(int(cells.sum()))
        vals, counts = np.unique(g[cells], return_counts=True)
        dom = int(vals[np.flatnonzero(counts == counts.max()).min()])
        ocol[dom] += 1
    sizes.sort(reverse=True)
    osz = tuple((sizes + [0] * K_OBJ)[:K_OBJ])
    return osz, tuple(ocol)


def sigma_of(g: np.ndarray) -> Sigma:
    g = np.asarray(g, dtype=int)
    h, w = g.shape
    cnt = tuple(int(np.sum(g == c)) for c in range(N_COLORS))
    bbox = content_bbox_shape(g) or (0, 0)
    syms = symmetries(g)
    anti = bool(g.shape[0] == g.shape[1] and np.array_equal(g, g[::-1, ::-1].T))
    sym = tuple([*("mirror_h" in syms, "mirror_v" in syms, "rot180" in syms,
                   "transpose" in syms), anti])
    osz, ocol = _object_summary(g)
    return Sigma(h=h, w=w, cnt=cnt, n_obj=n_objects(g),
                 bbox_h=int(bbox[0]), bbox_w=int(bbox[1]), sym=sym,
                 osz=osz, ocol=ocol)


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
        }
