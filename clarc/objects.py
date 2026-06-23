"""Dynamic object recognition (M7) — the recognizer's concrete side.

The system must INVENT how it sees objects per task, not inherit a fixed
4-connected segmenter. This module provides a small, extensible space of
segmentation strategies and a typed `Object` carrying the generating basis
`a_T` (geometric + color + shape attributes). The right segmentation for a task
is chosen by which one yields a clean input→output correspondence (see objsmt).

Each `Object` exposes integer/hashable attributes so the SMT layer (objsmt) can
reason over an explicit input↔output object MATCHING and the induced contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from clarc.contracts import bg as bg_of
from clarc.geometry import label_components

MAX_OBJECTS = 12   # segmentations producing more are rejected (SMT tractability)


def _canon_shape(mask_crop: np.ndarray) -> int:
    """D4-canonical hash of a binary shape (rotation/reflection invariant):
    the lexicographically smallest of its 8 dihedral images."""
    forms = []
    m = mask_crop
    for _ in range(4):
        forms.append(m); forms.append(m[:, ::-1])
        m = np.rot90(m)
    best = min((f.shape, f.tobytes()) for f in forms)
    return hash(best)


@dataclass
class Object:
    mask: np.ndarray            # bool, full-canvas
    color: int                  # dominant non-bg color
    palette: frozenset          # all colors present in the object
    size: int
    top: int
    left: int
    bh: int                     # bbox height / width
    bw: int
    shape_hash: int             # exact bbox mask pattern (position/color-blind)
    shape_canon: int            # D4-canonical (rotation/reflection invariant)
    shape: str                  # human-readable class: dot|hline|vline|rect|other
    n_holes: int                # enclosed background holes
    is_border: bool             # touches the grid edge

    @property
    def cells(self) -> np.ndarray:
        b = self.mask[self.top:self.top + self.bh, self.left:self.left + self.bw]
        return b


def _mk_object(g: np.ndarray, mask: np.ndarray) -> Object:
    rows = np.flatnonzero(mask.any(1))
    cols = np.flatnonzero(mask.any(0))
    t, l, b, r = int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])
    crop = mask[t:b + 1, l:r + 1]
    vals, cnts = np.unique(g[mask], return_counts=True)
    dom = int(vals[np.flatnonzero(cnts == cnts.max()).min()])
    size = int(mask.sum())
    bh, bw = b - t + 1, r - l + 1
    shape = ("dot" if size == 1 else "hline" if bh == 1 else "vline" if bw == 1
             else "rect" if size == bh * bw else "other")
    holed = int(ndimage.binary_fill_holes(mask).sum()) > size
    H, W = g.shape
    return Object(mask=mask, color=dom, palette=frozenset(int(v) for v in vals),
                  size=size, top=t, left=l, bh=bh, bw=bw,
                  shape_hash=hash((crop.shape, crop.tobytes())),
                  shape_canon=_canon_shape(crop), shape=shape, n_holes=int(holed),
                  is_border=bool(t == 0 or l == 0 or b == H - 1 or r == W - 1))


# --------------------------------------------------------------------------- #
# Segmentation strategies (the inducible space of "what is an object")
# --------------------------------------------------------------------------- #

def seg_connected(g: np.ndarray, conn: int = 1) -> list[Object]:
    """Connected components of non-background cells (conn=1: 4-, conn=2: 8-)."""
    bg = bg_of(g)
    lab, n = label_components(g != bg, conn)
    return [_mk_object(g, lab == k) for k in range(1, n + 1)]


def seg_connected_samecolor(g: np.ndarray, conn: int = 1) -> list[Object]:
    """Connected components where adjacency also requires equal color."""
    bg = bg_of(g)
    objs = []
    for c in np.unique(g):
        if c == bg:
            continue
        lab, n = label_components(g == c, conn)
        objs += [_mk_object(g, lab == k) for k in range(1, n + 1)]
    return objs


def seg_by_color(g: np.ndarray) -> list[Object]:
    """One object per non-background color (cells need not be connected)."""
    bg = bg_of(g)
    return [_mk_object(g, g == c) for c in np.unique(g) if c != bg and (g == c).any()]


def seg_by_row(g: np.ndarray) -> list[Object]:
    bg = bg_of(g)
    objs = []
    for i in range(g.shape[0]):
        m = np.zeros_like(g, dtype=bool)
        m[i] = g[i] != bg
        if m.any():
            objs.append(_mk_object(g, m))
    return objs


def seg_by_col(g: np.ndarray) -> list[Object]:
    bg = bg_of(g)
    objs = []
    for j in range(g.shape[1]):
        m = np.zeros_like(g, dtype=bool)
        m[:, j] = g[:, j] != bg
        if m.any():
            objs.append(_mk_object(g, m))
    return objs


SEGMENTERS = {
    "connected4": lambda g: seg_connected(g, 1),
    "connected8": lambda g: seg_connected(g, 2),
    "samecolor4": lambda g: seg_connected_samecolor(g, 1),
    "by_color": seg_by_color,
    "by_row": seg_by_row,
    "by_col": seg_by_col,
}


def segment(g: np.ndarray, strategy: str) -> list[Object]:
    return SEGMENTERS[strategy](np.asarray(g, dtype=int))
