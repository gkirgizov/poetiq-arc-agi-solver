"""Contract vocabulary — evaluation-based refinement predicates over grids.

A `Contract` is a named relational predicate `(input_grid, output_grid) -> bool`,
checked by EVALUATION on finite grids (no SMT). The vocabulary is deliberately
biased toward *informative invariants* that poetiq's per-cell diff does NOT make
explicit (color-histogram preservation, object-count preservation, pixelwise
recolor, content-bbox crop, output-is-subgrid, gained symmetry, ...).

A `Contract` is produced by a *builder*: given all train pairs, a builder returns
a concrete, parameter-bound Contract iff the pattern holds on EVERY pair, else
None. So an emitted Contract is, by construction, verified on the training data —
the soundness property the whole approach rests on. `Contract.check` re-evaluates
the same predicate (used in Phase 2 to test candidate outputs).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import ndimage

Pair = tuple[np.ndarray, np.ndarray]


# --------------------------------------------------------------------------- #
# Attribute extractors (a_T : Grid -> value)
# --------------------------------------------------------------------------- #

def shape(g: np.ndarray) -> tuple[int, int]:
    return (int(g.shape[0]), int(g.shape[1]))


def palette(g: np.ndarray) -> frozenset[int]:
    return frozenset(int(c) for c in np.unique(g))


def bg(g: np.ndarray) -> int:
    """Most-frequent color (ties -> smallest), the conventional ARC background."""
    vals, counts = np.unique(g, return_counts=True)
    m = counts.max()
    return int(vals[np.flatnonzero(counts == m).min()])


def color_hist(g: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(g, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def nonbg_mask(g: np.ndarray) -> np.ndarray:
    return g != bg(g)


def nonbg_count(g: np.ndarray) -> int:
    return int(nonbg_mask(g).sum())


def n_objects(g: np.ndarray, connectivity: int = 1) -> int:
    """Connected components of non-background cells (4-conn by default)."""
    structure = ndimage.generate_binary_structure(2, connectivity)
    _, n = ndimage.label(nonbg_mask(g), structure=structure)
    return int(n)


def content_bbox_shape(g: np.ndarray) -> Optional[tuple[int, int]]:
    m = nonbg_mask(g)
    if not m.any():
        return None
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    return (int(rows[-1] - rows[0] + 1), int(cols[-1] - cols[0] + 1))


def symmetries(g: np.ndarray) -> frozenset[str]:
    out = set()
    if np.array_equal(g, g[:, ::-1]):
        out.add("mirror_h")
    if np.array_equal(g, g[::-1, :]):
        out.add("mirror_v")
    if np.array_equal(g, g[::-1, ::-1]):
        out.add("rot180")
    if g.shape[0] == g.shape[1] and np.array_equal(g, g.T):
        out.add("transpose")
    return frozenset(out)


# --------------------------------------------------------------------------- #
# Contract object
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Contract:
    name: str
    descr: str                                   # human/LLM-readable
    check: Callable[[np.ndarray, np.ndarray], bool]  # (in, out) -> bool
    strength: int = 1                            # higher subsumes lower at same family
    family: str = ""                             # subsumption group; "" = unique

    def holds_on(self, pairs: list[Pair]) -> bool:
        return all(self.check(gi, go) for gi, go in pairs)


# --------------------------------------------------------------------------- #
# Builders: pair-list -> concrete verified Contract or None
# --------------------------------------------------------------------------- #

def _all(pairs: list[Pair], pred: Callable[[np.ndarray, np.ndarray], bool]) -> bool:
    try:
        return all(pred(gi, go) for gi, go in pairs)
    except Exception:
        return False


def _consistent_color_map(gi: np.ndarray, go: np.ndarray) -> bool:
    """Same shape and output[i,j] is a function of input[i,j] (pixelwise recolor)."""
    if gi.shape != go.shape:
        return False
    mapping: dict[int, int] = {}
    for a, b in zip(gi.ravel().tolist(), go.ravel().tolist()):
        if a in mapping:
            if mapping[a] != b:
                return False
        else:
            mapping[a] = b
    return True


def _is_subgrid(big: np.ndarray, small: np.ndarray) -> bool:
    H, W = big.shape
    h, w = small.shape
    if h > H or w > W:
        return False
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            if np.array_equal(big[r:r + h, c:c + w], small):
                return True
    return False


def build_shape_preserved(pairs):
    if _all(pairs, lambda gi, go: gi.shape == go.shape):
        return Contract("shape_preserved", "output.shape == input.shape",
                        lambda gi, go: gi.shape == go.shape, strength=1, family="shape")
    return None


def build_shape_eq_content_bbox(pairs):
    def pred(gi, go):
        b = content_bbox_shape(gi)
        return b is not None and shape(go) == b
    # Only claim a "crop to content" when it's a GENUINE crop on every pair
    # (bbox strictly smaller than the input); otherwise it's just shape_preserved.
    genuine = all(
        content_bbox_shape(gi) is not None and content_bbox_shape(gi) != shape(gi)
        for gi, _ in pairs
    )
    if genuine and _all(pairs, pred):
        return Contract("shape_eq_content_bbox",
                        "output.shape == bounding-box of input's non-background content (a crop)",
                        pred, strength=2, family="shape")
    return None


def build_shape_scaled(pairs):
    gi0, go0 = pairs[0]
    if gi0.shape[0] == 0 or gi0.shape[1] == 0:
        return None
    if go0.shape[0] % gi0.shape[0] or go0.shape[1] % gi0.shape[1]:
        return None
    kh, kw = go0.shape[0] // gi0.shape[0], go0.shape[1] // gi0.shape[1]
    if (kh, kw) == (1, 1):
        return None
    def pred(gi, go):
        return go.shape == (gi.shape[0] * kh, gi.shape[1] * kw)
    if _all(pairs, pred):
        return Contract("shape_scaled", f"output.shape == input.shape * ({kh}, {kw})",
                        pred, strength=2, family="shape")
    return None


def build_palette_preserved(pairs):
    if _all(pairs, lambda gi, go: palette(gi) == palette(go)):
        return Contract("palette_preserved", "output uses exactly the same color set as input",
                        lambda gi, go: palette(gi) == palette(go), strength=2, family="palette")
    return None


def build_palette_subset(pairs):
    if _all(pairs, lambda gi, go: palette(go) <= palette(gi)):
        return Contract("palette_subset", "output colors are a subset of input colors",
                        lambda gi, go: palette(go) <= palette(gi), strength=1, family="palette")
    return None


def build_bg_preserved(pairs):
    if _all(pairs, lambda gi, go: bg(gi) == bg(go)):
        return Contract("bg_preserved", "background (most-frequent color) is unchanged",
                        lambda gi, go: bg(gi) == bg(go), strength=1, family="bg")
    return None


def build_color_hist_preserved(pairs):
    if _all(pairs, lambda gi, go: color_hist(gi) == color_hist(go)):
        return Contract("color_hist_preserved",
                        "the multiset of colors is identical (same count of every color)",
                        lambda gi, go: color_hist(gi) == color_hist(go), strength=3, family="palette")
    return None


def build_nonbg_count_preserved(pairs):
    if _all(pairs, lambda gi, go: nonbg_count(gi) == nonbg_count(go)):
        return Contract("nonbg_count_preserved", "number of non-background cells is unchanged",
                        lambda gi, go: nonbg_count(gi) == nonbg_count(go), strength=1, family="count")
    return None


def build_object_count_preserved(pairs):
    if _all(pairs, lambda gi, go: n_objects(gi) == n_objects(go)):
        return Contract("object_count_preserved",
                        "number of connected non-background objects is unchanged",
                        lambda gi, go: n_objects(gi) == n_objects(go), strength=1, family="objects")
    return None


def build_recolor_only(pairs):
    def pred(gi, go):
        return gi.shape == go.shape and np.array_equal(nonbg_mask(gi), nonbg_mask(go))
    if _all(pairs, pred):
        return Contract("recolor_only",
                        "same shape and identical non-background mask (only colors change)",
                        pred, strength=2, family="recolor")
    return None


def build_pixelwise_recolor(pairs):
    # Require a SINGLE color map consistent ACROSS all pairs (the signature of a
    # genuine colormap task). This rejects e.g. rot180, where with repeated colors
    # the same input color maps to different outputs. The stored `check` is the
    # weaker per-pair "output is a recoloring of input" constraint we feed back.
    if any(gi.shape != go.shape for gi, go in pairs):
        return None
    mapping: dict[int, int] = {}
    changed = False
    for gi, go in pairs:
        for a, b in zip(gi.ravel().tolist(), go.ravel().tolist()):
            if a in mapping:
                if mapping[a] != b:
                    return None
            else:
                mapping[a] = b
            if a != b:
                changed = True
    if not changed:  # identity map -> not informative
        return None
    return Contract("pixelwise_recolor",
                    "same shape and output[i,j] depends only on input[i,j] (a fixed color map)",
                    _consistent_color_map, strength=3, family="recolor")


def build_output_subgrid(pairs):
    def pred(gi, go):
        return _is_subgrid(gi, go)
    if _all(pairs, pred):
        return Contract("output_is_subgrid",
                        "output appears verbatim as a contiguous sub-region of the input",
                        pred, strength=1, family="subgrid")
    return None


def build_output_symmetry(pairs):
    common = None
    for _, go in pairs:
        s = symmetries(go)
        common = s if common is None else (common & s)
    if not common:
        return None
    # Report only symmetries the *inputs* don't already always have (informative).
    in_common = None
    for gi, _ in pairs:
        s = symmetries(gi)
        in_common = s if in_common is None else (in_common & s)
    gained = common - (in_common or frozenset())
    report = gained or common
    names = sorted(report)
    sset = frozenset(report)
    def pred(gi, go):
        return sset <= symmetries(go)
    return Contract("output_symmetry", f"output always has symmetry: {', '.join(names)}",
                    pred, strength=1, family="symmetry")


BUILDERS: list[Callable[[list[Pair]], Optional[Contract]]] = [
    build_shape_preserved,
    build_shape_eq_content_bbox,
    build_shape_scaled,
    build_palette_preserved,
    build_palette_subset,
    build_bg_preserved,
    build_color_hist_preserved,
    build_nonbg_count_preserved,
    build_object_count_preserved,
    build_recolor_only,
    build_pixelwise_recolor,
    build_output_subgrid,
    build_output_symmetry,
]


# Subsumption: within a family, keep only the highest-strength contract.
# Cross-family drops where one strictly implies another.
_CROSS_DROP = {
    # color_hist_preserved (strength 3, family palette) implies these:
    "color_hist_preserved": {"palette_preserved", "palette_subset", "nonbg_count_preserved"},
    # pixelwise_recolor implies recolor_only and shape_preserved:
    "pixelwise_recolor": {"recolor_only", "shape_preserved"},
    "recolor_only": {"shape_preserved"},
    "shape_eq_content_bbox": set(),
    "shape_scaled": {"shape_preserved"},
}


def subsume(contracts: list[Contract]) -> list[Contract]:
    """Drop weaker contracts implied by stronger present ones."""
    by_name = {c.name: c for c in contracts}
    # within-family: keep max strength
    best_in_family: dict[str, Contract] = {}
    keep: set[str] = set()
    for c in contracts:
        if not c.family:
            keep.add(c.name)
            continue
        cur = best_in_family.get(c.family)
        if cur is None or c.strength > cur.strength:
            best_in_family[c.family] = c
    keep |= {c.name for c in best_in_family.values()}
    # cross-family implications
    dropped: set[str] = set()
    for name in list(keep):
        if name in _CROSS_DROP:
            dropped |= (_CROSS_DROP[name] & keep)
    keep -= dropped
    # preserve original order
    return [by_name[c.name] for c in contracts if c.name in keep and c.name in by_name]
