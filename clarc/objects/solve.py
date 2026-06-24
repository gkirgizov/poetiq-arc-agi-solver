"""Solve-by-contract (M7b) — the logical transformation IS the program.

When the induced object-contracts are tight enough to be forward-deterministic
(every input object maps to one output object by a shared rule: shape kept,
position shifted, colors mapped), we don't need code-gen at all: we APPLY the
contracts to the test input to construct the answer. The recognizer (choice of
segmentation) + the SMT-induced contracts do the work; correctness is guaranteed
by re-verifying the construction reproduces every train output before we trust it.

Scope (M7b): same-dimension, uniform per-object transforms (recolor / shift /
shape-preserve), which is the bulk of the shape-preserving structural stratum.
Per-object-CONDITIONAL rules (color depends on the object) need the LLM to
propose conditional contracts — that is M7c.
"""

from __future__ import annotations

import numpy as np

from clarc.contracts.vocab import bg as bg_of
from clarc.objects.base import segment
from clarc.objects.smt import ObjectContracts, induce_object_contracts


def apply_contracts(g: np.ndarray, c: ObjectContracts) -> np.ndarray | None:
    """Forward-apply the induced global contracts to construct an output, or None
    if they don't determine a same-dimension construction."""
    g = np.asarray(g, dtype=int)
    H, W = g.shape
    bg = bg_of(g)
    objs = segment(g, c.segmentation)
    out = np.full((H, W), bg, dtype=int)
    dr, dc = (c.dr, c.dc) if "pos_shift" in c.used else (0, 0)
    for o in objs:
        sub = g[o.top:o.top + o.bh, o.left:o.left + o.bw].copy()
        m = o.mask[o.top:o.top + o.bh, o.left:o.left + o.bw]
        if "color_map" in c.used:
            sub = np.array([[c.pi[v] if 0 <= v < 10 else v for v in row] for row in sub])
        tr, tl = o.top + dr, o.left + dc
        rr, cc = np.nonzero(m)
        for di, dj in zip(rr, cc):
            r, col = tr + int(di), tl + int(dj)
            if 0 <= r < H and 0 <= col < W:
                out[r, col] = sub[di, dj]
    return out


def solve_by_contracts(train_pairs, test_inputs):
    """Induce object-contracts, build the forward construction, and ACCEPT it only
    if it reproduces every train output. Returns (predictions, contracts) or
    (None, contracts) when no contract construction reproduces train."""
    # same-dimension scope: every train pair must preserve grid dims
    if any(np.asarray(gi).shape != np.asarray(go).shape for gi, go in train_pairs):
        return None, None
    contracts = induce_object_contracts(train_pairs)
    if contracts is None:
        return None, contracts
    # verify the construction reproduces ALL train outputs (no false solves)
    for gi, go in train_pairs:
        pred = apply_contracts(gi, contracts)
        if pred is None or not np.array_equal(pred, np.asarray(go)):
            return None, contracts
    preds = [apply_contracts(np.asarray(t), contracts) for t in test_inputs]
    return preds, contracts
