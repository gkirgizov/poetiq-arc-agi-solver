"""Phase 1 tests: contract vocabulary + spec extraction.

The load-bearing property is SOUNDNESS: every contract the extractor emits must
re-verify on all train pairs. We assert it on a synthetic task and by sweeping
real ARC-AGI-1 training tasks.
"""

import json
import os

import numpy as np
import pytest

from clarc.contracts import bg, n_objects, palette, symmetries
from clarc.spec import extract_spec

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRAIN = os.path.join(_REPO, "data", "arc-prize-2024", "arc-agi_training_challenges.json")


def _load_train():
    with open(_TRAIN, encoding="utf-8") as f:
        return json.load(f)


def _pairs(task):
    return ([ex["input"] for ex in task["train"]],
            [ex["output"] for ex in task["train"]])


# ---- attribute sanity ----

def test_attribute_extractors():
    g = np.array([[0, 0, 0], [1, 0, 2], [0, 0, 0]])  # two disconnected non-bg cells
    assert bg(g) == 0
    assert palette(g) == frozenset({0, 1, 2})
    assert n_objects(g) == 2  # (1,0) and (1,2) are not 4-connected
    sym = symmetries(np.array([[1, 2, 1], [3, 4, 3]]))
    assert "mirror_h" in sym


# ---- synthetic rot180 ----

def test_extract_rot180_synthetic():
    # Repeated colors so rot180 is genuinely NOT a pixelwise recolor.
    train_in = [[[1, 1], [2, 3]], [[4, 5], [5, 6]]]
    train_out = [[[3, 2], [1, 1]], [[6, 5], [5, 4]]]
    spec = extract_spec(train_in, train_out, "rot180")
    names = {c.name for c in spec.contracts}
    # rotation preserves the color multiset, the shape, and object count.
    assert "color_hist_preserved" in names
    assert "shape_preserved" in names
    # subsumption removed the weaker palette contracts implied by color_hist.
    assert "palette_preserved" not in names and "palette_subset" not in names
    # rot180 with repeated colors is NOT a pixelwise recolor.
    assert "pixelwise_recolor" not in names
    # soundness
    for c in spec.contracts:
        assert c.holds_on(spec.pairs), c.name


# ---- real colormap task: pixelwise recolor must be detected ----

def test_extract_colormap_real():
    task = _load_train()["0d3d703e"]  # a pure color substitution
    ti, to = _pairs(task)
    spec = extract_spec(ti, to, "0d3d703e")
    names = {c.name for c in spec.contracts}
    assert "pixelwise_recolor" in names
    # recolor_only and shape_preserved are subsumed by pixelwise_recolor.
    assert "recolor_only" not in names and "shape_preserved" not in names


# ---- soundness + coverage sweep over real tasks ----

def test_spec_soundness_sweep():
    tasks = _load_train()
    ids = list(tasks.keys())[:60]
    nonempty = 0
    for tid in ids:
        ti, to = _pairs(tasks[tid])
        spec = extract_spec(ti, to, tid)
        # SOUNDNESS: every emitted contract holds on every train pair.
        for c in spec.contracts:
            assert c.holds_on(spec.pairs), f"{tid}:{c.name}"
        if not spec.is_empty():
            nonempty += 1
    # Most tasks have at least one structural invariant.
    assert nonempty >= len(ids) // 2


# ---- violations() detects a shape-breaking candidate ----

def test_violations_detects_shape_break():
    train_in = [[[1, 1], [2, 3]]]
    train_out = [[[3, 2], [1, 1]]]  # rot180 w/ repeats -> shape_preserved holds
    spec = extract_spec(train_in, train_out, "t")
    assert any(c.name == "shape_preserved" for c in spec.contracts)
    # candidate produced a 1x1 output -> violates shape_preserved
    bad = spec.violations(train_in, [np.array([[9]])])
    assert any(c.name == "shape_preserved" for c in bad)
    # the exact correct output violates nothing
    good = spec.violations(train_in, [np.array([[3, 2], [1, 1]])])
    assert good == []
