"""Confidence gate (objconf): mode-2-first tiering — drop vacuous, soften trivial
near-identity sets, reserve HARD for confident/forward-solving transforms. The three
structural-stratum anchors are the pre-registered acceptance test."""

from __future__ import annotations

import numpy as np

from clarc.objconf import _vacuous, forward_solves, gate, identity_like, tier_names
from clarc.objsmt import induce_object_contracts
from clarc.run import _load


def _swap(g):
    g = np.array(g)
    o = g.copy(); o[g == 1] = 2; o[g == 2] = 1
    return o


def test_gate_promotes_forward_solving_transform_to_hard():
    """A global recolor that forward-CONSTRUCTS the train outputs is the rule itself —
    inject it firmly (hard); the identity params (no shift/scale) are dropped."""
    g1, g2 = [[1, 1, 0], [0, 0, 0], [0, 2, 2]], [[0, 1, 0], [2, 0, 0], [0, 0, 2]]
    pairs = [(np.array(g1), _swap(g1)), (np.array(g2), _swap(g2))]
    c = induce_object_contracts(pairs)
    gate(c, pairs)
    assert forward_solves(c, pairs)
    assert "color_map" in tier_names(c, "hard")
    assert not identity_like(c, pairs)                       # a real transform forbids the no-op
    # vacuous identity params never reach the prompt
    assert _vacuous("size_scale", c) and "size_scale" in tier_names(c, "drop")
    assert _vacuous("pos_shift", c) and "pos_shift" in tier_names(c, "drop")


def test_anchor_1c56ad9f_hard_tier_emptied():
    """The over-constraining anchor: the dual induces a coarse near-identity block under
    by_row; the gate must leave NO hard tier (so it can't over-constrain, cf. the G1
    test-overfit). Everything injectable is demoted to soft, identity params dropped."""
    challenges, _ = _load("2024-eval")
    t = challenges["1c56ad9f"]
    pairs = [(np.array(e["input"]), np.array(e["output"])) for e in t["train"]]
    c = induce_object_contracts(pairs)
    gate(c, pairs)
    assert identity_like(c, pairs)                           # admits the no-op -> trivial
    assert tier_names(c, "hard") == []                       # ACCEPTANCE: no hard injection
    assert "color_map" in tier_names(c, "drop")              # color_map=id is vacuous


def test_anchor_noise_wins_induce_nothing():
    """The two 'G1 wins' induce no object-contract at all -> the gate is a pure no-op,
    so G3 == A0 for them (no regression possible). Confirms they were generation
    variance, not contract-driven (the Stage-1 finding)."""
    challenges, _ = _load("2024-eval")
    for tid in ("0e671a1a", "11e1fe23"):
        t = challenges[tid]
        pairs = [(np.array(e["input"]), np.array(e["output"])) for e in t["train"]]
        assert induce_object_contracts(pairs) is None
