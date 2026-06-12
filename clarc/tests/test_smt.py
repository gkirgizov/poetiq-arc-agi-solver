"""Canonical refutation fixtures for the CHECK/SYNTH machinery.

The tile-ratio case is THE worked example of cross-pair param sharing: pairs
3x3->9x9 and 4x4->12x16 admit per-pair tile factors but no SHARED ones, so the
whole `tile` class refutes — invisible to any per-pair predicate.
"""

from __future__ import annotations

import numpy as np
import z3  # noqa: F401  (env sanity)

from clarc.absdomain import sigma_of
from clarc.dslparse import parse_pipeline
from clarc.smt import TaskSMT


def _facts(*pairs):
    return [(sigma_of(np.array(i)), sigma_of(np.array(o))) for i, o in pairs]


def _grid(h, w, fill=1, dot=True):
    g = np.full((h, w), fill, dtype=int)
    if dot:
        g[0, 0] = 3  # one nonbg cell so bbox/objects are nontrivial
    return g


def tile_ratio_facts():
    # 3x3 -> 9x9 (tile 3,3 fits) and 4x4 -> 12x16 (tile 3,4 fits) — but params
    # are shared across pairs, so `tile` as a single step cannot fit BOTH.
    return _facts(
        (_grid(3, 3), np.tile(_grid(3, 3), (3, 3))),
        (_grid(4, 4), np.tile(_grid(4, 4), (3, 4))),
    )


def test_tile_ratio_exact_and_class_refuted():
    smt = TaskSMT(tile_ratio_facts())
    p = parse_pipeline("tile(3,3)")
    r = smt.check_pipeline(p)
    assert r.refuted, f"expected refutation, got {r.status}"
    # Cores are minimal but not unique — assert a fact group participates,
    # not WHICH one (dims/hist/objects all witness this conflict).
    assert any(c.startswith("fact:") for c in r.core), r.core

    r_class = smt.check_pipeline(p, with_params=False)
    assert r_class.refuted  # the WHOLE tile class, any params
    assert any(c.startswith("slot:0:tile") for c in r_class.core)


def test_tile_ratio_single_pair_sat():
    # With only one pair, tile(3,3) fits — the conflict is genuinely cross-pair.
    smt = TaskSMT(tile_ratio_facts())
    assert not smt.check_pipeline(parse_pipeline("tile(3,3)"), drop_pair=1).refuted
    assert not smt.check_pipeline(parse_pipeline("tile(3,4)"), drop_pair=0).refuted


def test_correct_pipeline_sat():
    g1, g2 = _grid(3, 3), _grid(4, 5)
    facts = _facts((g1, np.rot90(g1, 2)), (g2, np.rot90(g2, 2)))
    smt = TaskSMT(facts)
    assert not smt.check_pipeline(parse_pipeline("rot180()")).refuted
    # and a wrong-shape candidate refutes
    assert smt.check_pipeline(parse_pipeline("tile(2,2)")).refuted


def test_recolor_inconsistent_map_refuted():
    # [[1]] -> [[2]] and [[1]] -> [[3]] cannot share one color map.
    facts = _facts(([[1]], [[2]]), ([[1]], [[3]]))
    smt = TaskSMT(facts)
    r = smt.check_pipeline(parse_pipeline("recolor(1->2)"), with_params=False)
    assert r.refuted


def test_synth_feasible_and_models():
    g = _grid(2, 2)
    facts = _facts((g, np.kron(g, np.ones((2, 2), dtype=int))))
    smt = TaskSMT(facts)
    assert smt.synth_feasible(2) is True
    models = smt.synth_models(2, max_models=8)
    names = {tuple(s.name for s in m.steps) for m in models}
    # shortest-first enumeration must surface a 1-step solution
    assert ("scale",) in names or ("tile",) in names, names
    assert all(len(m.steps) <= 2 for m in models)


def test_synth_depth_unsat():
    # 2x2 -> 17x3: no single primitive (nor any pair under the dims algebra)
    # reaches 17 rows from 2.
    facts = _facts((_grid(2, 2), _grid(17, 3)))
    smt = TaskSMT(facts)
    assert smt.synth_feasible(1) is False


def test_prim_impossible_anywhere_depth1():
    smt = TaskSMT(tile_ratio_facts())
    assert smt.prim_impossible_anywhere("tile", 1)
    # identity is also impossible (dims change), rot180 too:
    assert smt.prim_impossible_anywhere("identity", 1)
