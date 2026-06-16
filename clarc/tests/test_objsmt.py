"""M7a: dynamic objects + object-correspondence SMT (matching + induced contracts)."""

from __future__ import annotations

import numpy as np

from clarc.objects import segment
from clarc.objsmt import check_output, induce_object_contracts


def test_segmentation_strategies_see_objects_differently():
    # the 2-bar and 3-bar are 4-adjacent, so connected4 MERGES them (2 objects:
    # the 1-L and the 2+3 blob); by_color keeps every color separate (3). This is
    # exactly why the segmentation must be chosen per task, not fixed.
    g = np.array([[1, 1, 0, 2], [1, 0, 0, 2], [0, 0, 3, 3]])
    assert len(segment(g, "connected4")) == 2
    assert len(segment(g, "by_color")) == 3
    # a disconnected color forms ONE object under by_color but TWO under connected
    g2 = np.array([[5, 0, 5], [0, 0, 0]])
    assert len(segment(g2, "by_color")) == 1
    assert len(segment(g2, "connected4")) == 2


def test_induce_recolor_contract_global_map():
    # each object keeps shape+position, colors swap 1<->2 (a global color map)
    def swap(g):
        o = g.copy(); o[g == 1] = 2; o[g == 2] = 1; return o
    g1 = np.array([[1, 1, 0], [0, 0, 0], [0, 2, 2]])
    g2 = np.array([[0, 1, 0], [0, 0, 0], [2, 2, 0]])
    pairs = [(g1, swap(g1)), (g2, swap(g2))]
    c = induce_object_contracts(pairs)
    assert c is not None
    assert "shape_exact" in c.used and "pos_preserved" in c.used
    assert "color_map" in c.used
    assert c.pi[1] == 2 and c.pi[2] == 1     # the induced global swap
    # a candidate output that recolors WRONG (1->3) must be refuted
    from clarc.objects import segment as seg
    wrong = g1.copy(); wrong[g1 == 1] = 3; wrong[g1 == 2] = 1
    assert not check_output(seg(g1, c.segmentation), seg(wrong, c.segmentation), c)
    # the correct output is accepted
    assert check_output(seg(g1, c.segmentation), seg(swap(g1), c.segmentation), c)


def test_induce_position_shift_contract():
    # every object moves down by 1 row (a shared geometric contract), colors kept
    def shift(g):
        o = np.zeros_like(g); o[1:] = g[:-1]; return o
    g1 = np.array([[3, 0, 0], [0, 0, 4], [0, 0, 0]])
    g2 = np.array([[0, 5, 0], [0, 0, 0], [0, 0, 0]])
    pairs = [(g1, shift(g1)), (g2, shift(g2))]
    c = induce_object_contracts(pairs)
    assert c is not None
    assert "pos_shift" in c.used and (c.dr, c.dc) == (1, 0)
    assert "shape_exact" in c.used and "color_preserved" in c.used
    from clarc.objects import segment as seg
    # an output shifted the WRONG way is refuted
    bad = np.zeros_like(g1); bad[:-1] = g1[1:]
    assert not check_output(seg(g1, c.segmentation), seg(bad, c.segmentation), c)


def test_solve_by_contracts_uniform_transform():
    """A uniform per-object transform (recolor 1<->2, shapes/positions kept) is
    SOLVED purely logically: induce contracts, apply forward to test, no LLM."""
    from clarc.objsolve import solve_by_contracts
    def swap(g):
        g = np.array(g); o = g.copy(); o[g == 1] = 2; o[g == 2] = 1; return o
    g1 = [[1, 1, 0], [0, 0, 0], [0, 2, 2]]
    g2 = [[0, 1, 0], [2, 0, 0], [0, 0, 2]]
    test = [[1, 2, 0], [0, 0, 1]]
    preds, c = solve_by_contracts([(g1, swap(g1)), (g2, swap(g2))], [np.array(test)])
    assert preds is not None and np.array_equal(preds[0], swap(test))
    assert c.pi[1] == 2 and c.pi[2] == 1 and c.pi[3] == 3   # free colors stay identity


def test_solve_by_contracts_rejects_conditional_task():
    """A per-object CONDITIONAL transform (largest->3, others->1) is NOT a uniform
    global contract. Two pairs where 'largest' and 'color' DISAGREE (color 1 is
    largest in one, smallest in the other) defeat any global color-map, so the
    uniform solver correctly declines (no false solve) — the gap M7c must close."""
    from clarc.objsolve import solve_by_contracts
    g1 = [[1, 1, 0], [1, 0, 2]]            # color 1 is largest (size 3)
    o1 = [[3, 3, 0], [3, 0, 1]]            # largest -> 3, other -> 1
    g2 = [[2, 2, 0], [2, 0, 1]]            # color 2 is largest, color 1 is smallest
    o2 = [[3, 3, 0], [3, 0, 1]]            # largest -> 3, other -> 1  (so 1->1 here, 1->3 there)
    preds, _ = solve_by_contracts([(np.array(g1), np.array(o1)),
                                   (np.array(g2), np.array(o2))], [np.array(g1)])
    assert preds is None       # no global color-map fits both -> uniform menu declines


def test_diagnose_output_localizes_color_violation():
    """The witness decoder pinpoints WHICH object violates WHICH term, with the target
    value — the actionable CEGIS signal (vs the old 'objects do not match')."""
    from clarc.objects import segment as seg
    from clarc.objsmt import diagnose_output

    def swap(g):
        o = g.copy(); o[g == 1] = 2; o[g == 2] = 1
        return o
    g1, g2 = np.array([[1, 1, 0], [0, 0, 0], [0, 2, 2]]), np.array([[0, 1, 0], [2, 0, 0], [0, 0, 2]])
    c = induce_object_contracts([(g1, swap(g1)), (g2, swap(g2))])
    wrong = swap(g1); wrong[g1 == 1] = 1                    # leave the color-1 object un-swapped
    viols = diagnose_output(seg(g1, c.segmentation), seg(wrong, c.segmentation), c)
    cm = [v for v in viols if v.term == "color_map"]
    assert cm and "2" in cm[0].expected and "color 1" in cm[0].actual


def test_diagnose_output_names_missing_object_on_count_mismatch():
    from clarc.objects import segment as seg
    from clarc.objsmt import diagnose_output

    def swap(g):
        o = g.copy(); o[g == 1] = 2; o[g == 2] = 1
        return o
    g1, g2 = np.array([[1, 1, 0], [0, 0, 0], [0, 2, 2]]), np.array([[0, 1, 0], [2, 0, 0], [0, 0, 2]])
    c = induce_object_contracts([(g1, swap(g1)), (g2, swap(g2))])
    empty = np.zeros((3, 3), dtype=int)                     # 0 objects vs the input's 2
    viols = diagnose_output(seg(g1, c.segmentation), seg(empty, c.segmentation), c)
    assert viols and all(v.kind == "missing" for v in viols)


def test_refute_requires_consistent_matching():
    # recolor-all-to-5 task: shapes/positions preserved, all colors -> 5
    def to5(g):
        o = g.copy(); o[g != 0] = 5; return o
    g1 = np.array([[1, 1, 0], [0, 0, 2], [3, 0, 0]])
    g2 = np.array([[0, 7, 0], [8, 0, 0], [0, 0, 9]])
    pairs = [(g1, to5(g1)), (g2, to5(g2))]
    c = induce_object_contracts(pairs)
    assert c is not None
    assert "color_map" in c.used and all(c.pi[k] == 5 for k in (1, 2, 3, 7, 8, 9))
    from clarc.objects import segment as seg
    # output where one object kept its original color -> no consistent matching
    bad = to5(g1); bad[g1 == 1] = 1
    assert not check_output(seg(g1, c.segmentation), seg(bad, c.segmentation), c)
