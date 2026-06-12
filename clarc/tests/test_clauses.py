"""Generalization ladder, clause matching/counting, prompt rendering."""

from __future__ import annotations

from clarc.clauses import Clause, ClauseStore
from clarc.dsl import REGISTRY
from clarc.dslparse import parse_pipeline
from clarc.smt import TaskSMT
from clarc.tests.test_smt import _facts, _grid, tile_ratio_facts


def test_ladder_tile_ratio_learns_general_clause():
    smt = TaskSMT(tile_ratio_facts())
    store = ClauseStore(smt, depth=2)
    p = parse_pipeline("tile(3,3)")
    r = smt.check_pipeline(p)
    assert r.refuted
    clause = store.learn_from_refutation(p, r.core)
    assert clause.kind in ("at_pos", "anywhere")
    assert clause.prim == "tile"
    assert "tile" in clause.nl and "training pairs" in clause.nl
    # The clause now fires syntactically on ANY pipeline using tile there:
    assert store.match(parse_pipeline("tile(2,2)"))
    assert not store.match(parse_pipeline("rot180()"))


def test_clause_match_and_count():
    n = len(REGISTRY)
    c_any = Clause("anywhere", "tile", None, (), "x", depth=3)
    assert c_any.matches(parse_pipeline("crop_bbox(); tile(2,2)"))
    assert not c_any.matches(parse_pipeline("crop_bbox()"))
    assert c_any.n_blocked(3) == n**3 - (n - 1) ** 3 > 1
    c_pos = Clause("at_pos", "scale", 0, (), "x")
    assert c_pos.matches(parse_pipeline("scale(2,2); rot90()"))
    assert not c_pos.matches(parse_pipeline("rot90(); scale(2,2)"))
    assert c_pos.n_blocked(3) == n**2


def test_exact_clause_when_class_sat():
    # rot180 task: recolor CLASS is feasible (identity map fits!), but the
    # specific wrong map 1->4 refutes -> exact-level clause.
    import numpy as np
    g = np.array([[1, 2], [2, 1]])
    facts = _facts((g, np.rot90(g, 2)))
    smt = TaskSMT(facts)
    store = ClauseStore(smt, depth=2)
    p = parse_pipeline("recolor(1->4)")
    r = smt.check_pipeline(p)
    assert r.refuted
    clause = store.learn_from_refutation(p, r.core)
    assert clause.kind == "exact"
    assert store.match(parse_pipeline("recolor(1->4)"))
    assert not store.match(parse_pipeline("recolor(2->2)"))


def test_concrete_block_and_render():
    smt = TaskSMT(tile_ratio_facts())
    store = ClauseStore(smt, depth=2)
    store.add_concrete_block(parse_pipeline("rot180()"))
    text = store.render_for_prompt()
    assert "PROVABLY REFUTED" in text and "rot180()" in text
    assert store.match(parse_pipeline("rot180()"))
    # dedupe: adding again doesn't duplicate
    store.add_concrete_block(parse_pipeline("rot180()"))
    assert len(store.clauses) == 1


def test_loo_robustness_annotation():
    """The tile clause hinges on HAVING both pairs (each alone is tile-SAT) —
    deduction is still valid, but loo_robust must be False; a dims-mismatch
    refutation that any single pair already forces is loo_robust=True."""
    smt = TaskSMT(tile_ratio_facts())
    store = ClauseStore(smt, depth=1)
    p = parse_pipeline("tile(3,3)")
    clause = store.learn_from_refutation(p, smt.check_pipeline(p).core)
    assert clause.loo_robust is False

    facts = _facts((_grid(3, 3), _grid(3, 3)), (_grid(4, 4), _grid(4, 4)))
    smt2 = TaskSMT(facts)  # shape-preserving task: every pair refutes tile alone
    store2 = ClauseStore(smt2, depth=1)
    p2 = parse_pipeline("tile(2,2)")
    clause2 = store2.learn_from_refutation(p2, smt2.check_pipeline(p2).core)
    assert clause2.loo_robust is True
