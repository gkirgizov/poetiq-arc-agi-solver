"""E2 SYNTH — the dormant synthesizer, now wired: the LLM-free synthesis loop draws a
VERIFIED candidate from the clause-pruned feasible subspace (closes CEGIS criterion 4)."""

from __future__ import annotations

import numpy as np

from clarc.solve.generator import StubGenerator
from clarc.solve.loop import _blocked_from_clauses, solve_task
from clarc.cli.run import ARMS
from clarc.common.types import ClarcConfig


def _flip_h(g):
    return np.array(g)[:, ::-1]


async def test_e2_synth_solves_dsl_task_without_llm():
    ti = [[[1, 2, 0], [0, 3, 0]], [[5, 0, 1], [2, 2, 0]]]
    to = [_flip_h(g).tolist() for g in ti]
    cfg = ClarcConfig(max_iterations=3, seed=0, timeout_sandbox_s=10.0,
                      problem_id="flip", **ARMS["E2"])
    res = await solve_task(train_in=ti, train_out=to, test_in=[[[7, 0, 0], [0, 8, 9]]],
                           generator=StubGenerator(["garbage, not DSL"]), config=cfg, arm="E2")
    log = res["clarc_log"]
    assert log["solved"]
    synth = [r for r in log["records"] if r["stage"] == "synth" and r.get("synth_seeded")]
    assert synth and synth[0]["dsl_text"] == "flip_h()"   # verified program, no LLM used


def test_param_search_recovers_solving_params():
    """synth proposes the right SKELETON but z3 picks one arbitrary param witness (the σ
    abstraction can't pin params), so it usually fails concretely. param_search brute-forces
    the skeleton's small concrete param space and recovers the solver — the lever that turns
    pure-synth's 0/40 into real solves (e.g. split_binop_h(and,1) witness -> (or,1) solver)."""
    from clarc.dsl.core import REGISTRY, Pipeline, Step, param_search, run_pipeline
    truth = Pipeline((Step("split_binop_h", {"op": "or", "out_color": 1}),))
    grids = [np.array([[0, 2, 0, 3], [1, 0, 5, 0]]),
             np.array([[7, 0, 0, 4], [0, 8, 9, 0]]),
             np.array([[1, 1, 0, 0], [0, 0, 2, 2]])]
    pairs = [(g, run_pipeline(truth, g, REGISTRY)) for g in grids]
    solver = param_search(["split_binop_h"], pairs, cap=100)
    assert solver is not None and solver.steps[0].params == {"op": "or", "out_color": 1}
    # huge param spaces (recolor's 10^10 colour map) exceed the cap and are skipped
    assert param_search(["recolor"], pairs, cap=100) is None


def test_blocked_from_clauses_extracts_lattice():
    class _C:
        def __init__(self, kind, prim, pos=None):
            self.kind, self.prim, self.pos = kind, prim, pos

    class _Store:
        clauses = [_C("anywhere", "recolor"), _C("at_pos", "crop_bbox", 0),
                   _C("at_pos", "tile", 0), _C("exact", "rot90();flip_h()")]
    ba, bat = _blocked_from_clauses(_Store())
    assert ba == {"recolor"}                              # anywhere → blocked everywhere
    assert bat == {0: {"crop_bbox", "tile"}}              # at_pos → blocked at that slot
