"""E2 SYNTH — the dormant synthesizer, now wired: the LLM-free synthesis loop draws a
VERIFIED candidate from the clause-pruned feasible subspace (closes CEGIS criterion 4)."""

from __future__ import annotations

import numpy as np

from clarc.generator import StubGenerator
from clarc.loop import _blocked_from_clauses, solve_task
from clarc.run import ARMS
from clarc.types import ClarcConfig


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
