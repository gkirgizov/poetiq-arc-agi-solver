"""End-to-end D-arm loop test driven by StubGenerator (offline, $0).

The rot180 task again, but the stub emits DSL pipelines:
  1) tile(2,2)   — dims-impossible on a shape-preserving task: in D1/D2 it must
                   be REFUTED PRE-EXECUTION (zero sandbox calls that iteration);
  2) rot180()    — solves.

Also checks: D0 executes the bad candidate (try-and-drop), the duplicate memo,
dsl_invalid handling, and that learned clauses reach clarc_log.
"""

from __future__ import annotations

import json

from clarc.solve.generator import StubGenerator
from clarc.solve.loop import solve_task
from clarc.cli.run import ARMS
from clarc.common.types import ClarcConfig

TRAIN_IN = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
TRAIN_OUT = [[[4, 3], [2, 1]], [[8, 7], [6, 5]]]
TEST_IN = [[[9, 1], [2, 3]]]
TEST_EXPECTED = [[3, 2], [1, 9]]

BAD = "```dsl\ntile(2,2)\n```"
GOOD = "```dsl\nrot180()\n```"


def _cfg(arm: str, iters: int = 5) -> ClarcConfig:
    return ClarcConfig(max_iterations=iters, seed=0, shuffle_examples=False,
                       **ARMS[arm])


async def test_d2_refutes_pre_execution_then_solves():
    gen = StubGenerator([BAD, GOOD])
    result = await solve_task(train_in=TRAIN_IN, train_out=TRAIN_OUT,
                              test_in=TEST_IN, generator=gen, config=_cfg("D2"))
    log = result["clarc_log"]
    recs = log["records"]
    assert recs[0]["stage"] == "refuted" and recs[0]["refuted"]
    assert recs[0]["executed"] is False           # NEVER ran in the sandbox
    assert recs[0]["dsl_text"] == "tile(2, 2)"
    assert log["n_refuted"] == 1 and log["n_executed"] == 1
    assert log["n_clauses"] >= 1 and log["learned_clauses"]
    assert log["solved"] and result["iteration"] == 2
    assert json.loads(result["results"][0]["output"]) == TEST_EXPECTED


async def test_d0_executes_everything_try_and_drop():
    gen = StubGenerator([BAD, GOOD])
    result = await solve_task(train_in=TRAIN_IN, train_out=TRAIN_OUT,
                              test_in=TEST_IN, generator=gen, config=_cfg("D0"))
    log = result["clarc_log"]
    assert log["n_refuted"] == 0
    assert log["n_executed"] == 2                 # bad one executed and dropped
    assert log["solved"] and result["iteration"] == 2


async def test_d1_duplicate_memo():
    gen = StubGenerator([BAD, BAD, GOOD])
    result = await solve_task(train_in=TRAIN_IN, train_out=TRAIN_OUT,
                              test_in=TEST_IN, generator=gen, config=_cfg("D1"))
    log = result["clarc_log"]
    stages = [r["stage"] for r in log["records"]]
    # 1st BAD refuted; 2nd BAD is a dup (no second solver call, no execution).
    assert stages[0] == "refuted" and stages[1] == "dup"
    assert log["n_dup"] == 1 and log["solved"]


async def test_dsl_invalid_consumes_iteration_with_feedback():
    gen = StubGenerator(["no fence at all", "```dsl\nbogus(1)\n```", GOOD])
    result = await solve_task(train_in=TRAIN_IN, train_out=TRAIN_OUT,
                              test_in=TEST_IN, generator=gen, config=_cfg("D2"))
    log = result["clarc_log"]
    stages = [r["stage"] for r in log["records"]]
    assert stages[0] == "dsl_invalid" and stages[1] == "dsl_invalid"
    assert log["n_dsl_invalid"] == 2 and log["solved"]
