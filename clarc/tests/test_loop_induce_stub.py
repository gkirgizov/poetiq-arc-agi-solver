"""M6c (faithful, decomposed): end-to-end self-extension (arm E0) via a stub.

The task — recolor border-touching objects to 2, interior objects to 1 — is a
per-object rule keyed on an attribute (is_border) that NO built-in DSL primitive
exposes, so D2 (catalog only) can't express it. E0 induces a NARROW recolor rule
over the detected objects' attributes (not a free-form transform); the system
wraps it in the typed detect→rule→render scaffold, auto-derives its contract,
and solves the task. Offline, $0.
"""

from __future__ import annotations

import json

import numpy as np

from clarc.generator import StubGenerator
from clarc.loop import solve_task
from clarc.run import ARMS
from clarc.types import ClarcConfig

# border-touching objects -> 2, interior objects -> 1 (shapes/positions kept)
TRAIN_IN = [
    [[5, 5, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 3, 3, 0], [0, 0, 3, 3, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0], [0, 7, 7, 0], [8, 0, 0, 0]],
]
TRAIN_OUT = [
    [[2, 2, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0], [0, 1, 1, 0], [2, 0, 0, 0]],
]
TEST_IN = [[[9, 0, 0], [0, 4, 0], [0, 0, 0]]]   # 9 is border, 4 is interior
TEST_OUT = [[2, 0, 0], [0, 1, 0], [0, 0, 0]]

# the narrow per-object rule the model induces (NOT a whole-grid transform)
PROPOSAL = '''```python
KIND = "recolor"
DESCRIPTION = "border-touching objects -> 2, interior objects -> 1"
def color_of(o, scene):
    return 2 if o["is_border"] else 1
```'''
USE_IT = "```dsl\nind_brd_0()\n```"   # induced name = ind_{problem_id}_{attempt}


async def test_e0_induces_object_rule_and_solves():
    gen = StubGenerator([PROPOSAL, USE_IT])
    cfg = ClarcConfig(max_iterations=4, seed=0, shuffle_examples=False,
                      timeout_sandbox_s=10.0, prim_use_library=False,
                      problem_id="brd", **ARMS["E0"])
    res = await solve_task(train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
                           generator=gen, config=cfg, arm="E0")
    log = res["clarc_log"]
    assert log["n_induced"] >= 1, log
    ind = log["induced_prims"][0]
    assert ind["name"] == "ind_brd_0"
    assert "is_border" in ind["code"]          # the induced rule reasons over attrs
    assert "def color_of" in ind["code"] and "_detect" in ind["code"]  # typed scaffold
    assert log["solved"] is True
    assert json.loads(res["results"][0]["output"]) == TEST_OUT


async def test_d2_without_induction_cannot_solve():
    # No built-in primitive exposes 'is_border', so a catalog-only pipeline can't
    # express the rule → D2 doesn't solve (control showing induction unlocks it).
    gen = StubGenerator(["```dsl\nobjects(); recolor_all(1); render()\n```"])
    cfg = ClarcConfig(max_iterations=3, seed=0, shuffle_examples=False,
                      timeout_sandbox_s=10.0, **ARMS["D2"])
    res = await solve_task(train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
                           generator=gen, config=cfg, arm="D2")
    assert res["clarc_log"]["solved"] is False
    assert res["clarc_log"].get("n_induced", 0) == 0
