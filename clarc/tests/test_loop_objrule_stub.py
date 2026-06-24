"""M7c (arm E1): end-to-end induction of a rule over DYNAMICALLY-segmented
objects (recolor/move/drop), via a stub.

Task: erase the marker color (1) and recolor every other object to it. This needs
(a) by_color segmentation (the recogniser's choice) and (b) the DROP action — the
fixed-4-conn recolor/select kinds (E0) cannot express it. The stub picks the
segmentation and writes one rule over object attributes; the system scaffolds
detect→rule→render, gates the contract, and solves. Offline, $0.
"""

from __future__ import annotations

import json

import numpy as np

from clarc.solve.generator import StubGenerator
from clarc.solve.loop import solve_task
from clarc.cli.run import ARMS
from clarc.common.types import ClarcConfig


def _rule(g):
    g = np.array(g)
    out = g.copy()
    out[(g != 0) & (g != 1)] = 1   # every non-marker object -> color 1
    out[g == 1] = 0                # erase the marker
    return out


TRAIN_IN = [
    [[1, 0, 0], [0, 5, 5], [0, 5, 0]],
    [[0, 0, 1], [7, 7, 0], [0, 7, 0]],
]
TRAIN_OUT = [_rule(g).tolist() for g in TRAIN_IN]
TEST_IN = [[[1, 0, 0], [0, 0, 8], [0, 8, 8]]]
TEST_OUT = _rule(TEST_IN[0]).tolist()

PROPOSAL = '''```python
SEGMENT = "by_color"
DESCRIPTION = "erase marker color 1; recolor every other object to 1"
def rule(o, scene):
    if o["color"] == 1:
        return None
    return 1
```'''
USE_IT = "```dsl\nind_mk_0()\n```"


async def test_e1_dynamic_object_rule_drops_and_recolors():
    gen = StubGenerator([PROPOSAL, USE_IT])
    cfg = ClarcConfig(max_iterations=4, seed=0, shuffle_examples=False,
                      timeout_sandbox_s=10.0, prim_use_library=False,
                      problem_id="mk", **ARMS["E1"])
    res = await solve_task(train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
                           generator=gen, config=cfg, arm="E1")
    log = res["clarc_log"]
    assert log["n_induced"] >= 1, log
    ind = log["induced_prims"][0]
    assert "by_color" in ind["code"] and "def rule" in ind["code"]   # dynamic seg + rule
    assert log["solved"] is True
    assert json.loads(res["results"][0]["output"]) == TEST_OUT
