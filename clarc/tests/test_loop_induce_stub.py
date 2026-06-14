"""M6c: end-to-end self-extension (arm E0) driven by a StubGenerator.

The task — overlay the grid with its transpose (symmetrize across the main
diagonal) — is NOT expressible by any built-in primitive. With induction off the
loop can't solve it; with induction on, the stub 'LLM' proposes the missing
transform, its contract is auto-derived + gated, the SMT oracle flips to feasible,
and the loop solves the task using the induced primitive. Offline, $0.
"""

from __future__ import annotations

import numpy as np

from clarc.generator import StubGenerator
from clarc.loop import solve_task
from clarc.run import ARMS
from clarc.types import ClarcConfig


def _sym(g):
    g = np.asarray(g)
    return np.where(g != 0, g, g.T)


TRAIN_IN = [[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
            [[4, 0, 5], [0, 6, 0], [0, 0, 7]]]
TRAIN_OUT = [_sym(g).tolist() for g in TRAIN_IN]
TEST_IN = [[[8, 0, 0], [9, 1, 0], [0, 0, 2]]]

INDUCE_CODE = ('import numpy as np\nDESCRIPTION = "overlay the transpose"\n'
               'def transform(grid):\n'
               '    g = np.asarray(grid)\n'
               '    return np.where(g != 0, g, g.T)\n')
PROPOSAL = f"```python\n{INDUCE_CODE}\n```"
USE_IT = "```dsl\ninduced_0()\n```"


async def test_e0_induces_missing_primitive_and_solves():
    # The stub answers the induction prompt first (proposal), then every solve
    # iteration with a pipeline using the induced primitive.
    gen = StubGenerator([PROPOSAL, USE_IT])
    cfg = ClarcConfig(max_iterations=4, seed=0, shuffle_examples=False,
                      timeout_sandbox_s=10.0, prim_use_library=False, **ARMS["E0"])
    res = await solve_task(train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
                           generator=gen, config=cfg, arm="E0")
    log = res["clarc_log"]
    assert log["n_induced"] >= 1, log
    assert log["induced_prims"][0]["name"] == "induced_0"
    assert log["solved"] is True
    import json
    assert json.loads(res["results"][0]["output"]) == _sym(TEST_IN[0]).tolist()


async def test_d2_without_induction_cannot_solve():
    # Same task, arm D2 (no induction): the stub can only emit built-in pipelines,
    # none of which express the rule → not solved (control showing induction is
    # what unlocks it). The stub emits a plausible-but-wrong built-in pipeline.
    gen = StubGenerator(["```dsl\ntranspose()\n```"])
    cfg = ClarcConfig(max_iterations=3, seed=0, shuffle_examples=False,
                      timeout_sandbox_s=10.0, **ARMS["D2"])
    res = await solve_task(train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
                           generator=gen, config=cfg, arm="D2")
    assert res["clarc_log"]["solved"] is False
    assert res["clarc_log"].get("n_induced", 0) == 0
