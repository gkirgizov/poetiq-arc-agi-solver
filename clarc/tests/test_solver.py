"""Guided code-gen solver: G0 == A0 (no duals), G1 injects object constraints +
counterexample feedback. The generator keeps full code-gen power throughout."""

from __future__ import annotations

import numpy as np

from clarc.dual.object_dual import ObjectDual
from clarc.generator import StubGenerator
from clarc.solver import guided_solve
from clarc.types import ClarcConfig

# recolor task: swap colors 1<->2, shapes/positions preserved
TRAIN_IN = [[[1, 1, 0], [0, 0, 0], [0, 2, 2]], [[0, 1, 0], [2, 0, 0], [0, 0, 2]]]
TRAIN_OUT = [[[2, 2, 0], [0, 0, 0], [0, 1, 1]], [[0, 2, 0], [1, 0, 0], [0, 0, 1]]]
TEST_IN = [[[1, 2, 0], [0, 0, 1]]]

IDENTITY = "def transform(grid):\n    return grid"
SWAP = ("import numpy as np\ndef transform(grid):\n"
        "    g = np.array(grid); o = g.copy(); o[g == 1] = 2; o[g == 2] = 1; return o")


def test_object_dual_extracts_constraints_and_counterexamples():
    d = ObjectDual()
    d.extract([(np.array(i), np.array(o)) for i, o in zip(TRAIN_IN, TRAIN_OUT)])
    block = d.prompt_block()
    assert "remapped" in block and "1→2" in block        # the learned color swap, as guidance
    # identity output (no swap) violates the learned color map -> counterexample
    cx = d.refute([(np.array(TRAIN_IN[0]), np.array(TRAIN_IN[0]))])
    assert cx is not None and "object" in cx.invariant
    # the correct output satisfies it -> no counterexample
    cx2 = d.refute([(np.array(TRAIN_IN[0]), np.array(TRAIN_OUT[0]))])
    assert cx2 is None


def _cfg():
    return ClarcConfig(max_iterations=4, seed=0, shuffle_examples=False,
                       timeout_sandbox_s=10.0, problem_id="swap")


async def test_g0_equals_a0_no_duals():
    res = await guided_solve(train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
                             generator=StubGenerator([IDENTITY, SWAP]), config=_cfg(), arm="G0")
    log = res["clarc_log"]
    assert log["solved"] and res["iteration"] == 2
    assert log["has_constraints"] is False and log["n_counterexamples"] == 0  # pure A0
    assert log["n_duals"] == 0


async def test_g1_injects_constraints_and_counterexample_feedback():
    res = await guided_solve(train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
                             generator=StubGenerator([IDENTITY, SWAP]), config=_cfg(), arm="G1")
    log = res["clarc_log"]
    assert log["solved"] and res["iteration"] == 2
    assert log["has_constraints"] is True          # object dual injected the color map
    assert log["n_counterexamples"] >= 1           # identity attempt was refuted with a CE
