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


def test_soft_prompt_header_defers_to_examples():
    pairs = [(np.array(i), np.array(o)) for i, o in zip(TRAIN_IN, TRAIN_OUT)]
    hard = ObjectDual()
    hard.extract(pairs)
    soft = ObjectDual(soft_prompt=True)
    soft.extract(pairs)
    # the body (the learned color map) is identical; only the header framing differs
    assert "MUST satisfy ALL" in hard.prompt_block()
    assert "TRUST THE EXAMPLES" in soft.prompt_block()
    assert "1→2" in soft.prompt_block()            # guidance preserved, just down-phrased
    assert "color_map" in hard.active_names()      # injected names surfaced for the log


def test_g1_control_vs_g3_gated_prompt():
    """G1 (ungated) keeps the verbatim hard 'MUST' header; G3 (gated + soft_prompt) on a
    forward-solving recolor uses the example-deferring header — never the absolute MUST."""
    pairs = [(np.array(i), np.array(o)) for i, o in zip(TRAIN_IN, TRAIN_OUT)]
    g1 = ObjectDual(); g1.extract(pairs)
    g3 = ObjectDual(gate=True, strong_ce=True, soft_prompt=True); g3.extract(pairs)
    assert "MUST satisfy ALL" in g1.prompt_block()          # control unchanged
    assert "MUST satisfy ALL" not in g3.prompt_block()
    assert "TRUST THE EXAMPLES" in g3.prompt_block()


async def test_g3_uses_structured_witness_ce():
    """G3's strong CE carries a STRUCTURED witness (the identity attempt on a recolor
    task gets a sound cell-diff); G1's vague CE has no structured violations."""
    g3 = await guided_solve(train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
                            generator=StubGenerator([IDENTITY, SWAP]), config=_cfg(), arm="G3")
    g1 = await guided_solve(train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
                            generator=StubGenerator([IDENTITY, SWAP]), config=_cfg(), arm="G1")
    assert g3["clarc_log"]["solved"] and g1["clarc_log"]["solved"]
    assert g3["clarc_log"]["records"][0]["ce"]["viols"]       # structured witness present
    assert not g1["clarc_log"]["records"][0]["ce"]["viols"]   # vague CE, no structured payload


async def test_g1_emits_per_iteration_records():
    """Instrumentation: the G-arm log now carries the per-iteration `records` the
    offline CE-actionability audit consumes (structured CE + produced grids)."""
    res = await guided_solve(train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
                             generator=StubGenerator([IDENTITY, SWAP]), config=_cfg(), arm="G1")
    recs = res["clarc_log"]["records"]
    assert len(recs) == 2                                   # refuted identity, then solving swap
    r0 = recs[0]
    assert r0["ce"] is not None and "object" in r0["ce"]["invariant"]
    assert r0["produced"] is not None                      # candidate output grids, for repair-tracking
    assert "color_map" in r0["active"]                     # injected constraint names
    assert recs[1]["stage"] == "solved"
