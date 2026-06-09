"""End-to-end loop test driven by StubGenerator (offline).

Exercises the real sandbox subprocess + poetiq eval/feedback, but with scripted
generations: a wrong attempt (identity) followed by a correct one (rot180).
This is the Phase 0 proof that the loop wiring is sound.
"""

import json

from clarc.generator import StubGenerator
from clarc.loop import solve_task
from clarc.types import ClarcConfig

# A tiny rot180 task.
TRAIN_IN = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
TRAIN_OUT = [[[4, 3], [2, 1]], [[8, 7], [6, 5]]]
TEST_IN = [[[9, 1], [2, 3]]]
TEST_EXPECTED = [[3, 2], [1, 9]]

IDENTITY = "def transform(grid):\n    return grid"
ROT180 = "import numpy as np\ndef transform(grid):\n    return np.rot90(grid, 2)"


async def test_loop_solves_on_second_attempt():
    gen = StubGenerator([IDENTITY, ROT180])
    cfg = ClarcConfig(max_iterations=5, seed=0, shuffle_examples=False)
    result = await solve_task(
        train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
        generator=gen, config=cfg,
    )
    # Early-exit on the correct (2nd) generation.
    assert result["iteration"] == 2
    assert all(r["success"] for r in result["train_results"])
    # Test prediction is the rot180 of the test input.
    assert json.loads(result["results"][0]["output"]) == TEST_EXPECTED
    assert result.get("cost_usd") == 0.0


async def test_loop_returns_best_when_never_solved():
    # Only ever emits identity -> never solves rot180; must return best-so-far.
    gen = StubGenerator([IDENTITY])
    cfg = ClarcConfig(max_iterations=3, seed=0, shuffle_examples=False,
                      return_best_result=True)
    result = await solve_task(
        train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
        generator=gen, config=cfg,
    )
    assert result["iteration"] >= 1
    assert not all(r["success"] for r in result["train_results"])
    # A best result still carries a (failed) test attempt and token bookkeeping.
    assert result["results"] and "output" in result["results"][0]
    assert result["prompt_tokens"] == 0  # stub reports no tokens
