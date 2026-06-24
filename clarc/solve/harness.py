"""Shared runner core for the clarc CLIs (``run`` / ``experiment`` / ``ablate``).

The three entry points repeat two things; this factors them out so they stay in sync:

- ``make_generator(args)`` — build a generator factory from CLI flags (``--stub`` →
  offline identity generator; otherwise the Claude Code CLI generator with the model /
  thinking knobs present on ``args``).
- ``solve_one(...)`` — solve ONE ``(task, arm)`` cell, dispatching on the arm family:
  ``G*`` arms go through the guided-code-gen solver (poetiq loop + the logical dual),
  everything else through the contract/DSL CDCL loop. (This dispatch previously lived
  only in ``experiment.py``; routing it here gives ``run``/``ablate`` G-arm support too.)

Each CLI keeps its own argument parsing and report format — they aggregate differently
(single-run summary vs resumable failure-band vs seed-ensemble pass@2), which is a
genuine difference, not duplication.
"""

from __future__ import annotations

from typing import Callable

from arc_agi.io import build_kaggle_two_attempts

from clarc.solve.generator import ClaudeCodeGenerator, StubGenerator
from clarc.solve.loop import solve_task
from clarc.common.types import ClarcConfig

_IDENTITY = "def transform(grid):\n    return grid"


def make_generator(args) -> Callable[[], object]:
    """Return a zero-arg factory producing a fresh-or-shared Generator per CLI flags.

    ``--stub`` → a new identity ``StubGenerator`` each call (no API spend). Otherwise a
    shared ``ClaudeCodeGenerator`` configured from ``args``; the optional thinking knobs
    (``effort`` / ``max_thinking`` / ``thinking``) are passed only when present, so CLIs
    that don't expose them still work.
    """
    if getattr(args, "stub", False):
        return lambda: StubGenerator([_IDENTITY])
    kwargs: dict = {"model": args.model, "timeout_s": args.timeout}
    if getattr(args, "effort", None) is not None:
        kwargs["effort"] = args.effort
    if getattr(args, "max_thinking", None) is not None:
        kwargs["max_thinking_tokens"] = args.max_thinking
    if getattr(args, "thinking", None) is not None:
        kwargs["thinking"] = args.thinking
    gen = ClaudeCodeGenerator(**kwargs)
    return lambda: gen


async def solve_one(tid, task, arm, generator, *, iters, seed, timeout,
                    lean=False, model=None):
    """Solve ONE ``(task, arm)`` cell → ``(ARCAGIResult, kaggle_preds)``.

    ``G*`` arms route to ``clarc.solve.solver.guided_solve``; all others to
    ``clarc.solve.loop.solve_task``. The arm's ingredient flags come from ``clarc.cli.run.ARMS``.
    """
    from clarc.cli.run import ARMS
    ti = [e["input"] for e in task["train"]]
    to = [e["output"] for e in task["train"]]
    test_in = [e["input"] for e in task["test"]]
    cfg = ClarcConfig(model=model, max_iterations=iters, seed=seed, lean_prompt=lean,
                      request_timeout_s=timeout, problem_id=tid, **ARMS[arm])
    if arm.startswith("G"):
        from clarc.solve.solver import guided_solve
        res = await guided_solve(train_in=ti, train_out=to, test_in=test_in,
                                 generator=generator, config=cfg, arm=arm)
    else:
        res = await solve_task(train_in=ti, train_out=to, test_in=test_in,
                               generator=generator, config=cfg, arm=arm)
    preds = build_kaggle_two_attempts([res], test_in)
    return res, preds
