"""Guided code-gen solver — poetiq's refinement loop AUGMENTED by the logical dual.

This is the literal poetiq A0 loop (SOLVER_PROMPT_1 + feedback memory + best-result)
with one orthogonal addition: a list of `Dual`s that (1) inject induced invariants
into the prompt to constrain the generator's search, and (2) turn invariant
violations into COUNTEREXAMPLE feedback (CEGIS-style repair). With `duals=[]` this
is byte-for-byte the A0 arm, so it is strictly ≥ A0 by construction; the duals can
only guide and prune.

Arms (run.py): G0 = no duals (== A0, control) ; G1 = ObjectDual ; G2 = ObjectDual
+ SigmaDual.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from arc_agi.prompts import FEEDBACK_PROMPT, SOLVER_PROMPT_1
from arc_agi.solve_coding import (
    _build_feedback,
    _build_prompt,
    _eval_on_train_and_test,
    _make_example,
    create_examples,
    format_problem,
)
from arc_agi.types import ARCAGIResult, ARCAGISolution, RunResult

from clarc.analyze import parse_grid
from clarc.dual.base import Dual
from clarc.types import ClarcConfig, Generator


def make_duals(arm: str) -> list[Dual]:
    from clarc.dual.object_dual import ObjectDual
    from clarc.dual.sigma_dual import SigmaDual
    if arm == "G1":
        return [ObjectDual()]
    if arm == "G2":
        return [ObjectDual(), SigmaDual()]
    return []   # G0 == A0


async def guided_solve(
    *,
    train_in: list[list[list[int]]],
    train_out: list[list[list[int]]],
    test_in: list[list[list[int]]],
    generator: Generator,
    config: ClarcConfig,
    arm: str = "G0",
) -> ARCAGIResult:
    cfg = config
    rng = np.random.default_rng(cfg.seed)
    pairs = [(np.asarray(gi, dtype=int), np.asarray(go, dtype=int))
             for gi, go in zip(train_in, train_out)]

    duals = make_duals(arm)
    for d in duals:
        d.extract(pairs)
    constraints = "\n\n".join(b for d in duals if (b := d.prompt_block()).strip())

    solutions: list[ARCAGISolution] = []
    best_score, best_result = -1.0, None
    last_train: list[RunResult] = [RunResult(success=False, output="", soft_score=0.0,
                                             error="no iterations", code="")]
    last_test: Optional[list[RunResult]] = None
    total_cost = total_ptok = total_ctok = 0
    n_gen_fail = n_cx = 0

    def finalize(result, solved, it):
        result["cost_usd"] = total_cost          # type: ignore[typeddict-unknown-key]
        result["clarc_log"] = {                  # type: ignore[typeddict-unknown-key]
            "task_id": cfg.problem_id, "arm": arm, "solved": solved,
            "iterations_to_solve": it, "n_gen_failures": n_gen_fail,
            "n_counterexamples": n_cx, "has_constraints": bool(constraints),
            "n_duals": len(duals), "total_cost_usd": round(total_cost, 4),
        }
        return result

    for it in range(cfg.max_iterations):
        problem_str = format_problem(_make_example(train_in, train_out, test_in),
                                     cfg.shuffle_examples, cfg.seed + it)
        message = _build_prompt(SOLVER_PROMPT_1, problem=problem_str)
        if constraints:
            message += "\n\n" + constraints
        if solutions:
            mask = rng.uniform(size=len(solutions)) < cfg.selection_probability
            selected = [s for s, keep in zip(solutions, mask, strict=False) if keep]
            if selected:
                fb = create_examples(selected, max_examples=cfg.max_solutions,
                                     improving_order=cfg.improving_order)
                message += "\n\n" + _build_prompt(FEEDBACK_PROMPT, feedback=fb)

        g = await generator.generate(message, seed=cfg.seed + it)
        total_cost += g.cost_usd
        total_ptok += g.prompt_tokens
        total_ctok += g.completion_tokens
        if g.code is None:
            n_gen_fail += 1
            continue

        train_res, test_res = await _eval_on_train_and_test(
            g.code, train_in, train_out, test_in, timeout_s=cfg.timeout_sandbox_s)
        last_train, last_test = train_res, test_res

        if all(r["success"] for r in train_res):
            return finalize(ARCAGIResult(train_results=train_res, results=test_res,
                                         iteration=it + 1, prompt_tokens=total_ptok,
                                         completion_tokens=total_ctok), True, it + 1)

        # counterexample-guided feedback: prepend the first invariant the candidate
        # violates so the next attempt repairs it specifically.
        cand_out = [parse_grid(r) for r in train_res]
        cx = None
        for d in duals:
            cx = d.refute(list(zip([np.asarray(x) for x in train_in], cand_out)))
            if cx:
                break
        feedback, score = _build_feedback(train_res, train_in, train_out)
        if cx is not None:
            n_cx += 1
            feedback = ("STRUCTURE VIOLATION (fix this first): " + cx.render()
                        + "\n\n" + feedback)
        solutions.append(ARCAGISolution(code=g.code, feedback=feedback, score=score))
        if score >= best_score:
            best_score = score
            best_result = ARCAGIResult(train_results=train_res, results=test_res,
                                       iteration=it + 1, prompt_tokens=total_ptok,
                                       completion_tokens=total_ctok)

    if cfg.return_best_result and best_result is not None:
        return finalize(best_result, False, None)
    if last_test is None:
        last_test = [RunResult(success=False, output="", soft_score=0.0,
                               error="no valid solutions", code="")]
    return finalize(ARCAGIResult(train_results=last_train, results=last_test,
                                 iteration=cfg.max_iterations, prompt_tokens=total_ptok,
                                 completion_tokens=total_ctok), False, None)
