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

from dataclasses import asdict, is_dataclass
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
from clarc.instrument import IterRecord, RunLog
from clarc.spec import extract_spec, loo_trusted
from clarc.types import ClarcConfig, Generator


def _vscore(spec, trusted: set, test_in, cand_test) -> int:
    """# of LOO-TRUSTED train invariants that the candidate's TEST output violates
    (lower = better). This is the dual's only NON-REDUNDANT signal: the baseline never
    sees the test output, so a train-passing program whose test output breaks an
    invariant that held on every train pair AND survived leave-one-out is the overfit
    signature it cannot detect. A failed/unparsed test output is maximally penalized."""
    if any(g is None for g in cand_test):
        return 999
    if not trusted:
        return 0                                   # no trusted invariant ⇒ no signal, defer to order
    bad = spec.violations(test_in, cand_test)
    return sum(1 for c in bad if c.name in trusted)


def make_duals(arm: str) -> list[Dual]:
    from clarc.dual.object_dual import ObjectDual
    from clarc.dual.sigma_dual import SigmaDual
    if arm == "G1":
        return [ObjectDual()]                                    # ungated, vague CE (control)
    if arm == "G2":
        return [ObjectDual(), SigmaDual()]                       # + σ (control)
    if arm == "G3":                                              # confidence-gated + strong CE
        return [ObjectDual(gate=True, strong_ce=True, soft_prompt=True)]
    if arm == "G4":
        return [ObjectDual(gate=True, strong_ce=True, soft_prompt=True), SigmaDual()]
    return []   # G0 == A0


def _injected_names(duals: list[Dual]) -> list[str]:
    """Names of the invariants each dual actually injected into the prompt (so the
    per-iteration record knows what was `active`). Defensive: duals need not expose
    it. In G3+ this returns only the HARD-tier (injected) contract names."""
    names: list[str] = []
    for d in duals:
        f = getattr(d, "active_names", None)
        if callable(f):
            names += list(f())
    return names


def _ce_payload(cxs: list) -> dict:
    """Serialize the aggregated Counterexamples (incl. structured `violations`) for the log."""
    viols = [v for cx in cxs for v in (getattr(cx, "violations", None) or [])]
    return {"invariant": "+".join(cx.invariant for cx in cxs),
            "detail": " | ".join(cx.render() for cx in cxs),
            "viols": [asdict(v) if is_dataclass(v) else v for v in viols]}


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

    # Verified-selection arm (G5): keep sampling past the first train-pass and submit the
    # train-passing candidate whose TEST output respects the LOO-trusted train invariants
    # (the overfit filter the baseline can't run — it early-exits on the first train-pass).
    vselect = arm == "G5"
    vspec = extract_spec(train_in, train_out, cfg.problem_id) if vselect else None
    vtrusted = loo_trusted(train_in, train_out, cfg.problem_id) if vselect else set()
    passers: list[tuple] = []   # (train_res, test_res, cand_test_grids, train_score)

    runlog = RunLog(task_id=cfg.problem_id, arm=arm, seed=cfg.seed)
    active_names = _injected_names(duals)

    solutions: list[ARCAGISolution] = []
    best_score, best_result = -1.0, None
    last_train: list[RunResult] = [RunResult(success=False, output="", soft_score=0.0,
                                             error="no iterations", code="")]
    last_test: Optional[list[RunResult]] = None
    total_cost = total_ptok = total_ctok = 0
    n_cx = 0

    vlog: dict = {}   # verified-selection record (passer test outputs + vscores) for offline eval

    def finalize(result):
        result["cost_usd"] = total_cost          # type: ignore[typeddict-unknown-key]
        log = runlog.to_dict()
        log.update({"has_constraints": bool(constraints), "n_duals": len(duals),
                    "n_counterexamples": n_cx})
        if vlog:
            log["vselect"] = vlog
        result["clarc_log"] = log                # type: ignore[typeddict-unknown-key]
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
            runlog.add(IterRecord(iteration=it + 1, stage="gen_error", error=g.error,
                                  cost_usd=g.cost_usd, prompt_tokens=g.prompt_tokens,
                                  completion_tokens=g.completion_tokens, active=active_names))
            continue

        train_res, test_res = await _eval_on_train_and_test(
            g.code, train_in, train_out, test_in, timeout_s=cfg.timeout_sandbox_s)
        last_train, last_test = train_res, test_res
        cand_out = [parse_grid(r) for r in train_res]
        produced = [None if go is None else go.tolist() for go in cand_out]

        if all(r["success"] for r in train_res):
            runlog.solved = True
            if runlog.iterations_to_solve is None:
                runlog.iterations_to_solve = it + 1
            runlog.add(IterRecord(iteration=it + 1, stage="solved", soft_score=1.0,
                                  cost_usd=g.cost_usd, prompt_tokens=g.prompt_tokens,
                                  completion_tokens=g.completion_tokens,
                                  active=active_names, produced=produced))
            if not vselect:
                return finalize(ARCAGIResult(train_results=train_res, results=test_res,
                                             iteration=it + 1, prompt_tokens=total_ptok,
                                             completion_tokens=total_ctok))
            passers.append((train_res, test_res, [parse_grid(r) for r in test_res], 1.0))
            if len(passers) >= cfg.vselect_k:
                break
            continue   # keep sampling diverse train-passers to select among

        # counterexample-guided feedback: aggregate EVERY dual's witness (object-structure
        # CE above σ-invariant CE) so the next attempt repairs all of them, not just the first.
        cxs = []
        for d in duals:
            cx_d = d.refute(list(zip([np.asarray(x) for x in train_in], cand_out)))
            if cx_d is not None:
                cxs.append(cx_d)
        feedback, score = _build_feedback(train_res, train_in, train_out)
        ce = None
        if cxs:
            n_cx += 1
            ce = _ce_payload(cxs)
            ce_text = "\n".join(cx.render() for cx in cxs)
            feedback = "STRUCTURE VIOLATION (fix this first): " + ce_text + "\n\n" + feedback
        runlog.add(IterRecord(
            iteration=it + 1, stage=("structural" if cxs else "semantic"),
            conflict_type=("structural" if cxs else "semantic"), soft_score=score,
            violated=[cx.invariant for cx in cxs], ce=ce, produced=produced,
            cost_usd=g.cost_usd, prompt_tokens=g.prompt_tokens,
            completion_tokens=g.completion_tokens, active=active_names))
        solutions.append(ARCAGISolution(code=g.code, feedback=feedback, score=score))
        if score >= best_score:
            best_score = score
            best_result = ARCAGIResult(train_results=train_res, results=test_res,
                                       iteration=it + 1, prompt_tokens=total_ptok,
                                       completion_tokens=total_ctok)

    if vselect and passers:
        # Record every passer (in collection order) with its vscore + test output, so the
        # selection strategy can be evaluated OFFLINE within-task (first vs min-vscore vs
        # oracle) — isolating the selection effect from generator sampling noise.
        vsc = [_vscore(vspec, vtrusted, test_in, p[2]) for p in passers]
        pick = min(range(len(passers)), key=lambda i: vsc[i])   # min-vscore; ties keep order
        vlog.update({"n_passers": len(passers), "trusted": sorted(vtrusted), "picked": pick,
                     "passers": [{"vscore": vsc[i],
                                  "test_out": [g.tolist() if g is not None else None
                                               for g in passers[i][2]]}
                                 for i in range(len(passers))]})
        runlog.learned = [f"vselect: {len(passers)} passers, picked #{pick} "
                          f"(vscore={vsc[pick]} of {vsc}); trusted={sorted(vtrusted)}"]
        tr, te, _, _ = passers[pick]
        return finalize(ARCAGIResult(train_results=tr, results=te,
                                     iteration=runlog.iterations_to_solve or cfg.max_iterations,
                                     prompt_tokens=total_ptok, completion_tokens=total_ctok))

    if cfg.return_best_result and best_result is not None:
        return finalize(best_result)
    if last_test is None:
        last_test = [RunResult(success=False, output="", soft_score=0.0,
                               error="no valid solutions", code="")]
    return finalize(ARCAGIResult(train_results=last_train, results=last_test,
                                 iteration=cfg.max_iterations, prompt_tokens=total_ptok,
                                 completion_tokens=total_ctok))
