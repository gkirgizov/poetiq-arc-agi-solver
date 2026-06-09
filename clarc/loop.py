"""The clarc solve loop (one task, one expert).

Ingredient flags on ClarcConfig select the ablation arm:
  A0  all off                         -> faithful poetiq baseline through the CLI
  A5  spec_inject                     -> static verified invariants in the prompt
  A1  spec_inject+clause_learn+clause_inject (+clause_prune for A2) -> CDCL over fixed basis
  A1L A1 + learn_contracts            -> CDCL + ADAPTIVE open-vocabulary induction

CDCL mapping: the verified contracts are the learned clauses; the contract pre-check
is conflict detection (cheapest-first, before the exact-equality check); the
LearnedStore is the clause DB; clause_prune is the backjump. Phase 4 adds genuine
learning: on a *semantic* conflict (the known basis is silent), the LLM proposes a
new invariant which is GATED by sandboxed verification (holds on all train pairs +
violated by the failure) before it joins the store and the cross-task library.
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

from clarc.analyze import classify_conflict, parse_grid, render_violation_feedback
from clarc.instrument import IterRecord, RunLog
from clarc.learn import propose_contract, verify_on_pairs
from clarc.library import ContractLibrary
from clarc.spec import extract_spec, loo_trusted
from clarc.store import LearnedStore
from clarc.types import ClarcConfig, Generator, LearnedContract


async def solve_task(
    *,
    train_in: list[list[list[int]]],
    train_out: list[list[list[int]]],
    test_in: list[list[list[int]]],
    generator: Generator,
    config: ClarcConfig,
    arm: str = "A0",
) -> ARCAGIResult:
    cfg = config
    rng = np.random.default_rng(cfg.seed)

    use_contracts = cfg.spec_inject or cfg.clause_learn or cfg.learn_contracts
    spec = extract_spec(train_in, train_out, cfg.problem_id) if use_contracts else None
    store = (LearnedStore(contracts=list(spec.contracts))
             if (spec and (cfg.clause_learn or cfg.learn_contracts)) else None)
    if store is not None and cfg.clause_prune:
        store.trusted = loo_trusted(train_in, train_out, cfg.problem_id)

    pairs_np = [(np.asarray(gi, dtype=int), np.asarray(go, dtype=int))
                for gi, go in zip(train_in, train_out)]
    sem_conflicts = 0

    # cross-task library: load + SOUND reuse (re-verify each candidate on THIS task).
    library: Optional[ContractLibrary] = None
    if cfg.learn_contracts and cfg.use_library and store is not None:
        library = ContractLibrary.load()
        for e in library.candidates():
            if len(store.learned) >= cfg.max_learned:
                break
            if await verify_on_pairs(e.code, pairs_np, timeout_s=cfg.timeout_sandbox_s):
                store.add_learned(LearnedContract(name=e.name, descr=e.descr,
                                                  code=e.code, origin="library"))

    runlog = RunLog(task_id=cfg.problem_id, arm=arm, seed=cfg.seed)

    solutions: list[ARCAGISolution] = []
    best_score = -1.0
    best_result: Optional[ARCAGIResult] = None
    last_train: list[RunResult] = [
        RunResult(success=False, output="", soft_score=0.0, error="No iterations ran", code="")
    ]
    last_test: Optional[list[RunResult]] = None

    total_cost = 0.0
    total_ptok = 0
    total_ctok = 0

    def finalize(result: ARCAGIResult) -> ARCAGIResult:
        runlog.learned = [lc.descr for lc in (store.learned if store else [])]
        if library is not None and store is not None:
            for lc in store.learned:
                library.record_use(lc.code, verified=True, solved=runlog.solved)
            library.save()
        result["cost_usd"] = total_cost              # type: ignore[typeddict-unknown-key]
        result["clarc_log"] = runlog.to_dict()       # type: ignore[typeddict-unknown-key]
        return result

    for it in range(cfg.max_iterations):
        # ---- build prompt ----
        example = _make_example(train_in, train_out, test_in)
        problem_str = format_problem(example, cfg.shuffle_examples, cfg.seed + it)
        message = _build_prompt(SOLVER_PROMPT_1, problem=problem_str)

        # contract injection: dynamic store (fixed + discovered invariants) when
        # learning/inducing, else the static spec when only spec_inject is on.
        if store is not None and (cfg.clause_inject or cfg.learn_contracts):
            block = store.render_for_prompt()
            if block:
                message += "\n\n" + block
        elif spec is not None and cfg.spec_inject and not spec.is_empty():
            message += "\n\n" + spec.render_for_prompt()

        # poetiq-style feedback memory (baseline channel, present in all arms)
        if solutions:
            mask = rng.uniform(size=len(solutions)) < cfg.selection_probability
            selected = [s for s, keep in zip(solutions, mask, strict=False) if keep]
            if selected:
                fb = create_examples(
                    selected, max_examples=cfg.max_solutions,
                    improving_order=cfg.improving_order,
                )
                message += "\n\n" + _build_prompt(FEEDBACK_PROMPT, feedback=fb)

        # ---- generate ----
        g = await generator.generate(message, seed=cfg.seed + it)
        total_cost += g.cost_usd
        total_ptok += g.prompt_tokens
        total_ctok += g.completion_tokens
        if g.code is None:
            stage = ("gen_timeout" if g.error == "generator-timeout"
                     else "parse_fail" if g.error in (None, "no-code-block")
                     else "gen_error")
            runlog.add(IterRecord(iteration=it + 1, stage=stage, error=g.error,
                                  cost_usd=g.cost_usd, prompt_tokens=g.prompt_tokens,
                                  completion_tokens=g.completion_tokens))
            continue
        code = g.code

        # ---- evaluate on train + test ----
        train_res, test_res = await _eval_on_train_and_test(
            code, train_in, train_out, test_in, timeout_s=cfg.timeout_sandbox_s
        )
        last_train, last_test = train_res, test_res

        # ---- exact check (early exit) ----
        if all(r["success"] for r in train_res):
            runlog.solved = True
            runlog.iterations_to_solve = it + 1
            runlog.add(IterRecord(iteration=it + 1, stage="solved", soft_score=1.0,
                                  cost_usd=g.cost_usd, prompt_tokens=g.prompt_tokens,
                                  completion_tokens=g.completion_tokens))
            return finalize(ARCAGIResult(
                train_results=train_res, results=test_res, iteration=it + 1,
                prompt_tokens=total_ptok, completion_tokens=total_ctok,
            ))

        # ---- contract pre-check: conflict detection (cheapest-first) ----
        violated = []
        conflict_type = None
        produced: list = []
        if spec is not None and (cfg.clause_learn or cfg.learn_contracts):
            produced = [parse_grid(r) for r in train_res]
            violated = spec.violations(train_in, produced)
            conflict_type = classify_conflict(violated)
            if store is not None and cfg.clause_learn:
                store.note(violated)

        # ---- feedback: poetiq diff, prefixed with the precise structured reason ----
        feedback, score = _build_feedback(train_res, train_in, train_out)
        if violated and cfg.clause_learn:
            feedback = render_violation_feedback(violated) + "\n\n" + feedback

        # ---- pruning (A2): a candidate violating a TRUSTED invariant is provably
        #      train-wrong; keep it out of the best/answer pool (the backjump).
        pruned = False
        harmful = False
        if cfg.clause_prune and store is not None and violated and any(
            c.name in store.trusted for c in violated
        ):
            pruned = True
            harmful = best_result is not None and score > best_score

        solutions.append(ARCAGISolution(code=code, feedback=feedback, score=score))

        if not pruned and score >= best_score:
            best_score = score
            best_result = ARCAGIResult(
                train_results=train_res, results=test_res, iteration=it + 1,
                prompt_tokens=None, completion_tokens=None,
            )

        # ---- adaptive induction (Phase 4): when the known basis is SILENT (semantic
        #      conflict), ask for a new invariant and GATE it by sandboxed verification.
        if (cfg.learn_contracts and store is not None and conflict_type == "semantic"
                and len(store.learned) < cfg.max_learned):
            sem_conflicts += 1
            if sem_conflicts % max(1, cfg.propose_every) == 0:
                lc = await propose_contract(
                    generator, pairs_np, produced,
                    idx=len(store.learned), seed=cfg.seed + it,
                    timeout_s=cfg.timeout_sandbox_s,
                )
                if lc is not None and store.add_learned(lc) and library is not None:
                    library.add(lc.name, lc.descr, lc.code)
                    library.save()

        runlog.add(IterRecord(
            iteration=it + 1, stage=(conflict_type or "wrong"), soft_score=score,
            violated=[c.name for c in violated], conflict_type=conflict_type,
            pruned=pruned, harmful_prune=harmful, cost_usd=g.cost_usd,
            prompt_tokens=g.prompt_tokens, completion_tokens=g.completion_tokens,
        ))

    # ---- no full solve: return best-so-far (poetiq semantics) ----
    if cfg.return_best_result and best_result is not None:
        best_result["prompt_tokens"] = total_ptok
        best_result["completion_tokens"] = total_ctok
        return finalize(best_result)

    if last_test is None:
        last_test = [RunResult(success=False, output="", soft_score=0.0,
                               error="Failed to generate any valid solutions.", code="")]
    return finalize(ARCAGIResult(
        train_results=last_train, results=last_test, iteration=cfg.max_iterations,
        prompt_tokens=total_ptok, completion_tokens=total_ctok,
    ))
