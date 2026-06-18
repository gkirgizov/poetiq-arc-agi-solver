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

from clarc.absdomain import sigma_of
from clarc.analyze import parse_grid, render_violation_feedback
from clarc.clauses import ClauseStore, _facts_nl
from clarc.dsl import REGISTRY, compile_pipeline, render_catalog
from clarc.dslparse import DslError, extract_dsl_block, parse_pipeline
from clarc.learn_prim import gate_code, induce_object_rule, induce_primitive
from clarc.prim_library import PrimLibrary
from clarc.instrument import IterRecord, RunLog
from clarc.learn import induced_violations, propose_contract, verify_on_pairs
from clarc.library import ContractLibrary
from clarc.smt import TaskSMT
from clarc.spec import extract_spec, loo_trusted
from clarc.store import LearnedStore
from clarc.types import ClarcConfig, Generator, LearnedContract


def _blocked_from_clauses(clause_store):
    """(blocked_anywhere, blocked_at) from the learned-clause lattice, so SYNTH
    enumerates ONLY from the un-pruned subspace — the piece that closes CEGIS
    criterion 4 (the generator draws from the clause-pruned feasible space)."""
    ba: set[str] = set()
    bat: dict[int, set[str]] = {}
    if clause_store is not None:
        for c in clause_store.clauses:
            if c.kind == "anywhere":
                ba.add(c.prim)
            elif c.kind == "at_pos" and c.pos is not None:
                bat.setdefault(c.pos, set()).add(c.prim)
    return ba, bat

# Terse single-attempt prompt: curbs the CLI's tendency to let strong models
# deliberate for many minutes on hard puzzles. Uses the same $$problem$$ slot.
LEAN_SOLVER_PROMPT = '''Solve this Abstract Reasoning (ARC) puzzle by writing Python code.

Write exactly one function `transform(grid: np.ndarray) -> np.ndarray` that maps each
input grid to its output grid, consistent with ALL the examples. Use numpy/scipy.
Output EXACTLY one ```python code block and nothing else. Make a single best attempt
— do not deliberate at length or explore many alternatives.

$$problem$$
'''

# DSL-emission prompt (D-arms, cfg.dsl_required): the model must answer in the
# typed pipeline language, so candidates are machine-checkable BEFORE execution.
DSL_SOLVER_PROMPT = '''Solve this Abstract Reasoning (ARC) puzzle by emitting ONE pipeline in the tiny DSL below.

A pipeline is a few steps separated by ';', applied LEFT TO RIGHT to the input grid,
producing the output grid. Available primitives (use ONLY these, args must be in-domain):

''' + render_catalog() + '''

Format rules:
- Answer with EXACTLY ONE fenced block tagged dsl, and nothing else after it.
- recolor args look like: recolor(1->2, 5->0)   (unmapped colors stay unchanged)
- Example of a complete, correctly formatted answer:

```dsl
crop_bbox(); scale(2,2)
```

Make a single best attempt — short pipelines (1-3 steps) are usually right.

$$problem$$
'''


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
    total_cost = 0.0
    total_ptok = 0
    total_ctok = 0

    # --- DSL ⇄ SMT dual (D-arms) -------------------------------------------
    task_smt: Optional[TaskSMT] = None
    clause_store: Optional[ClauseStore] = None
    seen_pipelines: set[str] = set()
    # Task-local primitive registry: a copy when we may induce (keeps induction
    # task-scoped + concurrency-safe), else the shared global.
    dsl_registry = dict(REGISTRY) if (cfg.dsl_required and cfg.induce_prims) else REGISTRY
    induced_prims: list = []   # InducedPrimitive admitted this task (for the log)
    prim_lib: Optional[PrimLibrary] = None
    if cfg.dsl_required and (cfg.z3_refute or cfg.z3_learn or cfg.induce_prims):
        facts = [(sigma_of(gi), sigma_of(go)) for gi, go in pairs_np]
        task_smt = TaskSMT(facts, timeout_ms=cfg.z3_timeout_ms, registry=dsl_registry)
        # loo_annotate off live: it's a diagnostics-only annotation and costs
        # several extra solver calls per learned clause (computable offline).
        clause_store = ClauseStore(task_smt, depth=cfg.dsl_depth_max,
                                   loo_annotate=False)

    async def _admit(ind, *, from_lib: bool) -> bool:
        """Add an admitted induced primitive to the task registry (and library)."""
        if ind is None or ind.name in dsl_registry:
            return False
        dsl_registry[ind.name] = ind.to_primitive()
        induced_prims.append(ind)
        if prim_lib is not None and not from_lib:
            prim_lib.add(ind)
        return True

    # --- self-extending DSL (M6, arm E0): induce new primitives when the library
    #     is insufficient, gated by the auto-derived-contract soundness check.
    #     Fires BEFORE the solve loop so the model can use the new primitives from
    #     iteration 0. NOTE: induction is PROACTIVE, not oracle-gated — the SMT
    #     oracle's SAT verdict cannot distinguish "a built-in works" from
    #     "abstraction too coarse to rule built-ins out" (a false-feasible), so
    #     gating induction on UNSAT would miss exactly the tasks that need it. The
    #     oracle verdict is recorded for analysis instead.
    if cfg.dsl_required and cfg.induce_prims and task_smt is not None:
        # Library reuse: re-derive the top-utility prims' contracts on THIS task
        # (sound reuse; cap to a few so the registry doesn't bloat). Reuse is
        # ADDITIVE — it never blocks fresh induction, because a stored prim being
        # SOUND on this task does not make it RELEVANT to this task.
        if cfg.prim_use_library:
            prim_lib = PrimLibrary.load()
            for e in prim_lib.candidates()[: max(1, cfg.max_induced_prims)]:
                ind = await gate_code(e.code, e.name, e.descr, pairs_np,
                                      seed=cfg.seed, timeout_s=cfg.timeout_sandbox_s)
                await _admit(ind, from_lib=True)
        # Fresh, task-specific induction: ALWAYS attempt (the task usually needs a
        # new rule), up to max attempts, stop after the first admitted prim. E1
        # (dynamic_objects) induces a rule over DYNAMICALLY-segmented objects
        # (recolor/move/drop); E0 uses the fixed-4-conn recolor/select kinds.
        for a in range(cfg.max_induced_prims):
            rep: dict = {}
            inducer = induce_object_rule if cfg.dynamic_objects else induce_primitive
            ind = await inducer(generator, pairs_np,
                                name=f"ind_{cfg.problem_id or 'x'}_{a}",
                                seed=cfg.seed + a, report=rep)
            total_cost += rep.get("cost_usd", 0.0)
            if await _admit(ind, from_lib=False):
                break

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

    def finalize(result: ARCAGIResult) -> ARCAGIResult:
        runlog.learned = [lc.descr for lc in (store.learned if store else [])]
        if library is not None and store is not None:
            for lc in store.learned:
                library.record_use(lc.code, verified=True, solved=runlog.solved)
            library.save()
        if prim_lib is not None:
            for ind in induced_prims:
                prim_lib.record_use(ind.code, verified=True, solved=runlog.solved)
            prim_lib.save()
        result["cost_usd"] = total_cost              # type: ignore[typeddict-unknown-key]
        log = runlog.to_dict()
        if store is not None:
            # Full induced/reused invariants WITH predicate code — the artifact the
            # per-iteration history refers to; descriptions alone can't be re-run.
            log["learned_contracts"] = [
                {"name": lc.name, "descr": lc.descr, "code": lc.code, "origin": lc.origin}
                for lc in store.learned
            ]
        if clause_store is not None:
            log["learned_clauses"] = [
                {"kind": c.kind, "prim": c.prim, "pos": c.pos, "nl": c.nl,
                 "loo_robust": c.loo_robust, "core": list(c.core)}
                for c in clause_store.clauses
            ]
            log["n_clauses"] = len(clause_store.clauses)
        if cfg.induce_prims:
            log["induced_prims"] = [
                {"name": ind.name, "descr": ind.descr, "code": ind.code,
                 "contract": ind.contract.render()} for ind in induced_prims
            ]
            log["n_induced"] = len(induced_prims)
        result["clarc_log"] = log                    # type: ignore[typeddict-unknown-key]
        return result

    synth_pipes: list = []          # E2: current SYNTH-feasible, clause-pruned skeletons
    synth_nclauses = -1             # forces an initial SYNTH on iteration 0

    for it in range(cfg.max_iterations):
        # ---- build prompt ----
        example = _make_example(train_in, train_out, test_in)
        problem_str = format_problem(example, cfg.shuffle_examples, cfg.seed + it)
        base_prompt = (DSL_SOLVER_PROMPT if cfg.dsl_required
                       else LEAN_SOLVER_PROMPT if cfg.lean_prompt else SOLVER_PROMPT_1)
        message = _build_prompt(base_prompt, problem=problem_str)

        # newly INDUCED primitives (E0): flag them so the model composes with them.
        if induced_prims:
            lines = ["NEWLY AVAILABLE PRIMITIVES (induced for this task — prefer them):"]
            lines += [f"  {ind.name}()  — {ind.descr}" for ind in induced_prims]
            message += "\n\n" + "\n".join(lines)

        # learned refutation clauses (D2): machine-checked impossibilities.
        if clause_store is not None and cfg.z3_inject:
            block = clause_store.render_for_prompt(cfg.max_clauses)
            if block:
                message += "\n\n" + block

        # SYNTH (E2): enumerate skeletons from the clause-pruned FEASIBLE subspace
        # (closes criterion 4 — generation is drawn from the pruned space). A skeleton
        # that concretely reproduces train is a VERIFIED solve (no LLM needed); the rest
        # seed the prompt. Re-synth only when the clause lattice has strengthened.
        if cfg.synth_seed and task_smt is not None:
            nc = len(clause_store.clauses) if clause_store is not None else 0
            if nc != synth_nclauses:
                synth_nclauses = nc
                ba, bat = _blocked_from_clauses(clause_store)
                synth_pipes = task_smt.synth_models(cfg.dsl_depth_max, max_models=cfg.synth_k,
                                                    blocked_anywhere=ba, blocked_at=bat)
                for sp in synth_pipes:
                    sp_text = sp.pretty()
                    if sp_text in seen_pipelines:
                        continue
                    seen_pipelines.add(sp_text)
                    s_tr, s_te = await _eval_on_train_and_test(
                        compile_pipeline(sp, dsl_registry), train_in, train_out, test_in,
                        timeout_s=cfg.timeout_sandbox_s)
                    if all(r["success"] for r in s_tr):
                        runlog.solved = True
                        runlog.iterations_to_solve = it + 1
                        runlog.add(IterRecord(iteration=it + 1, stage="synth", dsl_text=sp_text,
                                              executed=True, synth_seeded=True, soft_score=1.0))
                        return finalize(ARCAGIResult(
                            train_results=s_tr, results=s_te, iteration=it + 1,
                            prompt_tokens=total_ptok, completion_tokens=total_ctok))
            if synth_pipes:
                feas = "\n".join("  " + sp.pretty() for sp in synth_pipes[:cfg.synth_k])
                message += ("\n\nMACHINE-VERIFIED FEASIBLE skeletons (type-correct and consistent "
                            "with ALL training pairs in the verified abstract semantics — complete "
                            "or adapt ONE of these):\n" + feas)

        # contract injection: dynamic store (fixed + discovered invariants) when
        # learning/inducing, else the static spec when only spec_inject is on.
        active_contracts: list[str] = []
        if store is not None and (cfg.clause_inject or cfg.learn_contracts):
            block = store.render_for_prompt()
            if block:
                message += "\n\n" + block
                active_contracts = ([c.name for c in store.contracts]
                                    + [lc.name for lc in store.learned])
        elif spec is not None and cfg.spec_inject and not spec.is_empty():
            message += "\n\n" + spec.render_for_prompt()
            active_contracts = [c.name for c in spec.contracts]

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

        pipeline = None          # set in DSL mode once a fresh candidate survives
        dsl_text: Optional[str] = None
        if cfg.dsl_required:
            if g.error not in (None, "no-code-block"):
                stage = ("gen_timeout" if g.error == "generator-timeout" else "gen_error")
                runlog.add(IterRecord(iteration=it + 1, stage=stage, error=g.error,
                                      cost_usd=g.cost_usd,
                                      prompt_tokens=g.prompt_tokens,
                                      completion_tokens=g.completion_tokens))
                continue
            src = extract_dsl_block(g.raw or "")
            try:
                if src is None:
                    raise DslError("no ```dsl fenced block found")
                pipeline = parse_pipeline(src, dsl_registry)
            except DslError as e:
                fb = (f"INVALID DSL ({e}). Answer with exactly one ```dsl block, "
                      f"steps separated by ';', only catalog primitives with "
                      f"in-domain args, e.g.: crop_bbox(); scale(2,2)")
                solutions.append(ARCAGISolution(code=(src or (g.raw or "")[-300:]),
                                                feedback=fb, score=0.0))
                runlog.add(IterRecord(iteration=it + 1, stage="dsl_invalid",
                                      error=str(e), dsl_text=src,
                                      cost_usd=g.cost_usd, prompt_tokens=g.prompt_tokens,
                                      completion_tokens=g.completion_tokens))
                continue
            dsl_text = pipeline.pretty()
            if dsl_text in seen_pipelines:
                # exact duplicate: its eval/feedback is already in `solutions` —
                # don't execute again (keeps "executions avoided" honest).
                runlog.add(IterRecord(iteration=it + 1, stage="dup", dup_hit=True,
                                      dsl_text=dsl_text, cost_usd=g.cost_usd,
                                      prompt_tokens=g.prompt_tokens,
                                      completion_tokens=g.completion_tokens))
                continue
            seen_pipelines.add(dsl_text)

            # ---- pre-execution refutation (D1/D2): clause match, then CHECK ----
            check_ms = 0.0
            if cfg.z3_refute and task_smt is not None:
                fired = clause_store.match(pipeline) if clause_store is not None else []
                refuted, nl, core = bool(fired), "", ()
                ms, learned_nl, blocked = 0.0, None, 0
                if fired:
                    nl, core = fired[0].nl, fired[0].core
                else:
                    res = task_smt.check_pipeline(pipeline)
                    ms = check_ms = res.ms
                    if res.refuted:
                        refuted, core = True, res.core
                        if cfg.z3_learn and clause_store is not None:
                            cl = clause_store.learn_from_refutation(pipeline, res.core)
                            nl, learned_nl = cl.nl, cl.nl
                            blocked = cl.n_blocked(cfg.dsl_depth_max)
                        else:
                            nl = (f"with these steps the training pairs' "
                                  f"{_facts_nl(res.core)} cannot be reproduced")
                if refuted:
                    fb = f"PROVABLY IMPOSSIBLE (machine-checked, was not executed): {nl}"
                    solutions.append(ARCAGISolution(code=dsl_text, feedback=fb, score=0.0))
                    runlog.add(IterRecord(
                        iteration=it + 1, stage="refuted", dsl_text=dsl_text,
                        refuted=True, refutation_core=list(core),
                        clause_fired=[c.nl for c in fired], clause_learned=learned_nl,
                        class_pruned=blocked, solver_ms=ms, cost_usd=g.cost_usd,
                        prompt_tokens=g.prompt_tokens,
                        completion_tokens=g.completion_tokens))
                    continue
            code = compile_pipeline(pipeline, dsl_registry)
        else:
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
                                  dsl_text=dsl_text, executed=True,
                                  solver_ms=(check_ms if cfg.dsl_required else 0.0),
                                  cost_usd=g.cost_usd, prompt_tokens=g.prompt_tokens,
                                  completion_tokens=g.completion_tokens))
            return finalize(ARCAGIResult(
                train_results=train_res, results=test_res, iteration=it + 1,
                prompt_tokens=total_ptok, completion_tokens=total_ctok,
            ))

        # ---- contract pre-check: conflict detection (cheapest-first) ----
        violated = []
        violated_learned = []
        conflict_type = None
        produced: list = []
        if spec is not None and (cfg.clause_learn or cfg.learn_contracts):
            produced = [parse_grid(r) for r in train_res]
            violated = spec.violations(train_in, produced)               # fixed basis
            if store is not None and store.learned:                       # induced/reused
                violated_learned = await induced_violations(
                    store.learned, train_in, produced, timeout_s=cfg.timeout_sandbox_s
                )
            conflict_type = "structural" if (violated or violated_learned) else "semantic"
            if store is not None and cfg.clause_learn:
                store.note(violated)

        # ---- DSL concrete failure despite abstract SAT: block the instance and
        #      log WHICH σ components the abstraction let through (abs_weak —
        #      the abstraction-refinement worklist).
        abs_weak: list[str] = []
        if pipeline is not None:
            if clause_store is not None and cfg.z3_learn:
                clause_store.add_concrete_block(pipeline)
            weak: set[str] = set()
            for r, (_, go) in zip(train_res, pairs_np):
                if r["success"]:
                    continue
                po = parse_grid(r)
                if po is None:
                    continue
                sp, sg = sigma_of(po), sigma_of(go)
                if (sp.h, sp.w) != (sg.h, sg.w):
                    weak.add("dims")
                if sp.cnt != sg.cnt:
                    weak.add("hist")
                if sp.n_obj != sg.n_obj:
                    weak.add("objects")
                if (sp.bbox_h, sp.bbox_w) != (sg.bbox_h, sg.bbox_w):
                    weak.add("bbox")
                if sp.sym != sg.sym:
                    weak.add("sym")
            abs_weak = sorted(weak)

        # ---- feedback: poetiq diff, prefixed with the precise structured reasons ----
        feedback, score = _build_feedback(train_res, train_in, train_out)
        if (violated or violated_learned) and (cfg.clause_learn or cfg.learn_contracts):
            feedback = render_violation_feedback(violated + violated_learned) + "\n\n" + feedback

        # ---- pruning (A2): a candidate violating a TRUSTED fixed invariant OR ANY
        #      induced invariant (each holds on all train pairs by the gate, so a
        #      violation proves train-wrongness) is kept out of the best pool.
        pruned = False
        harmful = False
        if cfg.clause_prune and store is not None and (
            any(c.name in store.trusted for c in violated) or bool(violated_learned)
        ):
            pruned = True
            harmful = best_result is not None and score > best_score

        # In DSL mode the feedback memory carries the model's own language, not
        # the compiled python.
        solutions.append(ARCAGISolution(code=(dsl_text or code), feedback=feedback,
                                        score=score))

        if not pruned and score >= best_score:
            best_score = score
            best_result = ARCAGIResult(
                train_results=train_res, results=test_res, iteration=it + 1,
                prompt_tokens=None, completion_tokens=None,
            )

        # ---- adaptive induction (Phase 4): when the known basis is SILENT (semantic
        #      conflict), ask for a new invariant and GATE it by sandboxed verification.
        prop_report: dict = {}
        prop_admitted: Optional[bool] = None
        if (cfg.learn_contracts and store is not None and conflict_type == "semantic"
                and len(store.learned) < cfg.max_learned):
            sem_conflicts += 1
            if sem_conflicts % max(1, cfg.propose_every) == 0:
                lc = await propose_contract(
                    generator, pairs_np, produced,
                    idx=len(store.learned), seed=cfg.seed + it,
                    timeout_s=cfg.timeout_sandbox_s, report=prop_report,
                )
                prop_admitted = lc is not None and store.add_learned(lc)
                if prop_admitted and library is not None:
                    library.add(lc.name, lc.descr, lc.code)
                    library.save()

        runlog.add(IterRecord(
            iteration=it + 1, stage=(conflict_type or "wrong"), soft_score=score,
            violated=[c.name for c in violated] + [lc.name for lc in violated_learned],
            conflict_type=conflict_type,
            pruned=pruned, harmful_prune=harmful, cost_usd=g.cost_usd,
            prompt_tokens=g.prompt_tokens, completion_tokens=g.completion_tokens,
            active=active_contracts,
            proposed=(f"{prop_report['stage']}: {prop_report['descr']}"
                      if prop_report.get("descr") else prop_report.get("stage")),
            prop_admitted=prop_admitted,
            prop_cost_usd=prop_report.get("cost_usd", 0.0),
            dsl_text=dsl_text, executed=True, abs_weak=abs_weak,
            solver_ms=(check_ms if cfg.dsl_required else 0.0),
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
