"""clarc runner: load ARC tasks, solve them, score, and report.

Phase 0: single-expert solve through the Claude Code CLI generator (A0 arm).
Ensemble voting and the A0-A5 ablation matrix arrive in Phase 3.

Examples:
    uv run python -m clarc.cli.run --tasks 0d3d703e,3c9b0459 --data 2024-train
    uv run python -m clarc.cli.run --num 5 --data 2024-eval --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import time

from arc_agi.scoring import score_task

# Dataset loading lives in clarc.common.data (dependency-free). Re-exported here as
# `_load` / `_DATASETS` so the many `from clarc.cli.run import _load` call sites keep working.
from clarc.common.data import DATASETS as _DATASETS
from clarc.common.data import load as _load
from clarc.solve.harness import make_generator, solve_one

# Ablation arms -> ingredient flags.
ARMS = {
    "A0": dict(),                                                              # baseline
    "A5": dict(spec_inject=True),                                             # spec-only
    "A3": dict(clause_learn=True, clause_prune=True),                         # prune, no prompt
    "A2": dict(spec_inject=True, clause_learn=True, clause_inject=True, clause_prune=True),
    "A1": dict(spec_inject=True, clause_learn=True, clause_inject=True),      # full (soft)
    "A1L": dict(spec_inject=True, clause_learn=True, clause_inject=True,      # + adaptive induction
                learn_contracts=True),
    # --- DSL ⇄ SMT dual (Phase 5): LLM emits typed pipelines ---
    "D0": dict(dsl_required=True),                                # try-and-drop baseline (the A4 tax)
    "D1": dict(dsl_required=True, z3_refute=True),                # + pre-execution refutation
    "D2": dict(dsl_required=True, z3_refute=True, z3_learn=True),  # + clause learning (+prompt inject)
    "E0": dict(dsl_required=True, z3_refute=True, z3_learn=True,  # + SELF-EXTENSION: induce new
               induce_prims=True),                               # primitives when the library is insufficient
    "E1": dict(dsl_required=True, z3_refute=True, z3_learn=True,  # + DYNAMIC OBJECTS: induce rules over
               induce_prims=True, dynamic_objects=True),         # task-chosen segmentation (recolor/move/drop)
    "E2": dict(dsl_required=True, z3_refute=True, z3_learn=True,  # + SYNTH: the LLM generates from the
               induce_prims=True, synth_seed=True),              # clause-pruned FEASIBLE subspace (closes criterion 4)
    "DS": dict(dsl_required=True, z3_refute=True, z3_learn=True,  # ablation: synth WITHOUT induction
               synth_seed=True),                                 # (isolates the synth contribution)
    "E3": dict(dsl_required=True, z3_refute=True, z3_learn=True,  # E2 + F4 generalization gate:
               induce_prims=True, synth_seed=True,               # induce from n-1, hold out 1 ->
               induce_holdout=True),                             # all-train check rejects overfit prims
    # --- GUIDED code-gen (the goal): poetiq A0 + the logical dual as a GUIDE ---
    "G0": dict(),   # == A0 control (no duals), routed through clarc.solve.solver.guided_solve
    "G1": dict(),   # + object-correspondence dual (ungated constraints + vague CE) — control
    "G2": dict(),   # + object + sigma duals — control
    "G3": dict(),   # + confidence-GATED object dual + witness-decoding (strong) CE + soft phrasing
    "G4": dict(),   # G3 + sigma
    "G5": dict(),   # VERIFIED SELECTION: no prompt guidance; keep sampling past first train-pass
                    # and submit the train-passer whose TEST output respects LOO-trusted invariants
}


async def _run_one(task_id, task, generator, arm, args, sem):
    async with sem:
        start = time.time()
        try:
            result, preds = await solve_one(task_id, task, arm, generator,
                                            iters=args.iters, seed=args.seed,
                                            timeout=args.timeout, model=args.model)
        except Exception as e:  # noqa: BLE001 — one bad task shouldn't kill the run
            return task_id, None, {"error": repr(e)}, time.time() - start
        log = result.get("clarc_log", {})
        meta = {
            "iteration": result.get("iteration"),
            "cost_usd": result.get("cost_usd", 0.0),
            "n_conflicts": log.get("n_conflicts", 0),
            "clause_yield": log.get("clause_yield", 0.0),
            "n_pruned": log.get("n_pruned", 0),
            "n_learned": log.get("n_learned", 0),
            "learned": log.get("learned", []),
            "log": log,
        }
        return task_id, preds, meta, time.time() - start


async def main_async(args) -> None:
    challenges, solutions = _load(args.data)

    if args.tasks:
        ids = [t for t in args.tasks.split(",") if t]
    else:
        ids = list(challenges.keys())[: args.num] if args.num else list(challenges.keys())

    generator = make_generator(args)()
    print(f"arm={args.arm} flags={ARMS[args.arm]}")
    sem = asyncio.Semaphore(args.concurrency)

    tasks = [
        asyncio.create_task(_run_one(tid, challenges[tid], generator, args.arm, args, sem))
        for tid in ids
        if tid in challenges
    ]

    total, correct, total_cost = 0, 0.0, 0.0
    for coro in asyncio.as_completed(tasks):
        task_id, preds, meta, elapsed = await coro
        if preds is None:
            print(f"! {task_id} ({round(elapsed)}s) {meta.get('error')}")
            continue
        total_cost += meta.get("cost_usd", 0.0) or 0.0
        cw = (f"conf={meta['n_conflicts']} yield={meta['clause_yield']:.2f} "
              f"prune={meta['n_pruned']} learned={meta['n_learned']}")
        if solutions is not None and task_id in solutions:
            s = score_task(preds, solutions[task_id])
            total += 1
            correct += s
            mark = "✓" if s == 1.0 else "✗"
            print(f"{mark} {task_id} it={meta['iteration']} {cw} "
                  f"${meta['cost_usd']:.3f} ({round(elapsed)}s) [{correct:g}/{total}]")
        else:
            print(f"· {task_id} it={meta['iteration']} {cw} ${meta['cost_usd']:.3f} ({round(elapsed)}s)")
        for d in meta["learned"]:
            print(f"      ↳ learned: {d}")

    print("\n=== summary ===")
    if total:
        print(f"accuracy: {correct/total*100:.1f}%  ({correct:g}/{total})")
    print(f"total cost: ${total_cost:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description="clarc runner")
    p.add_argument("--data", default="2024-eval", choices=sorted(_DATASETS))
    p.add_argument("--arm", default="A0", choices=sorted(ARMS), help="ablation arm")
    p.add_argument("--tasks", default="", help="comma-separated task ids")
    p.add_argument("--num", type=int, default=None, help="run first N tasks")
    p.add_argument("--model", default=None, help="claude model alias (e.g. opus, sonnet)")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-thinking", type=int, default=None, dest="max_thinking",
                   help="bound extended thinking via the hidden --max-thinking-tokens CLI flag")
    p.add_argument("--thinking", default=None, choices=["disabled", "adaptive"],
                   help="--thinking CLI flag; 'disabled' turns extended thinking off")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--stub", action="store_true", help="offline StubGenerator (no API spend)")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
