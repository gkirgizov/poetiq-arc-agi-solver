"""clarc runner: load ARC tasks, solve them, score, and report.

Phase 0: single-expert solve through the Claude Code CLI generator (A0 arm).
Ensemble voting and the A0-A5 ablation matrix arrive in Phase 3.

Examples:
    uv run python -m clarc.run --tasks 0d3d703e,3c9b0459 --data 2024-train
    uv run python -m clarc.run --num 5 --data 2024-eval --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Optional

from arc_agi.io import build_kaggle_two_attempts
from arc_agi.scoring import score_task

from clarc.generator import ClaudeCodeGenerator
from clarc.loop import solve_task
from clarc.types import ClarcConfig

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO_ROOT, "data")

_DATASETS = {
    "2024-train": ("arc-prize-2024", "training"),
    "2024-eval": ("arc-prize-2024", "evaluation"),
    "2025-train": ("arc-prize-2025", "training"),
    "2025-eval": ("arc-prize-2025", "evaluation"),
}

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
    # --- GUIDED code-gen (the goal): poetiq A0 + the logical dual as a GUIDE ---
    "G0": dict(),   # == A0 control (no duals), routed through clarc.solver.guided_solve
    "G1": dict(),   # + object-correspondence dual (ungated constraints + vague CE) — control
    "G2": dict(),   # + object + sigma duals — control
    "G3": dict(),   # + confidence-GATED object dual + witness-decoding (strong) CE + soft phrasing
    "G4": dict(),   # G3 + sigma
    "G5": dict(),   # VERIFIED SELECTION: no prompt guidance; keep sampling past first train-pass
                    # and submit the train-passer whose TEST output respects LOO-trusted invariants
}


def _load(dataset: str) -> tuple[dict, Optional[dict]]:
    folder, split = _DATASETS[dataset]
    base = os.path.join(_DATA, folder)
    with open(os.path.join(base, f"arc-agi_{split}_challenges.json"), encoding="utf-8") as f:
        challenges = json.load(f)
    solutions = None
    sol_path = os.path.join(base, f"arc-agi_{split}_solutions.json")
    if os.path.exists(sol_path):
        with open(sol_path, encoding="utf-8") as f:
            solutions = json.load(f)
    return challenges, solutions


async def _run_one(task_id, task, generator, cfg_base, sem):
    async with sem:
        start = time.time()
        train = task.get("train", [])
        test = task.get("test", [])
        train_in = [ex["input"] for ex in train]
        train_out = [ex["output"] for ex in train]
        test_in = [ex["input"] for ex in test]
        cfg_kwargs = {k: v for k, v in cfg_base.items() if not k.startswith("_")}
        cfg = ClarcConfig(**cfg_kwargs, problem_id=task_id)
        try:
            result = await solve_task(
                train_in=train_in, train_out=train_out, test_in=test_in,
                generator=generator, config=cfg, arm=cfg_base.get("_arm", "A0"),
            )
        except Exception as e:  # noqa: BLE001 — one bad task shouldn't kill the run
            return task_id, None, {"error": repr(e)}, time.time() - start
        preds = build_kaggle_two_attempts([result], test_in)
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

    generator = ClaudeCodeGenerator(model=args.model, timeout_s=args.timeout,
                                    max_thinking_tokens=args.max_thinking,
                                    thinking=args.thinking)
    cfg_base = dict(
        model=args.model, max_iterations=args.iters, seed=args.seed,
        request_timeout_s=args.timeout, _arm=args.arm, **ARMS[args.arm],
    )
    print(f"arm={args.arm} flags={ARMS[args.arm]}")
    sem = asyncio.Semaphore(args.concurrency)

    tasks = [
        asyncio.create_task(_run_one(tid, challenges[tid], generator, cfg_base, sem))
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
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
