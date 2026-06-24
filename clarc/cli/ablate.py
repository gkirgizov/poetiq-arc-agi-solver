"""Ablation runner + report for the clarc head-to-head.

Runs a set of arms x seeds over the curated dev set, votes across seeds with
poetiq's build_kaggle_two_attempts (pass@2), scores against ARC-AGI-1 solutions,
and prints per-stratum / overall aggregates plus a paired A0-vs-A1 breakdown for
the go/no-go decision.

Validate the pipeline with NO API spend:
    uv run python -m clarc.cli.ablate --stub --num 3 --arms A0,A1 --seeds 0,1

Real run (spends; pick a cheap model and confirm budget):
    uv run python -m clarc.cli.ablate --model sonnet --arms A0,A5,A1 --seeds 0,1,2 --num 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import defaultdict

from arc_agi.io import build_kaggle_two_attempts
from arc_agi.scoring import score_task

from clarc.cli.devset import all_ids, load as load_devset, stratum_of
from clarc.solve.harness import make_generator, solve_one
from clarc.cli.run import _load

_HERE = os.path.dirname(os.path.abspath(__file__))


async def _one(task_id, task, arm, seed, gen_factory, args, sem):
    async with sem:
        test_in = [e["input"] for e in task["test"]]
        try:
            res, _preds = await solve_one(task_id, task, arm, gen_factory(),
                                          iters=args.iters, seed=seed,
                                          timeout=args.timeout, model=args.model)
        except Exception as e:  # noqa: BLE001 — one bad cell shouldn't kill the sweep
            return task_id, arm, seed, {"error": repr(e)}, None
        return task_id, arm, seed, res, test_in


def _aggregate(results, solutions, devset, arms, seeds):
    """results[(task,arm)] = list of (seed,res,test_in). Returns per-arm rollups."""
    rows = {}
    for arm in arms:
        per_stratum = defaultdict(lambda: {
            "n": 0, "correct": 0.0, "train_solved": 0, "iters": [],
            "yield": [], "pruned": 0, "harmful": 0, "cost": 0.0, "genfail": 0,
        })
        for tid in {t for (t, a) in results if a == arm}:
            runs = results[(tid, arm)]
            test_in = next((ti for (_, r, ti) in runs if ti is not None), None)
            res_list = [r for (_, r, ti) in runs if isinstance(r, dict) and "results" in r]
            strat = stratum_of(tid, devset)
            agg = per_stratum[strat]
            agg["n"] += 1
            # pass@2 via seed-ensemble voting
            if res_list and test_in is not None and solutions and tid in solutions:
                preds = build_kaggle_two_attempts(res_list, test_in)
                agg["correct"] += score_task(preds, solutions[tid])
            # per-seed mechanism stats
            for (_seed, r, _ti) in runs:
                if not isinstance(r, dict):
                    continue
                log = r.get("clarc_log", {})
                if log.get("solved"):
                    agg["train_solved"] += 1
                    if log.get("iterations_to_solve"):
                        agg["iters"].append(log["iterations_to_solve"])
                agg["yield"].append(log.get("clause_yield", 0.0))
                agg["pruned"] += log.get("n_pruned", 0)
                agg["harmful"] += log.get("n_harmful", 0)
                agg["cost"] += log.get("total_cost_usd", 0.0) or 0.0
                agg["genfail"] += log.get("n_gen_failures", 0)
        rows[arm] = per_stratum
    return rows


def _print_report(rows, arms, n_seeds):
    def fmt(agg):
        n = agg["n"] or 1
        med_iters = statistics.median(agg["iters"]) if agg["iters"] else float("nan")
        meanyield = statistics.mean(agg["yield"]) if agg["yield"] else 0.0
        return (f"acc={agg['correct']/n*100:5.1f}%  trainsolve={agg['train_solved']}/{n*n_seeds}  "
                f"med_iters={med_iters:.1f}  yield={meanyield:.2f}  "
                f"prune={agg['pruned']} harm={agg['harmful']} genfail={agg['genfail']}  "
                f"${agg['cost']:.2f}")

    print("\n=== ablation report (pass@2 via seed-ensemble) ===")
    for arm in arms:
        print(f"\n[{arm}]")
        ov = {"n": 0, "correct": 0.0, "train_solved": 0, "iters": [], "yield": [],
              "pruned": 0, "harmful": 0, "cost": 0.0, "genfail": 0}
        for strat in ("structural", "logic"):
            agg = rows[arm].get(strat)
            if not agg:
                continue
            print(f"  {strat:11s} {fmt(agg)}")
            for k in ov:
                ov[k] = ov[k] + agg[k] if not isinstance(ov[k], list) else ov[k] + agg[k]
        print(f"  {'OVERALL':11s} {fmt(ov)}")


async def main_async(args):
    devset = load_devset()
    challenges, solutions = _load(args.data)
    ids = all_ids(devset)
    if args.num:
        ids = devset["structural"][: args.num] + devset["logic"][: args.num]
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    gen_factory = make_generator(args)
    sem = asyncio.Semaphore(args.concurrency)

    print(f"tasks={len(ids)} arms={arms} seeds={seeds} model={args.model or 'CLI-default'} "
          f"stub={args.stub}  => {len(ids)*len(arms)*len(seeds)} runs")

    jobs = [
        asyncio.create_task(_one(tid, challenges[tid], arm, seed, gen_factory, args, sem))
        for tid in ids if tid in challenges
        for arm in arms for seed in seeds
    ]

    results = defaultdict(list)
    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, "ablation_raw.jsonl")
    start = time.time()
    done = 0
    with open(raw_path, "w", encoding="utf-8") as raw:
        for coro in asyncio.as_completed(jobs):
            tid, arm, seed, res, test_in = await coro
            results[(tid, arm)].append((seed, res, test_in))
            done += 1
            log = res.get("clarc_log", {}) if isinstance(res, dict) else {}
            raw.write(json.dumps({"task": tid, "arm": arm, "seed": seed, "log": log}) + "\n")
            raw.flush()
            if done % 10 == 0 or done == len(jobs):
                print(f"  ...{done}/{len(jobs)} ({round(time.time()-start)}s)")

    rows = _aggregate(results, solutions, devset, arms, seeds)
    _print_report(rows, arms, len(seeds))
    with open(os.path.join(args.out, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump({arm: {s: dict(a) for s, a in rows[arm].items()} for arm in arms}, f, indent=2)
    print(f"\nwrote {args.out}/ablation_summary.json and ablation_raw.jsonl")


def main():
    p = argparse.ArgumentParser(description="clarc ablation runner")
    p.add_argument("--data", default="2024-eval")
    p.add_argument("--arms", default="A0,A5,A1")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--num", type=int, default=None, help="N tasks per stratum (default: all)")
    p.add_argument("--model", default=None)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--stub", action="store_true", help="offline StubGenerator (no API spend)")
    p.add_argument("--out", default=os.path.join(_HERE, "..", "output", "ablation"))
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
