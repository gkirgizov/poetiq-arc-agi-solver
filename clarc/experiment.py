"""Failure-band experiment — RESUMABLE across restarts.

Phase A: run A0 over candidate eval tasks (low iters) to MEASURE which tasks the
model does not one-shot -> the "failure band" (the only regime contracts can help).
Phase B: run arms [A0, A5, A1, A1L] on the band (full iters) to compare ingredients.

Resumability (the whole point): every completed run is appended to <out>/runs.jsonl
keyed by (phase, task, arm). On launch we load that file and SKIP any cell already
present — so a kill loses at most one in-flight cell, and re-running joins results.
Pass --band to supply the band explicitly and skip Phase A entirely.

    uv run python -m clarc.experiment --model sonnet --band 05a7bcf2,0607ce86 --arms A0,A5,A1,A1L --iters 8 --concurrency 2
    uv run python -m clarc.experiment --report-only        # just print the report from runs.jsonl
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

from clarc.generator import ClaudeCodeGenerator, StubGenerator
from clarc.loop import solve_task
from clarc.run import ARMS, _load
from clarc.types import ClarcConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_IDENTITY = "def transform(grid):\n    return grid"
_OUT = os.path.join(_HERE, "..", "output", "experiment")


def _load_checkpoint(path):
    """(phase, task, arm) -> record, from prior runs."""
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
                done[(r["phase"], r["task"], r["arm"])] = r
            except (json.JSONDecodeError, KeyError):
                pass
    return done


async def _run(tid, task, arm, gen, *, iters, seed, sem, timeout, lean=False):
    async with sem:
        ti = [e["input"] for e in task["train"]]
        to = [e["output"] for e in task["train"]]
        test_in = [e["input"] for e in task["test"]]
        cfg = ClarcConfig(model=None, max_iterations=iters, seed=seed, lean_prompt=lean,
                          request_timeout_s=timeout, problem_id=tid, **ARMS[arm])
        res = await solve_task(train_in=ti, train_out=to, test_in=test_in,
                               generator=gen, config=cfg, arm=arm)
        preds = build_kaggle_two_attempts([res], test_in)
        return res, preds


def _rec(phase, tid, arm, log, acc):
    return {"phase": phase, "task": tid, "arm": arm,
            "solved": bool(log.get("solved")), "iters": log.get("iterations_to_solve"),
            "acc": acc, "conf": log.get("n_conflicts", 0), "yield": log.get("clause_yield", 0.0),
            "learned": log.get("learned", []), "n_learned": log.get("n_learned", 0),
            "pruned": log.get("n_pruned", 0), "harm": log.get("n_harmful", 0),
            "genfail": log.get("n_gen_failures", 0), "cost": log.get("total_cost_usd", 0.0)}


def _med(xs):
    return statistics.median(xs) if xs else float("nan")


def _maxdim(task):
    d = 0
    for e in task["train"]:
        for g in (e["input"], e["output"]):
            d = max(d, len(g), len(g[0]))
    for e in task["test"]:
        d = max(d, len(e["input"]), len(e["input"][0]))
    return d


def _report(path, arms):
    done = _load_checkpoint(path)
    B = defaultdict(dict)
    for (phase, tid, arm), r in done.items():
        if phase == "B":
            B[arm][tid] = r
    print("\n=== FAILURE-BAND REPORT (from checkpoint; pass@1) ===")
    for arm in arms:
        recs = B.get(arm, {})
        n = len(recs)
        if not n:
            print(f"[{arm:3s}] (no runs yet)")
            continue
        acc = sum(r["acc"] for r in recs.values())
        solved = sum(1 for r in recs.values() if r["solved"])
        iters_solved = [r["iters"] for r in recs.values() if r["solved"] and r["iters"]]
        conf = sum(r["conf"] for r in recs.values())
        nlearned = sum(r["n_learned"] for r in recs.values())
        pruned = sum(r["pruned"] for r in recs.values())
        harm = sum(r["harm"] for r in recs.values())
        genfail = sum(r["genfail"] for r in recs.values())
        cost = sum(r["cost"] or 0 for r in recs.values())
        meanyield = statistics.mean([r["yield"] for r in recs.values()])
        print(f"[{arm:3s}] acc={acc/n*100:5.1f}% solved={solved}/{n} med_iters={_med(iters_solved):.1f} "
              f"conf={conf} yield={meanyield:.2f} learned={nlearned} prune={pruned} harm={harm} "
              f"genfail={genfail} ${cost:.2f}")
    induced = sorted({d for arm in arms for r in B.get(arm, {}).values() for d in r["learned"]})
    print(f"\nInduced contracts ({len(induced)}):")
    for d in induced:
        print(f"  - {d}")
    if not induced:
        print("  (none)")


async def main_async(args):
    os.makedirs(args.out, exist_ok=True)
    runs_path = os.path.join(args.out, "runs.jsonl")
    arms = args.arms.split(",")

    if args.report_only:
        _report(runs_path, arms)
        return

    challenges, solutions = _load(args.data)
    done = _load_checkpoint(runs_path)
    sink = open(runs_path, "a", encoding="utf-8")  # APPEND — never clobber prior results

    def gen():
        return StubGenerator([_IDENTITY]) if args.stub else ClaudeCodeGenerator(
            model=args.model, timeout_s=args.timeout, effort=args.effort,
            max_thinking_tokens=args.max_thinking)

    def score(tid, preds):
        return score_task(preds, solutions[tid]) if (solutions and tid in solutions and preds) else 0.0

    def write(rec):
        sink.write(json.dumps(rec) + "\n")
        sink.flush()
        done[(rec["phase"], rec["task"], rec["arm"])] = rec

    sem = asyncio.Semaphore(args.concurrency)

    # ---------- Phase A (skip if --band given) ----------
    if args.band:
        band = [t for t in args.band.split(",") if t]
    else:
        cand = list(challenges.keys())[: args.n]
        ids = [t for t in cand if _maxdim(challenges[t]) <= args.max_grid]
        todo = [t for t in ids if ("A", t, "A0") not in done]
        print(f"PHASE A: {len(todo)} new / {len(ids)} gen-safe tasks "
              f"(excluded {len(cand)-len(ids)} with grid>{args.max_grid}; reusing "
              f"{len(ids)-len(todo)} from checkpoint), iters={args.sweep_iters}")

        async def _a(tid):
            res, preds = await _run(tid, challenges[tid], "A0", gen(), iters=args.sweep_iters,
                                    seed=0, sem=sem, timeout=args.timeout, lean=args.lean)
            write(_rec("A", tid, "A0", res.get("clarc_log", {}), score(tid, preds)))

        await asyncio.gather(*[asyncio.create_task(_a(t)) for t in todo])
        sweepvals = {t: done[("A", t, "A0")] for t in ids if ("A", t, "A0") in done}
        # ARC-hard band: baseline got the TEST wrong (acc<1) AND generation actually
        # produced a candidate (cost>0). Excludes gen-timeouts (teach nothing) and
        # test-correct one-shots. Train-fit-but-test-wrong = overfit (clearest place
        # structural contracts could improve generalization) -> surface those first.
        # Band = tasks where the baseline left ROOM for a contract effect: it either
        # got the test wrong, OR needed >1 iteration to solve (room to converge faster).
        # Excludes gen-timeouts (no attempt) and clean one-shot solves (no room).
        def _room(v):
            return (v.get("cost") or 0) > 0 and (
                v.get("acc", 0) < 1.0 or (v.get("iters") or 1) > 1)
        band = [t for t, v in sweepvals.items() if _room(v)]
        band.sort(key=lambda t: (sweepvals[t].get("acc", 0) == 1.0, sweepvals[t].get("iters") or 1))
        band = band[: args.max_band]
        test_ok = sum(1 for v in sweepvals.values() if v.get("acc", 0) == 1.0)
        genfail = sum(1 for v in sweepvals.values() if (v.get("cost") or 0) == 0)
        fails = sum(1 for v in sweepvals.values() if (v.get("cost") or 0) > 0 and v.get("acc", 0) < 1.0)
        print(f"  test-correct={test_ok}/{len(sweepvals)} · gen-fail={genfail} · test-fail={fails} · "
              f"band(fail OR multi-iter)={len(band)}: {band}")

    if not band:
        print("No failure band — model one-shot everything. Nothing for contracts to do.")
        sink.close()
        return

    # ---------- Phase B: arms on the band (resumable) ----------
    cells = [(t, arm) for t in band for arm in arms if ("B", t, arm) not in done]
    print(f"\nPHASE B: {len(cells)} new cells / {len(band)*len(arms)} total "
          f"(reusing {len(band)*len(arms)-len(cells)}), arms={arms}, iters={args.iters}")

    async def _b(tid, arm):
        res, preds = await _run(tid, challenges[tid], arm, gen(), iters=args.iters,
                                seed=0, sem=sem, timeout=args.timeout, lean=args.lean)
        write(_rec("B", tid, arm, res.get("clarc_log", {}), score(tid, preds)))
        print(f"  done {arm} {tid}: solved={done[('B',tid,arm)]['solved']} "
              f"conf={done[('B',tid,arm)]['conf']} learned={done[('B',tid,arm)]['n_learned']}")

    await asyncio.gather(*[asyncio.create_task(_b(t, a)) for t, a in cells])
    sink.close()
    _report(runs_path, arms)


def main():
    p = argparse.ArgumentParser(description="clarc failure-band experiment (resumable)")
    p.add_argument("--data", default="2024-eval")
    p.add_argument("--model", default=None)
    p.add_argument("--effort", default=None, help="claude effort: low|medium|high|xhigh|max")
    p.add_argument("--max-thinking", type=int, default=None, dest="max_thinking",
                   help="MAX_THINKING_TOKENS env to bound sonnet thinking time")
    p.add_argument("--band", default="", help="explicit comma-separated band (skip Phase A)")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--sweep-iters", type=int, default=3, dest="sweep_iters")
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--max-band", type=int, default=24, dest="max_band")
    p.add_argument("--max-grid", type=int, default=16, dest="max_grid",
                   help="skip tasks whose any grid dim exceeds this (avoids gen timeouts)")
    p.add_argument("--arms", default="A0,A5,A1,A1L")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--stub", action="store_true")
    p.add_argument("--lean", action="store_true", help="use terse single-attempt prompt")
    p.add_argument("--report-only", action="store_true", dest="report_only")
    p.add_argument("--out", default=_OUT)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
