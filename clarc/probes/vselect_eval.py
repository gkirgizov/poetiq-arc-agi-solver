"""Verified-selection eval ($0, within-run) — does picking the train-passer whose TEST
output respects the LOO-trusted invariants beat the baseline's first-passer choice?

The poetiq baseline early-exits on the FIRST train-passing candidate; if that candidate is
overfit (test-wrong) it loses, blind to the test behavior. Arm G5 keeps sampling and logs
every train-passer's test output + its vscore (# trusted invariants its test output
violates). This evaluator replays those logged passers against the solutions and compares
three SELECTION strategies on the *identical* candidate set — isolating selection from
generator sampling noise:

  first   : submit passers[0]            (= what the baseline early-exit submits)
  vselect : submit argmin vscore         (the dual's overfit filter)
  oracle  : any passer correct           (ceiling for selection)

    uv run python -m clarc.probes.vselect_eval --out output/vselect-eval --data 2024-eval
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from clarc.cli.run import _load


def _correct(test_out, sol) -> bool:
    if not test_out or len(test_out) != len(sol):
        return False
    for p, s in zip(test_out, sol):
        if p is None or not np.array_equal(np.asarray(p), np.asarray(s)):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="verified-selection within-run eval ($0)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="2024-eval")
    args = ap.parse_args()
    challenges, solutions = _load(args.data)
    if solutions is None:
        raise SystemExit(f"--data {args.data} has no solutions")

    n = base = vsel = orac = 0
    n_multi = n_decisive = 0          # decisive = passers disagree on correctness (oracle>0 and <n_passers)
    recovered, regressed = [], []
    for p in sorted(glob.glob(os.path.join(args.out, "logs", "*_G5.json"))):
        log = json.load(open(p, encoding="utf-8"))
        tid = log.get("task_id")
        vs = log.get("vselect")
        if not vs or tid not in solutions:
            continue
        sol = solutions[tid]
        passers = vs["passers"]
        corr = [_correct(pp["test_out"], sol) for pp in passers]
        n += 1
        b, v, o = corr[0], corr[vs["picked"]], any(corr)
        base += b; vsel += v; orac += o
        distinct = {json.dumps(pp["test_out"]) for pp in passers}
        if len(distinct) > 1:
            n_multi += 1
        if 0 < sum(corr) < len(corr):
            n_decisive += 1
        if v and not b:
            recovered.append(tid)
        if b and not v:
            regressed.append(tid)

    print(f"\n=== VERIFIED-SELECTION EVAL ({args.out}, {args.data}) ===")
    print(f"tasks with >=1 train-passer: {n}  (>=2 distinct passers: {n_multi}; "
          f"decisive — passers disagree on test-correctness: {n_decisive})")
    print(f"  first-passer (baseline early-exit): {base}/{n}")
    print(f"  vselect (min trusted-violation):    {vsel}/{n}")
    print(f"  oracle (any passer correct):        {orac}/{n}")
    print(f"  vselect RECOVERED (right where baseline wrong): {recovered}")
    print(f"  vselect REGRESSED (wrong where baseline right): {regressed}")
    if n_decisive:
        # how much of the recoverable gap (oracle-baseline) did vselect capture?
        gap = orac - base
        got = vsel - base
        print(f"  recoverable gap oracle-baseline={gap}; vselect captured={got}"
              f"{f' ({100*got//gap}%)' if gap else ''}")


if __name__ == "__main__":
    main()
