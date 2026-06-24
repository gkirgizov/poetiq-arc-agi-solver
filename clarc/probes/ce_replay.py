"""CE-opportunity replay ($0) — can the verifier even SEE the errors the generator makes?

A counterexample can only guide search if some VERIFIED invariant is violated by the
wrong candidate. This harness replays every logged wrong candidate (the `produced`
grids in the per-iteration records) against the available verifier vocabulary and asks:
on the iterations where the solver FAILED, would any CE have fired?

Finding (2026-06-16, output/exp-guided-gate): on the 6 devset tasks A0 fails, the
object-correspondence CE fired 1/42 wrong iterations and the broadened grid-level (σ)
set 1/42 — the wrong candidates SATISFY shape/count/palette/recolor-only/bg invariants
and err only in CONTENT (which color/where, conditional on attributes), which the
envelope verifiers can't discriminate. So strengthening CE FIRING needs content-pinning
contracts (M7c conditional rules), not more envelope invariants. This tool measures that
opportunity for free before any paid run.

    uv run python -m clarc.probes.ce_replay --out output/exp-guided-gate [--arms G1,G3] [--only-fails]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

from clarc.objects.dual.object_dual import ObjectDual
from clarc.cli.run import _load
from clarc.contracts.spec import extract_spec


def _grids(prod):
    return [None if g is None else np.asarray(g, dtype=int) for g in (prod or [])]


def replay(out_dir: str, data: str, arms: list[str], only_fails: bool) -> dict:
    challenges, solutions = _load(data)
    paths = sorted(glob.glob(os.path.join(out_dir, "logs", "B_*_G*.json")))
    by_task_arm = {}
    for p in paths:
        name = os.path.basename(p)[2:-5]            # strip "B_" and ".json"
        tid, arm = name.rsplit("_", 1)
        if arm in arms:
            by_task_arm[(tid, arm)] = p

    agg = Counter()
    per_task = {}
    for (tid, arm), p in sorted(by_task_arm.items()):
        task = challenges.get(tid)
        if task is None:
            continue
        log = json.load(open(p, encoding="utf-8"))
        if only_fails and log.get("solved"):
            continue
        ti = [e["input"] for e in task["train"]]
        to = [e["output"] for e in task["train"]]
        spec = extract_spec(ti, to, tid)
        od = ObjectDual()
        od.extract([(np.asarray(a), np.asarray(b)) for a, b in zip(ti, to)])
        n_wrong = n_sigma = n_obj = n_either = 0
        sig_names: set[str] = set()
        for rec in log.get("records", []):
            if rec.get("stage") == "solved" or rec.get("produced") is None:
                continue
            n_wrong += 1
            pg = _grids(rec["produced"])
            sv = spec.violations(ti, pg)
            cx = od.refute([(np.asarray(a), g) for a, g in zip(ti, pg)])
            if sv:
                n_sigma += 1; sig_names |= {c.name for c in sv}
            if cx:
                n_obj += 1
            if sv or cx:
                n_either += 1
        if n_wrong:
            per_task[(tid, arm)] = dict(wrong=n_wrong, sigma=n_sigma, obj=n_obj,
                                        either=n_either, sig=sorted(sig_names),
                                        spec=[c.name for c in spec.contracts])
            agg["wrong"] += n_wrong; agg["sigma"] += n_sigma
            agg["obj"] += n_obj; agg["either"] += n_either
    return {"agg": dict(agg), "per_task": per_task}


def main() -> None:
    ap = argparse.ArgumentParser(description="CE-opportunity replay (offline, $0)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="2024-eval")
    ap.add_argument("--arms", default="G1,G3")
    ap.add_argument("--only-fails", action="store_true",
                    help="only tasks the arm did NOT solve (where guidance is needed)")
    args = ap.parse_args()
    r = replay(args.out, args.data, args.arms.split(","), args.only_fails)
    a = r["agg"]
    w = a.get("wrong", 0)
    print(f"\n=== CE OPPORTUNITY ({args.out}, arms={args.arms}"
          f"{', fails only' if args.only_fails else ''}) ===")
    for (tid, arm), d in r["per_task"].items():
        print(f"  {tid} {arm}: wrong={d['wrong']} σ-CE={d['sigma']} obj-CE={d['obj']} "
              f"either={d['either']}  σ={d['sig']}")
    if w:
        print(f"\nTOTAL wrong iters={w} · a CE would fire on {a['either']} "
              f"({100*a['either']//w}%) · σ={a['sigma']} object={a['obj']}")
        print("Low % ⇒ the verifier can't SEE the errors (content, not envelope) — "
              "CE firing needs content-pinning (conditional) contracts, not more envelope σ.")
    else:
        print("  (no wrong iterations found)")


if __name__ == "__main__":
    main()
