"""Soundness audit: replay every PRE-EXECUTION refutation from an experiment's
sidecar logs and verify none of those candidates actually fits the train pairs.

A nonzero count is an unsound abstract contract (a false refutation) and
invalidates the pruning claim — this is the tripwire, run after every paid run:

    uv run python -m clarc.audit_refutations --out output/exp-dsl-haiku [--data 2024-eval]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from clarc.dsl import REGISTRY, DslRuntimeError, Primitive, run_pipeline
from clarc.dslparse import DslError, parse_pipeline
from clarc.dsltypes import Ty
from clarc.learn_prim import _make_apply
from clarc.run import _load


def _task_registry(log: dict) -> dict:
    """Rebuild the task-local registry (global + induced prims) from a sidecar,
    so refutations of pipelines that USE induced prims can be replayed. Only the
    apply (from stored code) is needed to check concrete fit."""
    reg = dict(REGISTRY)
    for ip in log.get("induced_prims", []):
        try:
            reg[ip["name"]] = Primitive(ip["name"], "induced", (), ip.get("descr", ""),
                                        _make_apply(ip["code"]), lambda a, b, P: [],
                                        in_type=Ty.GRID, out_type=Ty.GRID, code=ip["code"])
        except (SyntaxError, KeyError):
            pass
    return reg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="2024-eval")
    args = ap.parse_args()

    challenges, _ = _load(args.data)
    n_refuted = n_replayed = 0
    false_refutations: list[tuple[str, str]] = []
    # all DSL-arm sidecars (D0/D1/D2/E0/…) — refutations come from any of them
    paths = sorted(set(glob.glob(os.path.join(args.out, "logs", "*_D*.json")))
                   | set(glob.glob(os.path.join(args.out, "logs", "*_E*.json"))))
    for path in paths:
        with open(path, encoding="utf-8") as f:
            log = json.load(f)
        tid = log.get("task_id")
        task = challenges.get(tid)
        if task is None:
            continue
        pairs = [(np.asarray(e["input"], dtype=int), np.asarray(e["output"], dtype=int))
                 for e in task["train"]]
        reg = _task_registry(log)
        for rec in log.get("records", []):
            if not rec.get("refuted"):
                continue
            n_refuted += 1
            text = rec.get("dsl_text") or ""
            try:
                p = parse_pipeline(text, reg)
            except DslError:
                continue
            n_replayed += 1
            try:
                fits = all(np.array_equal(run_pipeline(p, gi, reg), go) for gi, go in pairs)
            except (DslRuntimeError, ValueError, IndexError):
                fits = False
            if fits:
                false_refutations.append((tid, text))

    print(f"refuted candidates: {n_refuted} (replayed {n_replayed})")
    print(f"FALSE refutations (must be 0): {len(false_refutations)}")
    for tid, text in false_refutations:
        print(f"  !! {tid}: {text}")
    raise SystemExit(1 if false_refutations else 0)


if __name__ == "__main__":
    main()
