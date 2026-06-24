"""DSL coverage probe ($0, NO LLM — fully reliable; the haiku CLI hangs, this can't).

The faithful-test conclusion is that E2 is capped by DSL/induction COVERAGE. This measures
the DSL ceiling DIRECTLY and reliably: for each devset task, enumerate feasible skeletons
from the abstract spec (z3 synth, the same machinery the E2 loop uses) and CONCRETELY check
whether any reproduces every train pair — and, if so, whether it generalizes to the test
pair. Pure z3 + in-process pipeline execution, so there is no LLM subprocess to stall.

    uv run python -m clarc.probes.synth_coverage [--depth 4 --models 16]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from clarc.cli import devset
from clarc.dsl.absdomain import sigma_of
# clarc.data is dependency-free (no generator import) — this probe must NOT trigger the
# LiteLLM network-import hang that the "0% CPU stall" evals hit.
from clarc.common.data import load as _load
from clarc.dsl.core import REGISTRY, param_search, pipeline_fits
from clarc.dsl.smt import TaskSMT

_HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--models", type=int, default=20)
    ap.add_argument("--cap", type=int, default=2000, help="max concrete param combos per skeleton")
    ap.add_argument("--data", default="2024-eval")
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "..", "output", "synth_cov.jsonl"))
    args = ap.parse_args()
    challenges, solutions = _load(args.data)
    d = devset.load()

    # RESUMABLE: each task's result is appended to a JSONL checkpoint and skipped on re-run,
    # so the idle-kills in this environment don't lose progress — relaunch to continue.
    done = {}
    if os.path.exists(args.ckpt):
        for line in open(args.ckpt, encoding="utf-8"):
            try:
                r = json.loads(line); done[r["tid"]] = r
            except (json.JSONDecodeError, KeyError):
                pass
    sink = open(args.ckpt, "a", encoding="utf-8")
    print(f"DSL coverage probe (depth={args.depth}, models={args.models}, cap={args.cap}, NO LLM) "
          f"— resuming with {len(done)}/40 done\n", flush=True)
    strat = {tid: s for s in ("structural", "logic") for tid in d[s]}
    for tid in d["logic"] + d["structural"]:   # logic first — the depth-1 solvable tasks live there
        if tid in done:
            continue
        t = challenges[tid]
        tr = [(e["input"], e["output"]) for e in t["train"]]
        te = [(e["input"], sol) for e, sol in zip(t["test"], (solutions or {}).get(tid, []))]
        facts = [(sigma_of(np.asarray(gi)), sigma_of(np.asarray(go))) for gi, go in tr]
        smt = TaskSMT(facts, timeout_ms=3000)
        t0 = time.monotonic()
        pipes = smt.synth_models(args.depth, max_models=args.models)
        seen, skels = set(), []
        for p in pipes:
            sk = tuple(s.name for s in p.steps)
            if sk and sk not in seen:
                seen.add(sk); skels.append(sk)
        solver = next((s for sk in skels if (s := param_search(sk, tr, cap=args.cap))), None)
        gen = (pipeline_fits(solver, te, REGISTRY) if (solver is not None and te) else None)
        rec = {"tid": tid, "stratum": strat[tid], "solved": solver is not None,
               "generalized": bool(gen), "solver": solver.pretty() if solver else None,
               "feasible": len(pipes), "secs": round(time.monotonic() - t0, 1)}
        sink.write(json.dumps(rec) + "\n"); sink.flush(); done[tid] = rec
        print(f"  [{strat[tid][:6]}] {tid}: {'SOLVES via ' + rec['solver'] if solver else 'no solver'}"
              f" gen={gen} ({rec['feasible']} feasible, {rec['secs']}s)", flush=True)
    sink.close()

    print("\n=== DSL COVERAGE WITH PARAM-SEARCH (no LLM) ===", flush=True)
    for s in ("structural", "logic"):
        rs = [r for r in done.values() if r["stratum"] == s]
        print(f"  {s}: train-solved {sum(r['solved'] for r in rs)}/{len(rs)} "
              f"(test-generalized {sum(r['generalized'] for r in rs)})")
    gg = sum(r["generalized"] for r in done.values())
    ss = sum(r["solved"] for r in done.values())
    print(f"  TOTAL: train-solved {ss}/{len(done)}, test-correct {gg}/{len(done)} "
          f"(was 0/40 before param-search — this is the lift).", flush=True)


if __name__ == "__main__":
    main()
