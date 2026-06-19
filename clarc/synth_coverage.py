"""DSL coverage probe ($0, NO LLM — fully reliable; the haiku CLI hangs, this can't).

The faithful-test conclusion is that E2 is capped by DSL/induction COVERAGE. This measures
the DSL ceiling DIRECTLY and reliably: for each devset task, enumerate feasible skeletons
from the abstract spec (z3 synth, the same machinery the E2 loop uses) and CONCRETELY check
whether any reproduces every train pair — and, if so, whether it generalizes to the test
pair. Pure z3 + in-process pipeline execution, so there is no LLM subprocess to stall.

    uv run python -m clarc.synth_coverage [--depth 4 --models 16]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

from clarc import devset
from clarc.absdomain import sigma_of
from clarc.dsl import REGISTRY, DslRuntimeError, Pipeline, param_search, run_pipeline
from clarc.smt import TaskSMT

# Inline data load — deliberately AVOID `from clarc.run import _load`, which transitively
# imports the generator → LiteLLM, whose import makes a network call that hangs the whole
# process when the network is degraded (the root cause of the "0% CPU stall" evals).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = {"2024-train": ("arc-prize-2024", "training"), "2024-eval": ("arc-prize-2024", "evaluation"),
             "2025-train": ("arc-prize-2025", "training"), "2025-eval": ("arc-prize-2025", "evaluation")}


def _load(dataset):
    folder, split = _DATASETS[dataset]
    base = os.path.join(_HERE, "..", "data", folder)
    with open(os.path.join(base, f"arc-agi_{split}_challenges.json"), encoding="utf-8") as f:
        challenges = json.load(f)
    sol_path = os.path.join(base, f"arc-agi_{split}_solutions.json")
    solutions = json.load(open(sol_path, encoding="utf-8")) if os.path.exists(sol_path) else None
    return challenges, solutions


def _solves(pipe: Pipeline, pairs) -> bool:
    # synth_models does NOT enforce the Grid/Selection type threading (only the parser
    # does), so it can emit ill-typed skeletons — those raise AttributeError/TypeError on
    # execution and simply aren't solutions.
    try:
        return all(np.array_equal(run_pipeline(pipe, np.asarray(gi), REGISTRY), np.asarray(go))
                   for gi, go in pairs)
    except (DslRuntimeError, ValueError, IndexError, KeyError, AttributeError, TypeError):
        return False


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
        gen = (_solves(solver, te) if (solver is not None and te) else None)
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
