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
import time

import numpy as np

from clarc import devset
from clarc.absdomain import sigma_of
from clarc.dsl import REGISTRY, DslRuntimeError, Pipeline, run_pipeline
from clarc.run import _load
from clarc.smt import TaskSMT


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
    ap.add_argument("--models", type=int, default=16)
    ap.add_argument("--data", default="2024-eval")
    args = ap.parse_args()
    challenges, solutions = _load(args.data)
    d = devset.load()

    print(f"DSL coverage probe (depth={args.depth}, models={args.models}, NO LLM) — z3 synth + "
          f"concrete check\n")
    rollup = {}
    for stratum in ("structural", "logic"):
        ids = d[stratum]
        train_cov = []     # tasks where a synth skeleton reproduces all train pairs
        test_gen = []      # ... and also reproduces the held-out test pair
        feas_only = []     # synth feasible but no concrete train-solver in top-K
        none = []
        for tid in ids:
            t = challenges[tid]
            tr = [(e["input"], e["output"]) for e in t["train"]]
            te = [(e["input"], sol) for e, sol in zip(t["test"], (solutions or {}).get(tid, []))]
            facts = [(sigma_of(np.asarray(gi)), sigma_of(np.asarray(go))) for gi, go in tr]
            smt = TaskSMT(facts, timeout_ms=3000)
            t0 = time.monotonic()
            pipes = smt.synth_models(args.depth, max_models=args.models)
            dt = time.monotonic() - t0
            solver = next((p for p in pipes if _solves(p, tr)), None)
            if solver is not None:
                gen = _solves(solver, te) if te else None
                (test_gen if gen else train_cov).append(tid)
                print(f"  [{stratum[:6]}] {tid}: SOLVES train via `{solver.pretty()}` "
                      f"-> test_generalizes={gen}  ({len(pipes)} feasible, {dt:.1f}s)")
            elif pipes:
                feas_only.append(tid)
            else:
                none.append(tid)
        rollup[stratum] = dict(train_solved=len(train_cov) + len(test_gen),
                               test_generalized=len(test_gen),
                               feasible_only=len(feas_only), infeasible=len(none), n=len(ids))
    print("\n=== DSL COVERAGE (no LLM, the reliable ceiling) ===")
    for s, r in rollup.items():
        print(f"  {s}: train-solved {r['train_solved']}/{r['n']} "
              f"(test-generalized {r['test_generalized']}); "
              f"feasible-but-unsolved {r['feasible_only']}; infeasible {r['infeasible']}")
    tot_g = sum(r["test_generalized"] for r in rollup.values())
    tot_n = sum(r["n"] for r in rollup.values())
    print(f"  TOTAL pure-DSL test-correct (no LLM, no induction): {tot_g}/{tot_n} "
          f"— this is the binding constraint E2 must beat via the LLM + induction.")


if __name__ == "__main__":
    main()
