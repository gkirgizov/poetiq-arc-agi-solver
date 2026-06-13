"""$0 offline probe of the DSL ⇄ SMT dual over the curated devset.

Measures, with no LLM in the loop:
  (a) COVERAGE floor   — tasks with >=1 depth-<=2 pipeline that exactly fits all
                         train pairs (the D0 solve-rate upper-bound at this depth);
  (b) REFUTATION power — fraction of non-fitting candidates CHECK refutes
                         pre-execution (sampled at depth 2, exhaustive at depth 1);
  (c) SOUNDNESS        — the invariant: NO train-fitting pipeline is ever refuted
                         (any hit = a contract bug; must be 0 before paid runs);
  (d) solver latency   — per-CHECK ms histogram.

    uv run python -m clarc.probe_dsl [--depth 2] [--sample 120] [--tasks N]
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

from clarc import devset
from clarc.absdomain import sigma_of
from clarc.dsl import REGISTRY, DslRuntimeError, Pipeline, Step, run_pipeline
from clarc.dsltypes import Ty
from clarc.run import _load
from clarc.smt import TaskSMT


def _param_grid(prim, cap: int | None = None) -> list[dict]:
    if prim.name == "recolor":
        return []  # handled per-task (maps derived from the pairs)
    combos = [dict(zip([p.name for p in prim.params], vals))
              for vals in itertools.product(*[p.values for p in prim.params])]
    if cap and len(combos) > cap:
        rng = np.random.default_rng(0)
        combos = [combos[i] for i in rng.choice(len(combos), cap, replace=False)]
    return combos


def _task_recolor_steps(pairs) -> list[Step]:
    """Plausible task-specific maps: the pixelwise map of pair 0 (if shapes match)."""
    gi, go = pairs[0]
    if gi.shape != go.shape:
        return []
    lut = {}
    for a, b in zip(gi.ravel(), go.ravel()):
        if lut.setdefault(int(a), int(b)) != int(b):
            return []
    params = {f"pi{c}": lut.get(c, c) for c in range(10)}
    return [Step("recolor", params)]


def _grid_steps(pairs, cap=None) -> list[Step]:
    """All Grid→Grid leaf steps (the M1 whole-grid primitives)."""
    out: list[Step] = []
    for prim in REGISTRY.values():
        if prim.in_type != Ty.GRID or prim.out_type != Ty.GRID or prim.name == "identity":
            continue
        if prim.name == "recolor":
            out += _task_recolor_steps(pairs)
            continue
        out += [Step(prim.name, g) for g in _param_grid(prim, cap=cap)]
    return out


def _sel_steps() -> list[Step]:
    """All Selection→Selection object-layer steps (M5)."""
    out: list[Step] = []
    for prim in REGISTRY.values():
        if prim.in_type == Ty.SELECTION and prim.out_type == Ty.SELECTION:
            out += [Step(prim.name, g) for g in _param_grid(prim)]
    return out


def _candidates(pairs, depth: int, sample: int, rng) -> tuple[list[Pipeline], list[Pipeline]]:
    """All TYPE-VALID pipelines up to `depth`: whole-grid sequences plus object
    pipelines `objects(); <1-2 selection ops>; render()`."""
    grid_steps = _grid_steps(pairs)
    sel_steps = _sel_steps()
    OBJ, REN = Step("objects", {}), Step("render", {})

    d1 = [Pipeline((s,)) for s in grid_steps]                          # 1 whole-grid op
    d1 += [Pipeline((OBJ, s, REN)) for s in sel_steps]                 # 1 object op
    if depth < 2:
        return d1, []

    capped = _grid_steps(pairs, cap=4)
    idx = rng.choice(len(capped), size=(min(sample, len(capped) ** 2), 2))
    d2 = [Pipeline((capped[i], capped[j])) for i, j in idx]            # 2 whole-grid ops
    # object pipelines with two selection ops (e.g. select_largest; recolor_all)
    if len(sel_steps) >= 2:
        sidx = rng.choice(len(sel_steps), size=(min(sample, len(sel_steps) ** 2), 2))
        d2 += [Pipeline((OBJ, sel_steps[i], sel_steps[j], REN)) for i, j in sidx]
    return d1, d2


def _fits(p: Pipeline, pairs) -> bool:
    try:
        return all(np.array_equal(run_pipeline(p, gi), go) for gi, go in pairs)
    except (DslRuntimeError, ValueError, IndexError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--sample", type=int, default=120, help="depth-2 candidates per task")
    ap.add_argument("--tasks", type=int, default=None, help="limit task count")
    args = ap.parse_args()

    challenges, _ = _load("2024-eval")
    ids = devset.all_ids()[: args.tasks]
    rng = np.random.default_rng(1)

    cover, false_ref, lat = {}, [], []
    n_fit_total = n_nonfit = n_refuted = 0
    for tid in ids:
        task = challenges[tid]
        pairs = [(np.asarray(e["input"], dtype=int), np.asarray(e["output"], dtype=int))
                 for e in task["train"]]
        smt = TaskSMT([(sigma_of(a), sigma_of(b)) for a, b in pairs])
        d1, d2 = _candidates(pairs, args.depth, args.sample, rng)
        fits_found = 0
        for p in d1 + d2:
            fit = _fits(p, pairs)
            t0 = time.monotonic()
            refuted = smt.check_pipeline(p).refuted
            lat.append((time.monotonic() - t0) * 1000)
            if fit:
                fits_found += 1
                n_fit_total += 1
                if refuted:
                    false_ref.append((tid, p.pretty()))
            else:
                n_nonfit += 1
                n_refuted += refuted
        stratum = devset.stratum_of(tid)
        cover.setdefault(stratum, []).append(fits_found > 0)
        print(f"  {tid} [{stratum:10s}] cands={len(d1)+len(d2):4d} fits={fits_found:3d} "
              f"power={n_refuted}/{n_nonfit}", flush=True)

    lat_a = np.array(lat)
    print("\n=== PROBE REPORT ===")
    for st, hits in sorted(cover.items()):
        print(f"coverage[{st}]: {sum(hits)}/{len(hits)} tasks have a fitting <=depth-{args.depth} pipeline")
    print(f"refutation power: {n_refuted}/{n_nonfit} non-fitting candidates refuted pre-execution "
          f"({100*n_refuted/max(n_nonfit,1):.1f}%)")
    print(f"FALSE refutations (MUST be 0): {len(false_ref)}")
    for tid, pp in false_ref[:10]:
        print(f"  !! {tid}: {pp}")
    print(f"solver latency ms: p50={np.percentile(lat_a,50):.1f} p95={np.percentile(lat_a,95):.1f} "
          f"max={lat_a.max():.0f} over {len(lat_a)} checks")


if __name__ == "__main__":
    main()
