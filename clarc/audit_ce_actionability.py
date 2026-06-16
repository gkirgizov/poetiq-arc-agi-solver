"""CE actionability — does a counterexample actually get REPAIRED next iteration?

The guided-solver's value rests on the claim that a structural counterexample is an
ACTIONABLE repair signal (HANDOFF §3j: the CE channel was under-engaged; strengthening
it should drive the gains). This audit measures it from the per-iteration `records`
the G-arm now logs (`clarc/solver.py` + `instrument.IterRecord.ce`/`produced`):

  for each CE fired at iteration t referencing train example k, look at the candidate
  the NEXT iteration produced (which SAW that CE) and ask: is example k now correct?

  CE_actionability = repaired / (repaired + persisted)     ← higher is better
  harm_rate        = P(soft_score drops after a CE)        ← lower is better

Repair is keyed on the train example (whose input/output never change), so it is robust
to per-iteration re-segmentation — no re-induction needed, only the task's train outputs.
G-arm CEs only fire on already-train-FAILING candidates against train-verified contracts,
so they are sound by construction; this audit is about USEFULNESS, not soundness.

    uv run python -m clarc.audit_ce_actionability --out output/exp-guided-gate
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from clarc.run import _load


def _eq(a, b) -> bool:
    if a is None or b is None:
        return False
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b)


def _next_with_produced(recs, i):
    for j in range(i + 1, len(recs)):
        if recs[j].get("produced") is not None:
            return recs[j]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="CE actionability / harm audit (G-arms)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="2024-eval")
    args = ap.parse_args()

    challenges, _ = _load(args.data)
    paths = sorted(glob.glob(os.path.join(args.out, "logs", "*_G*.json")))

    # per-arm and per-CE-kind tallies
    rep = defaultdict(int); per = defaultdict(int)            # repaired / (repaired+persisted)
    rep_kind = defaultdict(int); per_kind = defaultdict(int)
    harm = defaultdict(int); harm_den = defaultdict(int)
    fired = defaultdict(int)

    for path in paths:
        with open(path, encoding="utf-8") as f:
            log = json.load(f)
        arm = log.get("arm", "?")
        task = challenges.get(log.get("task_id"))
        if task is None:
            continue
        true_out = [e["output"] for e in task["train"]]
        recs = log.get("records", [])
        for i, r in enumerate(recs):
            ce = r.get("ce")
            if not ce or not ce.get("viols"):
                continue
            fired[arm] += 1
            nxt = _next_with_produced(recs, i)
            if nxt is None:
                continue
            prod = nxt["produced"]
            # harm: did the sharp feedback push the next candidate to a WORSE score?
            harm_den[arm] += 1
            if nxt.get("soft_score", 0.0) < r.get("soft_score", 0.0):
                harm[arm] += 1
            # repair: per referenced train example, is it correct now?
            exs = sorted({v["example"] for v in ce["viols"]})
            kinds = {v["example"]: v["kind"] for v in ce["viols"]}
            for k in exs:
                if k - 1 >= len(true_out) or k - 1 >= len(prod):
                    continue
                ok = _eq(prod[k - 1], true_out[k - 1])
                per[arm] += 1; rep[arm] += int(ok)
                per_kind[(arm, kinds[k])] += 1; rep_kind[(arm, kinds[k])] += int(ok)

    print(f"\n=== CE ACTIONABILITY ({args.out}) ===")
    arms = sorted(set(list(per) + list(fired)))
    if not arms:
        print("  (no G-arm logs with CE records found)")
        return
    for arm in arms:
        d = per[arm]
        act = rep[arm] / d if d else float("nan")
        hr = harm[arm] / harm_den[arm] if harm_den[arm] else float("nan")
        print(f"[{arm}] CE fired={fired[arm]} · actionability={act:.2f} "
              f"({rep[arm]}/{d}) · harm_rate={hr:.2f} ({harm[arm]}/{harm_den[arm]})")
        for kind in ("cell", "term", "missing", "extra"):
            dk = per_kind[(arm, kind)]
            if dk:
                print(f"      {kind:8s}: {rep_kind[(arm, kind)]}/{dk} repaired")


if __name__ == "__main__":
    main()
