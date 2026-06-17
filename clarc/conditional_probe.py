"""Lever-A go/no-go ($0): can a PARTIAL CONDITIONAL contract refute the wrong candidates
that the envelope verifiers can't see?

Envelope verifiers (object preservation, σ) fire on 2% of wrong iterations because the
errors are CONTENT: a valid recolor with the wrong colors. A conditional contract keys
the output color on a NON-color input attribute (size / size-rank / shape / border /
holes) — "output color = f(attribute)" that holds on every train pair. If a wrong
candidate assigns a color inconsistent with that train-induced map, it is REFUTABLE — the
counterexample the envelope can't produce.

This probe induces such maps on the logged failing tasks and replays the logged wrong
candidates to measure the refute-rate. Decision: a meaningful refute-rate ⇒ build Lever A
(induction + refutation + witness); ~0 ⇒ the rule needs global reasoning no object-local
conditional pins ⇒ Lever A is a dead end, go to Lever B (structurally-hard band).

    uv run python -m clarc.conditional_probe --out output/exp-guided-gate
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from clarc.objects import SEGMENTERS, segment
from clarc.run import _load


def _majority(grid: np.ndarray, mask: np.ndarray) -> int:
    vals = grid[mask]
    if vals.size == 0:
        return -1
    v, c = np.unique(vals, return_counts=True)
    return int(v[c.argmax()])


def _ranks(objs):
    order = sorted(range(len(objs)), key=lambda i: -objs[i].size)
    rank = [0] * len(objs)
    for r, i in enumerate(order):
        rank[i] = r
    return rank


# key(object, rank_desc, n_objects) -> hashable attribute the output color may depend on
_KEYS = {
    "in_color":   lambda o, r, n: o.color,
    "size":       lambda o, r, n: o.size,
    "rank_desc":  lambda o, r, n: r,          # 0 = largest  (global/relational)
    "rank_asc":   lambda o, r, n: n - 1 - r,  # 0 = smallest
    "shape":      lambda o, r, n: o.shape,
    "is_border":  lambda o, r, n: o.is_border,
    "n_holes":    lambda o, r, n: o.n_holes,
    "bbox":       lambda o, r, n: (o.bh, o.bw),
}


def induce_conditional(train_pairs):
    """Return [(keyname, seg, map)] for every (segmentation, attribute) whose
    attribute→output-color map is a consistent, non-trivial, discriminating FUNCTION on
    all train pairs (≥2 distinct keys, and at least one object actually recolored)."""
    conds = []
    # conditional recolor is defined only when output keeps the input's grid (read the
    # output color at the input object's cells); shape-changing tasks are out of scope.
    if any(np.asarray(gi).shape != np.asarray(go).shape for gi, go in train_pairs):
        return conds
    for seg in SEGMENTERS:
        try:
            segd = [(segment(np.asarray(gi), seg), np.asarray(go)) for gi, go in train_pairs]
        except (ValueError, KeyError):
            continue
        if any(len(objs) == 0 for objs, _ in segd):
            continue
        for kn, kf in _KEYS.items():
            d, ok, nontrivial = {}, True, False
            for objs, go in segd:
                rks, n = _ranks(objs), len(objs)
                for o, rk in zip(objs, rks):
                    k = kf(o, rk, n)
                    oc = _majority(go, o.mask)
                    if k in d and d[k] != oc:
                        ok = False
                        break
                    d[k] = oc
                    nontrivial = nontrivial or (oc != o.color)
                if not ok:
                    break
            if ok and nontrivial and len(d) >= 2:
                conds.append((kn, seg, dict(d)))
    return conds


def _loo_robust(conds, train_pairs):
    """Keep only conditionals that survive leave-one-out: re-induced from the remaining
    pairs AND correctly predict the held-out pair (the generalization gate). Needs ≥3
    pairs; with fewer, nothing is robust (can't tell coincidence from rule)."""
    if len(train_pairs) < 3:
        return []
    robust = []
    for kn, seg, d in conds:
        ok = True
        for i in range(len(train_pairs)):
            sub = train_pairs[:i] + train_pairs[i + 1:]
            sub_conds = [(k, s, m) for k, s, m in induce_conditional(sub) if k == kn and s == seg]
            if not sub_conds:
                ok = False
                break
            _, _, dsub = sub_conds[0]
            gi, go = (np.asarray(x) for x in train_pairs[i])
            objs = segment(gi, seg); rks, n = _ranks(objs), len(objs)
            for o, rk in zip(objs, rks):
                k = _KEYS[kn](o, rk, n)
                if k in dsub and _majority(go, o.mask) != dsub[k]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            robust.append((kn, seg, d))
    return robust


def _forward_solves(conds, train_pairs):
    """Does some single conditional reconstruct every train output exactly (M7c solve)?"""
    for kn, seg, d in conds:
        good = True
        for gi, go in train_pairs:
            gi, go = np.asarray(gi), np.asarray(go)
            objs = segment(gi, seg); rks, n = _ranks(objs), len(objs)
            pred = gi.copy()
            for o, rk in zip(objs, rks):
                k = _KEYS[kn](o, rk, n)
                if k in d:
                    pred[o.mask] = d[k]
            if not np.array_equal(pred, go):
                good = False
                break
        if good:
            return kn
    return None


def conditional_refutes(conds, gi: np.ndarray, cand: np.ndarray) -> bool:
    if gi.shape != cand.shape:                  # wrong dims — an envelope error, not ours
        return False
    for kn, seg, d in conds:
        objs = segment(gi, seg); rks, n = _ranks(objs), len(objs)
        for o, rk in zip(objs, rks):
            k = _KEYS[kn](o, rk, n)
            if k in d and _majority(cand, o.mask) != d[k]:
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Lever-A conditional-contract refute probe ($0)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="2024-eval")
    args = ap.parse_args()
    challenges, solutions = _load(args.data)

    # tasks A0 got wrong (the headroom)
    rows = {}
    for line in open(os.path.join(args.out, "runs.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("phase") == "B":
            rows.setdefault(r["task"], {})[r["arm"]] = r
    fails = [t for t in rows if rows[t].get("G0", {}).get("acc", 1) < 1.0]

    print(f"\n=== LEVER-A conditional-contract probe ({args.out}) ===")
    print("induce output-color = f(non-color attribute) on train; replay logged wrong "
          "candidates.\n")
    tot_wrong = tot_ref = tot_safe = 0
    for tid in sorted(fails):
        tr = challenges[tid]["train"]
        tp = [(e["input"], e["output"]) for e in tr]
        conds = induce_conditional(tp)
        robust = _loo_robust(conds, tp)
        keys = sorted({kn for kn, _, _ in conds})
        rkeys = sorted({kn for kn, _, _ in robust})
        solves = _forward_solves(conds, tp) if conds else None
        # generalization sanity: does the best conditional also hold on the held-out test?
        gen = None
        if conds and solutions and tid in solutions:
            te = challenges[tid]["test"]
            tpe = [(e["input"], sol) for e, sol in zip(te, solutions[tid])]
            gen = _forward_solves(conds, tpe) is not None
        # replay wrong candidates from G1's log (ungated, all iters)
        p = os.path.join(args.out, "logs", f"B_{tid}_G1.json")
        nw = nr = ns = 0
        if os.path.exists(p) and conds:
            log = json.load(open(p, encoding="utf-8"))
            for rec in log.get("records", []):
                if rec.get("stage") == "solved" or rec.get("produced") is None:
                    continue
                nw += 1
                pg = rec["produced"]
                if any(g is not None and conditional_refutes(conds, np.asarray(ti), np.asarray(g))
                       for (ti, _), g in zip(tp, pg)):
                    nr += 1
                if robust and any(g is not None and conditional_refutes(robust, np.asarray(ti), np.asarray(g))
                                  for (ti, _), g in zip(tp, pg)):
                    ns += 1
        tot_wrong += nw; tot_ref += nr; tot_safe += ns
        print(f"  {tid}: cond_keys={keys or '∅'}  LOO_robust={rkeys or '∅'}  solves={solves}  "
              f"test_gen={gen}  wrong={nw}  cond-CE={nr}  LOO-safe-CE={ns}")
    if tot_wrong:
        print(f"\nTOTAL wrong iters={tot_wrong} · conditional-CE refutes={tot_ref} "
              f"({100 * tot_ref // tot_wrong}%) · LOO-SAFE refutes={tot_safe} "
              f"({100 * tot_safe // tot_wrong}%) · envelope baseline=2%")
        print("  Raw-high but LOO-safe~0 ⇒ conditionals FIRE but are coincidental: a content CE "
              "needs a generalization gate, and at n≤4 there may be nothing safe to fire.")


if __name__ == "__main__":
    main()
