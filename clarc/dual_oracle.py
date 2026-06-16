"""Offline oracle ($0, no LLM) — does an induced object-contract GENERALIZE?

The over-constraining hazard (HANDOFF §3j, task 1c56ad9f): with only 2-4 train
pairs a bijective object-contract can hold COINCIDENTALLY — true on train, false on
the held-out test — and injecting it as a hard "MUST" steers the generator into an
overfit. This harness measures that hazard for free: induce on train, then check the
induced contracts against the held-out TEST pair (whose solutions are on disk).

For each task / contract the label is:
  - real         : induced contracts are consistent with every (bijective) test pair.
  - coincidental : induced + bijective on test, but the test pair refutes them
                   (apply mismatches, or no consistent matching) — the overfit signal.
  - untestable   : the chosen segmentation isn't bijective on test, or the contract is
                   vacuous on the test grid (excluded from the rate, not counted real).

v1 measures the CURRENT inducer (no confidence gate yet) so the Stage-2 gate can be
tuned against these labels. The three anchor verdicts (1c56ad9f coincidental;
0e671a1a, 11e1fe23 real) are the pre-registered acceptance test for the gate.

    uv run python -m clarc.dual_oracle                       # structural stratum
    uv run python -m clarc.dual_oracle --stratum all --out output/oracle-YYYYMMDD
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np

from clarc import devset
from clarc.objconf import gate, identity_like, tier_names
from clarc.objects import segment
from clarc.objsmt import ObjectContracts, check_output, induce_object_contracts
from clarc.objsolve import apply_contracts
from clarc.run import _load

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_HERE, "..", "output", "oracle")

# Revised anchor acceptance (Stage-1 findings): the over-constraining hazard is mode-2
# (trivial near-identity guidance), and the two "wins" are generation variance, not
# contract-driven. So the GATE must: empty 1c56ad9f's hard tier (no over-constraining),
# and be a no-op where nothing is induced.
ANCHORS = {
    "1c56ad9f": "no_hard",     # gate empties the hard tier -> can't over-constrain
    "0e671a1a": "no_induce",   # induces nothing -> gate is a no-op (G3 == A0)
    "11e1fe23": "no_induce",   # induces nothing (n=2) -> gate is a no-op
}


def _restrict(c: ObjectContracts, name: str) -> ObjectContracts:
    """A copy carrying ONLY contract `name` (same induced params) — to test one term."""
    return ObjectContracts(segmentation=c.segmentation, used=[name], pi=list(c.pi),
                           dr=c.dr, dc=c.dc, ksz=c.ksz)


def _vacuous_on_test(c: ObjectContracts, name: str, test_pairs) -> bool:
    """The parametric, coincidence-prone terms can be untestable on a given test grid
    (the param is never exercised). Conservative: only flag the clear cases."""
    if name == "pos_shift":
        return (c.dr, c.dc) == (0, 0)
    if name == "size_scale":
        return c.ksz == 1
    if name == "color_map":
        # vacuous iff no test input object has a color the map actually moves
        for ti, _ in test_pairs:
            for o in segment(ti, c.segmentation):
                if 0 <= o.color < 10 and c.pi[o.color] != o.color:
                    return False
        return True
    return False


def _holds_on_test(c: ObjectContracts, test_pairs) -> Optional[bool]:
    """True iff the contract HOLDS as an invariant on every bijective test pair; False
    if some test pair refutes it; None if no test pair is bijective under c.segmentation.

    This is the invariant-holds question (`check_output`: does a consistent in↔out
    matching exist under the fixed params), NOT the forward-construct question
    (`apply_contracts` == test_out) — the latter conflates 'the invariant generalizes'
    with 'the menu fully determines the transformation' and is tracked separately."""
    any_testable = False
    for ti, to in test_pairs:
        in_objs = segment(ti, c.segmentation)
        out_objs = segment(to, c.segmentation)
        if len(in_objs) != len(out_objs) or not in_objs:
            continue
        any_testable = True
        if not check_output(in_objs, out_objs, c):
            return False
    return any_testable or None


def _forward_solves_test(c: ObjectContracts, test_pairs) -> bool:
    """Stronger, separate signal: do the contracts forward-CONSTRUCT every test output
    (the deterministic M7b solve)? Used for the no-regression smoke, not the labels."""
    for ti, to in test_pairs:
        fwd = apply_contracts(ti, c)
        if fwd is None or fwd.shape != to.shape or not np.array_equal(fwd, to):
            return False
    return True


def _label(r: Optional[bool]) -> str:
    return {True: "real", False: "coincidental", None: "untestable"}[r]


def analyze_task(train_pairs, test_pairs) -> dict:
    contracts = induce_object_contracts(train_pairs)
    if contracts is None or not contracts.used:
        return {"induced": False, "test_label": "untestable", "gate": "no_induce",
                "contracts": []}
    task_label = _label(_holds_on_test(contracts, test_pairs))
    triv = identity_like(contracts, train_pairs)
    gate(contracts, train_pairs)                     # set per-contract tiers
    hard = tier_names(contracts, "hard")
    soft = tier_names(contracts, "soft")
    drop = tier_names(contracts, "drop")
    per_contract = []
    for name in contracts.used:
        lab = ("untestable" if _vacuous_on_test(contracts, name, test_pairs)
               else _label(_holds_on_test(_restrict(contracts, name), test_pairs)))
        per_contract.append({"name": name, "label": lab, "tier": contracts.tiers.get(name)})
    return {"induced": True, "seg": contracts.segmentation, "used": list(contracts.used),
            "render": contracts.render(), "test_label": task_label,
            "forward_solves_test": _forward_solves_test(contracts, test_pairs),
            "identity_like": triv, "gate": ("no_hard" if not hard else "hard:" + ",".join(hard)),
            "hard": hard, "soft": soft, "drop": drop, "contracts": per_contract}


def run(stratum: str, data: str, out_dir: str) -> dict:
    challenges, solutions = _load(data)
    if solutions is None:
        raise SystemExit(f"--data {data} has no solutions file (need held-out test outputs)")
    d = devset.load()
    ids = (d["structural"] + d["logic"]) if stratum == "all" else d[stratum]

    tasks: dict[str, dict] = {}
    for tid in ids:
        if tid not in challenges or tid not in solutions:
            continue
        t = challenges[tid]
        train_pairs = [(np.asarray(e["input"], int), np.asarray(e["output"], int))
                       for e in t["train"]]
        test_pairs = [(np.asarray(e["input"], int), np.asarray(sol, int))
                      for e, sol in zip(t["test"], solutions[tid])]
        rec = analyze_task(train_pairs, test_pairs)
        rec["n_train"] = len(train_pairs)
        tasks[tid] = rec

    induced = [r for r in tasks.values() if r["induced"]]
    labels = [r["test_label"] for r in induced]
    real = labels.count("real")
    coinc = labels.count("coincidental")
    untest = labels.count("untestable")
    denom = real + coinc
    rollup = {"n_tasks": len(tasks), "induced": len(induced),
              "real": real, "coincidental": coinc, "untestable": untest,
              "coincidental_rate": round(coinc / denom, 3) if denom else None,
              "identity_like": sum(1 for r in induced if r.get("identity_like")),
              "with_hard_tier": sum(1 for r in induced if r.get("hard"))}
    anchors = {tid: {"want": want, "got": tasks.get(tid, {}).get("gate", "missing"),
                     "render": tasks.get(tid, {}).get("render", "")}
               for tid, want in ANCHORS.items()}
    report = {"stratum": stratum, "data": data, "rollup": rollup,
              "anchors": anchors, "tasks": tasks}

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return report


def _print(report: dict) -> None:
    ru = report["rollup"]
    print(f"\n=== DUAL ORACLE ({report['stratum']}, {report['data']}) ===")
    print(f"induced on {ru['induced']}/{ru['n_tasks']} · test: real={ru['real']} "
          f"coincidental={ru['coincidental']} untestable={ru['untestable']} "
          f"(coincidental_rate={ru['coincidental_rate']})")
    print(f"gate: identity_like={ru['identity_like']}/{ru['induced']} · "
          f"tasks with a HARD tier={ru['with_hard_tier']}/{ru['induced']} "
          f"(rest inject only soft/deferring hints — the mode-2 fix)")
    print("\nanchors (want -> got gate verdict):")
    ok = True
    for tid, a in report["anchors"].items():
        mark = "✓" if a["got"] == a["want"] else "✗"
        ok = ok and a["got"] == a["want"]
        print(f"  {mark} {tid}: want={a['want']:10s} got={a['got']:20s} {a['render']}")
    print("\nper-task (induced only)  [test_label | gate]:")
    for tid, r in report["tasks"].items():
        if not r["induced"]:
            continue
        tiers = "".join("H" if r["contracts"] and c["tier"] == "hard" else "" for c in r["contracts"])
        cs = " ".join(f"{c['name']}:{(c['tier'] or '-')[:4]}" for c in r["contracts"])
        print(f"  [{r['test_label']:12s}|{r['gate']:14s}] {tid} n={r['n_train']} "
              f"seg={r.get('seg','?')} triv={int(r.get('identity_like', False))}")
        print(f"        {cs}")
    print(f"\nanchors all match: {ok}")


def main() -> None:
    ap = argparse.ArgumentParser(description="offline object-contract generalization oracle")
    ap.add_argument("--data", default="2024-eval")
    ap.add_argument("--stratum", default="structural", choices=["structural", "logic", "all"])
    ap.add_argument("--out", default=_OUT)
    args = ap.parse_args()
    _print(run(args.stratum, args.data, args.out))


if __name__ == "__main__":
    main()
