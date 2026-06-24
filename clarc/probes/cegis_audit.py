"""CEGIS faithfulness audit ($0) — is a run ACTUALLY a CEGIS / type-directed loop?

A run only counts as a real test of the hypothesis if all four defining criteria are
NON-TRIVIALLY active. From the per-iteration DSL-arm logs this reports:

  C1 typed-DSL candidates : %generations that parsed to a typed DSL pipeline
  C2 traceable feedback   : %refutations whose unsat core names BOTH a `slot:<prim>`
                            (the transformation) and a `fact:<group>` (the violated
                            contract) — i.e. the error is blamed on a step + a property
  C3 strengthening lattice: clauses learned (monotone store) + clause reuse (a learned
                            clause firing on a later candidate)
  C4 subspace pruning     : Σ skeletons blocked (class_pruned) AND whether SYNTH drew
                            candidates from the pruned space (synth_seeded) — the piece
                            that is dormant pre-Stage-1

Plus an F3 triviality probe on induced-primitive contracts (template `name:relation`):
the share of components that are mere preservation (`eq`) vs informative transforms
(`const`/`mul`/`le`/`ge`) — an all-`eq` contract is identity-like (trivial).

    uv run python -m clarc.probes.cegis_audit --out output/exp-dsl-haiku [--arms D2,E0]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter


def _traceable(core: list[str]) -> bool:
    has_slot = any(c.startswith("slot:") for c in core)
    has_fact = any(c.startswith("fact:") for c in core)
    return has_slot and has_fact


def _contract_relations(contract: str) -> Counter:
    rel = Counter()
    for comp in (contract or "").split(","):
        comp = comp.strip()
        if ":" not in comp:
            continue
        r = comp.split(":", 1)[1]
        kind = ("const" if r.startswith("const") else "mul" if r.startswith("mul")
                else r)   # eq | le | ge | free | const | mul
        rel[kind] += 1
    return rel


def audit(out_dir: str, arms: list[str]) -> dict:
    paths = []
    for a in arms:
        paths += glob.glob(os.path.join(out_dir, "logs", f"*_{a}.json"))
    paths = sorted(set(paths))

    cand = valid = 0
    refuted = traceable = 0
    pruned_total = synth_seeded = synth_feasible = 0
    clause_tasks = []        # n_clauses per task (lattice size)
    reuse_total = 0
    induced = []             # (name, relation Counter)
    solved = Counter(); ntask = Counter()

    for p in paths:
        arm = os.path.basename(p).rsplit("_", 1)[1].replace(".json", "")
        log = json.load(open(p, encoding="utf-8"))
        ntask[arm] += 1
        solved[arm] += int(bool(log.get("solved")))
        clause_tasks.append((arm, log.get("n_clauses", 0)))
        reuse_total += log.get("clause_reuse", 0)
        synth_feasible += log.get("n_synth_feasible", 0)
        for rec in log.get("records", []):
            if rec.get("dsl_text") is not None:
                cand += 1
                if rec.get("stage") != "dsl_invalid":
                    valid += 1
            if rec.get("refuted"):
                refuted += 1
                if _traceable(rec.get("refutation_core", [])):
                    traceable += 1
            pruned_total += rec.get("class_pruned", 0) or 0
            if rec.get("synth_seeded") or rec.get("stage") == "synth":   # set once Stage 1 lands
                synth_seeded += 1
        for ip in log.get("induced_prims", []):
            induced.append((ip.get("name"), _contract_relations(ip.get("contract", ""))))

    return {"arms": arms, "n_logs": len(paths),
            "C1_dsl": (valid, cand), "C2_trace": (traceable, refuted),
            "C3_clauses": clause_tasks, "C3_reuse": reuse_total,
            "C4_pruned": pruned_total, "C4_synth_seeded": synth_seeded,
            "C4_synth_feasible": synth_feasible,
            "induced": induced, "solved": dict(solved), "ntask": dict(ntask)}


def _print(r: dict) -> None:
    print(f"\n=== CEGIS FAITHFULNESS AUDIT ({r['n_logs']} logs, arms={r['arms']}) ===")
    v, c = r["C1_dsl"]
    print(f"C1 typed-DSL candidates : {v}/{c} ({100*v//max(1,c)}%) parsed to a typed pipeline")
    t, rf = r["C2_trace"]
    print(f"C2 traceable feedback   : {t}/{rf} refutations name BOTH slot:<prim> + fact:<group> "
          f"({100*t//max(1,rf)}%)")
    sizes = [n for _, n in r["C3_clauses"]]
    print(f"C3 strengthening lattice: clauses/task max={max(sizes or [0])} "
          f"mean={sum(sizes)/max(1,len(sizes)):.1f}; clause reuse total={r['C3_reuse']}")
    sf = r.get("C4_synth_feasible", 0)
    print(f"C4 subspace pruning     : Σ skeletons blocked={r['C4_pruned']:,}; "
          f"SYNTH feasible draws={sf}, synth-solves={r['C4_synth_seeded']}  "
          f"{'<-- DORMANT (criterion 4 only PARTIAL)' if not (sf or r['C4_synth_seeded']) else '<-- ACTIVE'}")
    print(f"solve: {r['solved']} of {r['ntask']}")
    if r["induced"]:
        print(f"\nF3 triviality probe — {len(r['induced'])} induced-prim contracts:")
        for name, rel in r["induced"]:
            tot = sum(rel.values()) or 1
            transforming = rel.get("const", 0) + rel.get("mul", 0)
            print(f"  {name}: {dict(rel)}  -> {100*rel.get('eq',0)//tot}% preservation(eq), "
                  f"{transforming} transforming(const/mul)"
                  f"{'  [TRIVIAL-leaning]' if transforming <= 1 else ''}")
    v, c = r["C1_dsl"]
    c4_active = bool(r.get("C4_synth_feasible", 0) or r["C4_synth_seeded"])
    print(f"\nverdict: criterion 4 (generate-from-pruned-space) = "
          f"{'ACTIVE' if c4_active else 'DORMANT'}"
          f"{'  — C1-C3 are 0 because the generator was a STUB; rerun with an LLM' if v == 0 else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description="CEGIS faithfulness audit ($0)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="D0,D1,D2,E0,E1")
    args = ap.parse_args()
    _print(audit(args.out, args.arms.split(",")))


if __name__ == "__main__":
    main()
