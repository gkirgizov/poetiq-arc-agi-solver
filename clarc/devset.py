"""Curate a deterministic, stratified dev set from ARC-AGI-1 (2024) evaluation.

Two strata:
  - structural : shape-preserving on all train pairs, small grids, and spec
    extraction yields >=1 contract -> the regime where contracts CAN help.
  - logic      : shape-changing tasks (the rule, not a structural invariant, is the
    difficulty) -> the control where we expect ~0 contract lift and require no harm.

Selection is deterministic (sorted task ids) and cached to devset_ids.json so every
run uses the identical set. Run `uv run python -m clarc.devset` to (re)generate.
"""

from __future__ import annotations

import json
import os

from clarc.spec import extract_spec

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_EVAL = os.path.join(_REPO, "data", "arc-prize-2024", "arc-agi_evaluation_challenges.json")
_CACHE = os.path.join(_HERE, "devset_ids.json")

STRUCT_MAX = 15   # max grid dim for the structural stratum (dense signal)
LOGIC_MAX = 20    # max grid dim for the logic stratum (bound prompt/cost)
PER_STRATUM = 20


def _dims_ok(task, cap):
    for ex in task["train"]:
        for grid in (ex["input"], ex["output"]):
            if len(grid) > cap or len(grid[0]) > cap:
                return False
    return True


def _shape_preserving(task):
    return all(
        len(ex["input"]) == len(ex["output"]) and len(ex["input"][0]) == len(ex["output"][0])
        for ex in task["train"]
    )


def curate(eval_path: str = _EVAL, per_stratum: int = PER_STRATUM) -> dict:
    with open(eval_path, encoding="utf-8") as f:
        tasks = json.load(f)

    structural, logic = [], []
    for tid in sorted(tasks):
        t = tasks[tid]
        ti = [e["input"] for e in t["train"]]
        to = [e["output"] for e in t["train"]]
        if _shape_preserving(t) and _dims_ok(t, STRUCT_MAX):
            spec = extract_spec(ti, to, tid)
            if not spec.is_empty():
                structural.append(tid)
        elif not _shape_preserving(t) and _dims_ok(t, LOGIC_MAX):
            logic.append(tid)

    return {
        "structural": structural[:per_stratum],
        "logic": logic[:per_stratum],
        "_counts": {"structural_pool": len(structural), "logic_pool": len(logic)},
    }


def load() -> dict:
    if os.path.exists(_CACHE):
        with open(_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return curate()


def all_ids(devset: dict | None = None) -> list[str]:
    d = devset or load()
    return list(d["structural"]) + list(d["logic"])


def stratum_of(tid: str, devset: dict | None = None) -> str:
    d = devset or load()
    if tid in d["structural"]:
        return "structural"
    if tid in d["logic"]:
        return "logic"
    return "other"


def main() -> None:
    d = curate()
    with open(_CACHE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    print(f"structural pool={d['_counts']['structural_pool']} -> picked {len(d['structural'])}")
    print(f"logic pool={d['_counts']['logic_pool']} -> picked {len(d['logic'])}")
    print(f"wrote {_CACHE}")


if __name__ == "__main__":
    main()
