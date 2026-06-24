"""Confidence gate for induced object-contracts — keep the dual ENHANCING, not
OVER-CONSTRAINING (HANDOFF §3j; oracle findings 2026-06-15).

The Stage-1 oracle showed the over-constraining hazard here is mode-2 (a coarse
near-identity "preserve everything" block that misguides the generator, e.g.
1c56ad9f under `by_row`), NOT mode-1 (a coincidental-FALSE bijection — rare, ~8%).
So this gate is mode-2-first: it splits induced contracts into tiers

  hard : trustworthy, informative -> inject firmly ("the output should satisfy …")
  soft : holds on train but weak/uninformative -> inject as a hint that DEFERS to
         the examples (so a near-identity block can't anchor the generator)
  drop : vacuous (identity params) -> never mention

Primary signals (cheap, computed from the segmented INPUT objects + induced params,
so no z3 matching needs surfacing):
  - vacuity              : color_map=id / size_scale(x1) / pos_shift(0,0) -> drop.
  - identity_like (set)  : the contract set does NOT forbid output==input on a pair
                           where the task DOES change something -> trivial -> all soft.
  - forward_solves (set) : contracts forward-CONSTRUCT every train output (M7b) -> we
                           captured the rule -> promote all non-vacuous to hard.
  - support (per term)   : # input objects that exercise a TRANSFORMING term -> a real,
                           well-supported recolor/shift/scale (support>=2) can be hard.
  - LOO (minor tiebreak) : at n>=3, demote a would-be-hard transforming term that
                           doesn't survive leave-one-out (catches the rare mode-1).
"""

from __future__ import annotations

import numpy as np

from clarc.objects.base import segment
from clarc.objects.smt import ObjectContracts, check_output, induce_object_contracts
from clarc.objects.solve import apply_contracts

TRANSFORMING = {"color_map", "pos_shift", "size_scale"}


def _vacuous(name: str, c: ObjectContracts) -> bool:
    """A transforming term whose induced parameter is the identity carries no info."""
    if name == "color_map":
        return all(c.pi[k] == k for k in range(10))
    if name == "size_scale":
        return c.ksz == 1
    if name == "pos_shift":
        return (c.dr, c.dc) == (0, 0)
    return False


def _support(name: str, c: ObjectContracts, train_pairs) -> int:
    """# input objects that NON-VACUOUSLY exercise `name`, summed over train pairs."""
    n = 0
    for ti, _ in train_pairs:
        objs = segment(np.asarray(ti), c.segmentation)
        if name == "color_map":
            n += sum(1 for o in objs if 0 <= o.color < 10 and c.pi[o.color] != o.color)
        elif name == "pos_shift":
            n += len(objs) if (c.dr, c.dc) != (0, 0) else 0
        elif name == "size_scale":
            n += sum(1 for o in objs if o.size > 1) if c.ksz != 1 else 0
        elif name in ("shape_exact", "bbox_preserved"):
            n += sum(1 for o in objs if o.bh * o.bw > 1)
        elif name == "shape_canon":
            n += sum(1 for o in objs if o.shape == "other" or o.n_holes > 0 or o.bh != o.bw)
        elif name == "color_preserved":
            n += len(objs) if len({o.color for o in objs}) >= 2 else 0
        elif name == "size_preserved":
            n += len(objs) if len({o.size for o in objs}) >= 2 else 0
        elif name == "pos_preserved":
            n += len(objs) if len(objs) >= 2 else 0
    return n


def _color_map_entries(c: ObjectContracts, train_pairs) -> int:
    """# distinct colors actually remapped (a→π[a]≠a) by some train input object — a π
    justified by a single color is the classic coincidence."""
    seen = set()
    for ti, _ in train_pairs:
        for o in segment(np.asarray(ti), c.segmentation):
            if 0 <= o.color < 10 and c.pi[o.color] != o.color:
                seen.add(o.color)
    return len(seen)


def identity_like(c: ObjectContracts, train_pairs) -> bool:
    """True iff the contract set ADMITS output==input on every pair that actually
    changes — i.e. it fails to force any transformation (the mode-2 hazard). A set
    with a real transforming term rejects the no-op and is NOT identity_like."""
    saw_change = False
    for ti, to in train_pairs:
        ti, to = np.asarray(ti), np.asarray(to)
        if ti.shape == to.shape and np.array_equal(ti, to):
            continue                                   # nothing to detect on this pair
        saw_change = True
        objs = segment(ti, c.segmentation)
        if not objs:
            return False
        if not check_output(objs, objs, c):            # set FORBIDS the no-op -> informative
            return False
    return saw_change                                  # admitted no-op everywhere a change exists


def forward_solves(c: ObjectContracts, train_pairs) -> bool:
    """Contracts forward-construct every train output (the M7b deterministic solve)."""
    for ti, to in train_pairs:
        to = np.asarray(to)
        fwd = apply_contracts(np.asarray(ti), c)
        if fwd is None or fwd.shape != to.shape or not np.array_equal(fwd, to):
            return False
    return True


def _loo_stable(name: str, c: ObjectContracts, train_pairs) -> bool:
    """Minor tiebreak (n>=3): is `name` re-induced with the SAME role on every
    leave-one-out subset? Demotes a would-be-hard transforming term that only the
    full (tiny) train set supports. Returns True when LOO can't run (<3 pairs)."""
    if len(train_pairs) < 3:
        return True
    for i in range(len(train_pairs)):
        sub = train_pairs[:i] + train_pairs[i + 1:]
        ci = induce_object_contracts([(np.asarray(a), np.asarray(b)) for a, b in sub])
        if ci is None or name not in ci.used or _vacuous(name, ci):
            return False
        if name == "color_map" and any(ci.pi[k] != c.pi[k] for k in range(10)):
            return False           # the map flips across folds -> coincidental
        if name == "pos_shift" and (ci.dr, ci.dc) != (c.dr, c.dc):
            return False
        if name == "size_scale" and ci.ksz != c.ksz:
            return False
    return True


def gate(c: ObjectContracts, train_pairs) -> ObjectContracts:
    """Set c.tiers[name] for each used contract (mode-2-first policy). Mutates and
    returns c. Idempotent given the same train_pairs."""
    solves = forward_solves(c, train_pairs)
    trivial = (not solves) and identity_like(c, train_pairs)
    tiers: dict[str, str] = {}
    for name in c.used:
        if _vacuous(name, c):
            tiers[name] = "drop"
        elif solves:
            tiers[name] = "hard"                       # the whole rule is captured -> trust it
        elif trivial:
            tiers[name] = "soft"                       # near-identity block -> never hard
        elif name in TRANSFORMING and _support(name, c, train_pairs) >= 2 \
                and (name != "color_map" or _color_map_entries(c, train_pairs) >= 2) \
                and _loo_stable(name, c, train_pairs):
            tiers[name] = "hard"                       # a real, well-supported transformation
        else:
            tiers[name] = "soft"
    c.tiers = tiers
    return c


def tier_names(c: ObjectContracts, tier: str) -> list[str]:
    """Contract names at a given tier (preserving `used` order). Ungated (empty tiers)
    treats everything as 'hard' so callers degrade to current behavior."""
    if not c.tiers:
        return list(c.used) if tier == "hard" else []
    return [n for n in c.used if c.tiers.get(n) == tier]


def diagnostics(c: ObjectContracts, train_pairs) -> dict:
    """Per-contract signals for the offline oracle / logs (no gating side effects)."""
    return {
        "forward_solves": forward_solves(c, train_pairs),
        "identity_like": identity_like(c, train_pairs),
        "contracts": {n: {"vacuous": _vacuous(n, c), "support": _support(n, c, train_pairs),
                          "tier": c.tiers.get(n)} for n in c.used},
    }
