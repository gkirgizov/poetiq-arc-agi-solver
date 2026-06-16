"""Object-correspondence SMT — the logical dual of an object-level transformation.

The transformation is described not by code but by CONTRACTS over an explicit
input-object ↔ output-object MATCHING: which objects correspond, and what is
preserved or mapped between matched pairs (shape, color, position, size). The z3
solver owns the reasoning — it finds the matching and the shared contract
parameters (a global color map, a global position shift, a size scale) that make
all train pairs consistent (INDUCTION), and later decides whether a candidate
output's objects can be matched under those contracts (REFUTATION). The LLM stays
the recognizer + generator; this module is the reasoner.

Scope (M7a): bijective matching (|input objects| == |output objects| per pair).
Object count changes (drop/create) are a later extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import z3
from scipy.optimize import linear_sum_assignment

from clarc.objects import MAX_OBJECTS, Object, SEGMENTERS, segment

# Contract menu: name -> (param_kind, weight). Tighter contracts score higher so
# induction prefers the segmentation/contract-set that most sharply pins the map.
_MENU = {
    "shape_exact":    ("none", 4),   # matched objects have identical bbox pattern
    "shape_canon":    ("none", 2),   # ...identical up to rotation/reflection
    "color_preserved":("none", 2),
    "color_map":      ("pi", 3),     # out.color == PI[in.color], PI shared
    "size_preserved": ("none", 2),
    "size_scale":     ("k", 2),      # out.size == K * in.size, K shared
    "pos_preserved":  ("none", 2),
    "pos_shift":      ("shift", 2),  # out.(top,left) == in.(top,left)+(DR,DC)
    "bbox_preserved": ("none", 1),
}


@dataclass
class ObjectContracts:
    segmentation: str
    used: list[str]
    pi: list[int]                       # color map (len 10), valid iff color_map used
    dr: int = 0
    dc: int = 0
    ksz: int = 1
    score: float = 0.0
    counts_ok: bool = True              # every train pair was bijective
    tiers: dict = field(default_factory=dict)   # name -> "hard"|"soft"|"drop" (set by objconf.gate;
                                                # empty = ungated, treat all `used` as injected)

    def render(self) -> str:
        parts = []
        for u in self.used:
            if u == "color_map":
                m = {c: self.pi[c] for c in range(10) if self.pi[c] != c}
                parts.append(f"color_map{m or '=id'}")
            elif u == "pos_shift":
                parts.append(f"pos_shift({self.dr},{self.dc})")
            elif u == "size_scale":
                parts.append(f"size_scale(x{self.ksz})")
            else:
                parts.append(u)
        return f"[seg={self.segmentation}] " + ", ".join(parts)


def _obj_ints(o: Object) -> dict:
    return {"color": o.color, "size": o.size, "top": o.top, "left": o.left,
            "bh": o.bh, "bw": o.bw, "shape": o.shape_hash, "canon": o.shape_canon}


def _contract_term(name, ai, bj, P):
    """The z3 relation for contract `name` on matched (input ai, output bj)."""
    if name == "shape_exact":
        return bj["shape"] == ai["shape"]
    if name == "shape_canon":
        return bj["canon"] == ai["canon"]
    if name == "color_preserved":
        return bj["color"] == ai["color"]
    if name == "color_map":
        return z3.Select(P["pi"], ai["color"]) == bj["color"]
    if name == "size_preserved":
        return bj["size"] == ai["size"]
    if name == "size_scale":
        return bj["size"] == P["k"] * ai["size"]
    if name == "pos_preserved":
        return z3.And(bj["top"] == ai["top"], bj["left"] == ai["left"])
    if name == "pos_shift":
        return z3.And(bj["top"] == ai["top"] + P["dr"], bj["left"] == ai["left"] + P["dc"])
    if name == "bbox_preserved":
        return z3.And(bj["bh"] == ai["bh"], bj["bw"] == ai["bw"])
    raise KeyError(name)


def _alloc_params() -> dict:
    pi = z3.Array("pi", z3.IntSort(), z3.IntSort())
    P = {"pi": pi, "dr": z3.Int("dr"), "dc": z3.Int("dc"), "k": z3.Int("k")}
    return P


def _param_domain(P):
    cs = [P["dr"] >= -30, P["dr"] <= 30, P["dc"] >= -30, P["dc"] <= 30,
          P["k"] >= 1, P["k"] <= 9]
    for c in range(10):
        cs += [z3.Select(P["pi"], c) >= 0, z3.Select(P["pi"], c) <= 9]
    return cs


def _matching(s, n: int, tag: str):
    """Bijective matching vars + constraints for an n×n correspondence."""
    M = [[z3.Bool(f"m_{tag}_{i}_{j}") for j in range(n)] for i in range(n)]
    for i in range(n):
        s.add(z3.PbEq([(M[i][j], 1) for j in range(n)], 1))
    for j in range(n):
        s.add(z3.PbEq([(M[i][j], 1) for i in range(n)], 1))
    return M


def induce_object_contracts(train_pairs, *, timeout_ms: int = 4000):
    """Try each segmentation; for the bijective ones, find (via z3.Optimize) the
    strongest menu contracts + shared params admitting a consistent matching on
    ALL train pairs. Return the best ObjectContracts, or None."""
    seg_objs = {}
    for strat in SEGMENTERS:
        ok = True
        per_pair = []
        for gi, go in train_pairs:
            oi, oo = segment(gi, strat), segment(go, strat)
            if len(oi) != len(oo) or not (1 <= len(oi) <= MAX_OBJECTS):
                ok = False
                break
            per_pair.append(([_obj_ints(o) for o in oi], [_obj_ints(o) for o in oo]))
        if ok and per_pair:
            seg_objs[strat] = per_pair

    best = None
    for strat, pairs in seg_objs.items():
        res = _induce_for_seg(pairs, timeout_ms)
        if res is not None and (best is None or res.score > best.score):
            res.segmentation = strat
            best = res
    return best


def _induce_for_seg(pairs, timeout_ms):
    opt = z3.Optimize()
    opt.set("timeout", timeout_ms)
    P = _alloc_params()
    opt.add(*_param_domain(P))
    use = {c: z3.Bool(f"use_{c}") for c in _MENU}
    for p, (oi, oo) in enumerate(pairs):
        n = len(oi)
        M = _matching(opt, n, f"p{p}")
        for i in range(n):
            for j in range(n):
                for c in _MENU:
                    opt.add(z3.Implies(z3.And(use[c], M[i][j]),
                                       _contract_term(c, oi[i], oo[j], P)))
    opt.maximize(z3.Sum(*[z3.If(use[c], _MENU[c][1], 0) for c in _MENU]))
    # secondary (lower-priority) objective: leave the color map as IDENTITY where
    # train doesn't constrain it, so colors unseen in train pass through unchanged.
    opt.maximize(z3.Sum(*[z3.If(z3.Select(P["pi"], c) == c, 1, 0) for c in range(10)]))
    if opt.check() != z3.sat:
        return None
    m = opt.model()
    used = [c for c in _MENU if z3.is_true(m[use[c]])]
    if not used:
        return None
    pi = [(m.eval(z3.Select(P["pi"], c)).as_long() if m else c) for c in range(10)]
    score = sum(_MENU[c][1] for c in used)
    return ObjectContracts(segmentation="", used=used, pi=pi,
                           dr=_intval(m, P["dr"]), dc=_intval(m, P["dc"]),
                           ksz=_intval(m, P["k"], 1), score=score)


def _intval(m, v, default=0):
    e = m.eval(v, model_completion=True)
    return e.as_long() if e is not None else default


def check_output(in_objs: list[Object], out_objs: list[Object],
                 contracts: ObjectContracts, *, timeout_ms: int = 2000) -> bool:
    """Is there a bijective matching of these objects consistent with the induced
    contracts (fixed params)? False ⇒ the candidate output is REFUTED."""
    if len(in_objs) != len(out_objs) or not in_objs:
        return False
    oi = [_obj_ints(o) for o in in_objs]
    oo = [_obj_ints(o) for o in out_objs]
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    P = _alloc_params()
    s.add(*_param_domain(P))
    # pin the induced params
    for c in range(10):
        s.add(z3.Select(P["pi"], c) == contracts.pi[c])
    s.add(P["dr"] == contracts.dr, P["dc"] == contracts.dc, P["k"] == contracts.ksz)
    n = len(oi)
    M = _matching(s, n, "chk")
    for i in range(n):
        for j in range(n):
            for c in contracts.used:
                s.add(z3.Implies(M[i][j], _contract_term(c, oi[i], oo[j], P)))
    return s.check() == z3.sat


# --------------------------------------------------------------------------- #
# Witness-decoding refutation: turn a REFUTED candidate into a per-object, per-term
# "this should be X, but it is Y" — the actionable CEGIS repair signal. The decision
# (is the candidate refuted?) stays with check_output / count; this only EXPLAINS an
# already-decided refutation, so a heuristic match can only weaken the wording, never
# fabricate a counterexample.
# --------------------------------------------------------------------------- #

@dataclass
class TermViolation:
    example: int            # 1-based train example index
    kind: str               # "term" | "cell" | "extra" | "missing"
    term: str               # menu term name, or "object_count" / "cell"
    expected: str           # what the rule says this object/cell SHOULD be
    actual: str             # what the candidate produced
    in_key: object = None   # (top,left) of the INPUT object — stable across re-segmentation
    out_pos: object = None  # (top,left) of the PRODUCED object / (r,c) of the cell
    n_objects_hit: int = 1


def _term_violated(name: str, ai: Object, bj: Object, c: ObjectContracts) -> bool:
    """Plain-Python mirror of `_contract_term` for a matched (input ai, output bj)."""
    if name == "shape_exact":
        return bj.shape_hash != ai.shape_hash
    if name == "shape_canon":
        return bj.shape_canon != ai.shape_canon
    if name == "color_preserved":
        return bj.color != ai.color
    if name == "color_map":
        return not (0 <= ai.color < 10) or c.pi[ai.color] != bj.color
    if name == "size_preserved":
        return bj.size != ai.size
    if name == "size_scale":
        return bj.size != c.ksz * ai.size
    if name == "pos_preserved":
        return (bj.top, bj.left) != (ai.top, ai.left)
    if name == "pos_shift":
        return (bj.top, bj.left) != (ai.top + c.dr, ai.left + c.dc)
    if name == "bbox_preserved":
        return (bj.bh, bj.bw) != (ai.bh, ai.bw)
    return False


def _expected_actual(name: str, ai: Object, bj: Object, c: ObjectContracts) -> tuple[str, str]:
    if name == "color_preserved":
        return f"color {ai.color}", f"color {bj.color}"
    if name == "color_map":
        e = c.pi[ai.color] if 0 <= ai.color < 10 else ai.color
        return f"color {e} (rule maps {ai.color}→{e})", f"color {bj.color}"
    if name == "size_preserved":
        return f"{ai.size} cells", f"{bj.size} cells"
    if name == "size_scale":
        return f"{c.ksz * ai.size} cells ({c.ksz}× the input's {ai.size})", f"{bj.size} cells"
    if name == "pos_preserved":
        return f"top-left at ({ai.top},{ai.left})", f"top-left at ({bj.top},{bj.left})"
    if name == "pos_shift":
        return (f"top-left at ({ai.top + c.dr},{ai.left + c.dc}) — input ({ai.top},{ai.left}) "
                f"shifted by ({c.dr},{c.dc})", f"top-left at ({bj.top},{bj.left})")
    if name == "bbox_preserved":
        return f"a {ai.bh}×{ai.bw} bounding box", f"a {bj.bh}×{bj.bw} one"
    if name == "shape_canon":
        return "the input's shape up to rotation/reflection", "a different shape"
    return f"the input {ai.shape}'s exact shape", "a different cell pattern"   # shape_exact


def _match_cost(ai: Object, bj: Object, c: ObjectContracts) -> float:
    """Distance from bj to where the induced rule would put ai (so the match tracks the
    rule's intent). Lower = more plausibly the same object post-transform."""
    dr, dc = (c.dr, c.dc) if "pos_shift" in c.used else (0, 0)
    k = c.ksz if "size_scale" in c.used else 1
    pos = abs(ai.top + dr - bj.top) + abs(ai.left + dc - bj.left)
    shape = 0.0 if ai.shape_canon == bj.shape_canon else 3.0
    size = abs(bj.size - k * ai.size) / max(1, ai.size)
    return pos + shape + 2.0 * size


# above this best-match cost the pair is too dissimilar to assert "should be" wording
_MATCH_COST_MAX = 12.0


def diagnose_output(in_objs, out_objs, contracts, *, example: int = 1,
                    names=None, max_terms: int = 3) -> list[TermViolation]:
    """Structured witness for a candidate output already known to be inconsistent
    (count mismatch OR check_output False). Hungarian-matches produced→expected objects
    on the rule-anchored cost, then reports the violated `names` terms + unmatched
    objects. Returns [] only if it can't decode (defensive)."""
    names = list(contracts.used if names is None else names)
    viols: list[TermViolation] = []
    n, m = len(in_objs), len(out_objs)
    matched_i: set[int] = set()
    matched_j: set[int] = set()
    if n and m:
        cost = np.array([[_match_cost(ai, bj, contracts) for bj in out_objs] for ai in in_objs])
        ri, cj = linear_sum_assignment(cost)
        for i, j in zip(ri, cj):
            matched_i.add(int(i)); matched_j.add(int(j))
            ai, bj = in_objs[i], out_objs[j]
            if cost[i, j] > _MATCH_COST_MAX:
                continue                                  # too dissimilar to name a correspondence
            for name in names:
                if _term_violated(name, ai, bj, contracts):
                    exp, act = _expected_actual(name, ai, bj, contracts)
                    viols.append(TermViolation(example, "term", name, exp, act,
                                               (ai.top, ai.left), (bj.top, bj.left)))
    for j in range(m):
        if j not in matched_j:
            bj = out_objs[j]
            viols.append(TermViolation(example, "extra", "object_count",
                         "no object here (the rule keeps the input's objects)",
                         f"an extra {bj.shape} at ({bj.top},{bj.left})",
                         None, (bj.top, bj.left)))
    for i in range(n):
        if i not in matched_i:
            ai = in_objs[i]
            viols.append(TermViolation(example, "missing", "object_count",
                         f"a counterpart for the input {ai.shape} at ({ai.top},{ai.left})",
                         "no matching object in your output", (ai.top, ai.left), None))
    # rank: structural correspondence (extra/missing) first, then terms; cap the noise
    order = {"missing": 0, "extra": 0, "term": 1, "cell": 1}
    viols.sort(key=lambda v: order.get(v.kind, 2))
    return viols[:max_terms]
