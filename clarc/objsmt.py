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
