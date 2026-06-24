"""z3 queries over the dual: CHECK (refute a candidate pre-execution) and SYNTH
(bounded achievability: which pipelines COULD fit all train pairs?).

Encoding: per train pair a chain of abstract states σ_0..σ_K (WF axioms hard);
slot/param variables are SHARED across pairs — a candidate class must fit ALL
pairs with ONE parameterization, which is where cross-pair refutations (e.g.
non-uniform dims ratios) come from.

Assumption labels (for unsat cores):
    slot:{k}:{prim}            this step participates
    param:{k}:{name}={value}   this emitted param binding
    fact:{i}:{group}:{in|out}  this observed fact group of pair i

`check_pipeline(..., with_params=False)` frees the params: UNSAT then refutes the
whole skeleton CLASS (every parameterization), not just the emitted candidate.
Timeout/unknown degrade to "not refuted" — never to wrongness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import z3

from clarc.dsl.absdomain import Sigma, ZState
from clarc.dsl.core import REGISTRY, Pipeline, Step, alloc_params, param_domain, param_index


@dataclass
class CheckResult:
    status: str                       # "sat" | "unsat" | "unknown"
    core: tuple[str, ...] = ()
    ms: float = 0.0

    @property
    def refuted(self) -> bool:
        return self.status == "unsat"


@dataclass
class TaskSMT:
    """Per-task solver context: the concrete σ facts of all train pairs.

    `registry` defaults to the global primitive registry; pass a task-local one
    (global + induced prims) so induction stays task-scoped and concurrency-safe."""

    facts: list[tuple[Sigma, Sigma]]
    timeout_ms: int = 2000
    registry: dict = None  # type: ignore[assignment]
    last_ms: float = field(default=0.0, init=False)

    def __post_init__(self):
        if self.registry is None:
            self.registry = REGISTRY

    # ------------------------------------------------------------------ CHECK
    def check_pipeline(
        self,
        pipeline: Pipeline,
        *,
        with_params: bool = True,
        drop_pair: int | None = None,
    ) -> CheckResult:
        t0 = time.monotonic()
        s = z3.Solver()
        s.set("timeout", self.timeout_ms)
        s.set(":core.minimize", True)

        K = len(pipeline.steps)
        pairs = [(i, f) for i, f in enumerate(self.facts) if i != drop_pair]

        # Shared slot/param consts.
        zparams = []
        assumptions: list[z3.BoolRef] = []
        slot_lits = []
        for k, step in enumerate(pipeline.steps):
            prim = self.registry[step.name]
            P = alloc_params(prim, f"k{k}")
            s.add(*param_domain(prim, P))
            zparams.append(P)
            lit = z3.Bool(f"slot:{k}:{step.name}")
            slot_lits.append(lit)
            assumptions.append(lit)
            if with_params:
                for name, val in step.params.items():
                    plit = z3.Bool(f"param:{k}:{name}={val}")
                    s.add(z3.Implies(plit, P[name] == param_index(prim, name, val)))
                    assumptions.append(plit)

        for i, (sin, sout) in pairs:
            states = [ZState(f"p{i}s{k}") for k in range(K + 1)]
            for st in states:
                s.add(*st.wf())
            for k, step in enumerate(pipeline.steps):
                prim = self.registry[step.name]
                rel = z3.And(*prim.encode(states[k], states[k + 1], zparams[k]))
                s.add(z3.Implies(slot_lits[k], rel))
            for end, st, sig in (("in", states[0], sin), ("out", states[K], sout)):
                for grp, eq in st.eq_concrete(sig).items():
                    flit = z3.Bool(f"fact:{i}:{grp}:{end}")
                    s.add(z3.Implies(flit, eq))
                    assumptions.append(flit)

        res = s.check(*assumptions)
        ms = (time.monotonic() - t0) * 1000
        self.last_ms = ms
        if res == z3.unsat:
            core = tuple(sorted(str(b) for b in s.unsat_core()))
            return CheckResult("unsat", core, ms)
        return CheckResult("sat" if res == z3.sat else "unknown", (), ms)

    # ------------------------------------------------------------------ SYNTH
    def _build_synth(
        self,
        depth: int,
        *,
        blocked_anywhere: set[str] = frozenset(),
        blocked_at: dict[int, set[str]] | None = None,
        drop_pair: int | None = None,
    ) -> tuple[z3.Solver, list[dict[str, z3.BoolRef]], list[dict]]:
        s = z3.Solver()
        s.set("timeout", max(self.timeout_ms, 4000))
        names = sorted(self.registry)
        use: list[dict[str, z3.BoolRef]] = []
        zparams: list[dict] = []
        for k in range(depth):
            u = {n: z3.Bool(f"use_{k}_{n}") for n in names}
            use.append(u)
            s.add(z3.PbEq([(u[n], 1) for n in names], 1))  # exactly one per slot
            ps = {n: alloc_params(self.registry[n], f"sk{k}_{n}") for n in names}
            for n in names:
                s.add(*param_domain(self.registry[n], ps[n]))
            zparams.append(ps)
        # identity-suffix symmetry breaking: once identity, always identity.
        for k in range(depth - 1):
            s.add(z3.Implies(use[k]["identity"], use[k + 1]["identity"]))
        for n in blocked_anywhere:
            for k in range(depth):
                s.add(z3.Not(use[k][n]))
        for k, ns in (blocked_at or {}).items():
            if k < depth:
                for n in ns:
                    s.add(z3.Not(use[k][n]))

        for i, (sin, sout) in enumerate(self.facts):
            if i == drop_pair:
                continue
            states = [ZState(f"q{i}s{k}") for k in range(depth + 1)]
            for st in states:
                s.add(*st.wf())
            for k in range(depth):
                for n in names:
                    rel = z3.And(*self.registry[n].encode(states[k], states[k + 1],
                                                     zparams[k][n]))
                    s.add(z3.Implies(use[k][n], rel))
            for st, sig in ((states[0], sin), (states[depth], sout)):
                for eq in st.eq_concrete(sig).values():
                    s.add(eq)
        return s, use, zparams

    def synth_feasible(self, depth: int, **kw) -> bool | None:
        """Is ANY pipeline of length <= depth abstractly consistent with all
        pairs? True/False, or None on timeout."""
        t0 = time.monotonic()
        s, _, _ = self._build_synth(depth, **kw)
        res = s.check()
        self.last_ms = (time.monotonic() - t0) * 1000
        return None if res == z3.unknown else res == z3.sat

    def synth_models(self, depth: int, max_models: int = 6, **kw) -> list[Pipeline]:
        """Enumerate up to max_models feasible skeletons (with one witness
        parameterization each), SHORTEST FIRST (Occam ordering), blocking each
        skeleton after extraction."""
        out: list[Pipeline] = []
        seen: set[tuple[str, ...]] = set()
        names = sorted(self.registry)
        for d in range(1, depth + 1):
            if len(out) >= max_models:
                break
            s, use, zparams = self._build_synth(d, **kw)
            # skip skeletons whose identity-stripped form was already reported
            while len(out) < max_models and s.check() == z3.sat:
                m = s.model()
                steps, blockers = [], []
                for k in range(d):
                    n = next(nm for nm in names if z3.is_true(m[use[k][nm]]))
                    blockers.append(use[k][n])
                    if n == "identity":
                        continue
                    prim = self.registry[n]
                    params = {}
                    for p in prim.params:
                        v = m[zparams[k][n][p.name]]
                        idx = v.as_long() if v is not None else 0
                        params[p.name] = p.values[idx]
                    steps.append(Step(n, params))
                skel = tuple(s_.name for s_ in steps)
                if steps and skel not in seen:
                    seen.add(skel)
                    out.append(Pipeline(tuple(steps)))
                s.add(z3.Not(z3.And(*blockers)))  # block this skeleton, params free
        return out

    def prim_impossible_anywhere(self, prim_name: str, depth: int,
                                 drop_pair: int | None = None) -> bool:
        """Verified position-lift: is EVERY <=depth pipeline that uses prim_name
        somewhere abstractly inconsistent with the (remaining) pairs?"""
        s, use, _ = self._build_synth(depth, drop_pair=drop_pair)
        s.add(z3.Or(*[use[k][prim_name] for k in range(depth)]))
        return s.check() == z3.unsat
