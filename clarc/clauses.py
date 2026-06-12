"""Conflict condensation: refutations -> logical clauses over the SOUGHT
transformation, via a generalization ladder where EVERY rung is verified by a
solver re-query (never syntactic generalization):

    rung 1  exact      this pipeline (with these params) cannot fit the pairs
    rung 2  at_pos     `prim` at step k cannot fit, with ANY params
    rung 3  anywhere   no <=depth pipeline using `prim` AT ALL can fit

Epistemic note (deliberate refinement of the LOO-gate idea in spec.loo_trusted):
spec contracts are INDUCED hypotheses about generalization, so they need a
leave-one-out gate before they may hard-prune. Clauses here are DEDUCED — each
carries an unsat-core certificate over the concrete σ facts of the train pairs,
so it is valid for this task by construction and always safe to enforce. The
LOO re-query is still run, but as a ROBUSTNESS annotation (`loo_robust`: does
the refutation survive dropping any single pair?) used for diagnostics and
prompt phrasing, not as a trust gate.

Clauses fire syntactically against future candidates (no z3 needed to re-block)
and are rendered into the prompt as machine-checked impossibilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from clarc.dsl import REGISTRY, Pipeline
from clarc.smt import TaskSMT

_GROUP_NL = {"dims": "dimensions", "hist": "color histogram",
             "objects": "object count", "bbox": "content bounding box",
             "sym": "symmetries"}


def _facts_nl(core: tuple[str, ...]) -> str:
    groups = sorted({_GROUP_NL.get(c.split(":")[2], c.split(":")[2])
                     for c in core if c.startswith("fact:")})
    return ", ".join(groups) if groups else "the training observations"


@dataclass(frozen=True)
class Clause:
    kind: str                       # "exact" | "at_pos" | "anywhere" | "concrete"
    prim: str                       # primitive name (or pipeline pretty for exact/concrete)
    pos: int | None
    core: tuple[str, ...]
    nl: str
    loo_robust: bool = False
    depth: int | None = None        # for "anywhere": the verified depth bound

    def matches(self, p: Pipeline) -> bool:
        if self.kind in ("exact", "concrete"):
            return p.pretty() == self.prim
        if self.kind == "at_pos":
            return self.pos is not None and self.pos < len(p.steps) \
                and p.steps[self.pos].name == self.prim
        if self.kind == "anywhere":
            return any(s.name == self.prim for s in p.steps)
        return False

    def n_blocked(self, depth: int) -> int:
        """How many <=depth skeletons (primitive-name sequences) this clause
        blocks — the class-prune factor vs try-and-drop's single point."""
        n = len(REGISTRY)
        if self.kind in ("exact", "concrete"):
            return 1
        if self.kind == "at_pos":
            return n ** (depth - 1)
        if self.kind == "anywhere":
            return n ** depth - (n - 1) ** depth
        return 0


@dataclass
class ClauseStore:
    """Per-task clause DB (the CDCL learned-clause store, over transformations)."""

    smt: TaskSMT
    depth: int = 4
    clauses: list[Clause] = field(default_factory=list)
    # LOO robustness is an annotation (see module docstring), not a gate; the
    # extra per-pair re-queries cost ~0.2-2s each on real tasks, so live runs
    # disable it and compute it offline when needed.
    loo_annotate: bool = True

    def match(self, p: Pipeline) -> list[Clause]:
        return [c for c in self.clauses if c.matches(p)]

    def _add(self, c: Clause) -> Clause:
        if not any(x.kind == c.kind and x.prim == c.prim and x.pos == c.pos
                   for x in self.clauses):
            self.clauses.append(c)
        return c

    # ---------------------------------------------------------------- ladder
    def learn_from_refutation(self, p: Pipeline, exact_core: tuple[str, ...]) -> Clause:
        """Called when check_pipeline(p) was UNSAT. Climb the ladder, verifying
        each rung with a re-query; store and return the most general clause."""
        r_class = self.smt.check_pipeline(p, with_params=False)
        if not r_class.refuted:
            nl = (f"`{p.pretty()}` is impossible: with these exact parameters it "
                  f"contradicts the training pairs' {_facts_nl(exact_core)}")
            return self._add(Clause("exact", p.pretty(), None, exact_core, nl,
                                    loo_robust=self._loo_exact(p)))

        # Class-level refutation. Try position-lift for each slot in the core.
        core = r_class.core
        slots = [(int(c.split(":")[1]), c.split(":")[2])
                 for c in core if c.startswith("slot:")]
        for pos, prim in slots:
            if self.smt.prim_impossible_anywhere(prim, self.depth):
                robust = (self.loo_annotate and len(self.smt.facts) >= 2 and all(
                    self.smt.prim_impossible_anywhere(prim, self.depth, drop_pair=i)
                    for i in range(len(self.smt.facts))))
                nl = (f"NO pipeline (up to {self.depth} steps) using `{prim}` "
                      f"anywhere can fit the training pairs — it contradicts their "
                      f"{_facts_nl(core)}")
                return self._add(Clause("anywhere", prim, None, core, nl,
                                        loo_robust=robust, depth=self.depth))
        pos, prim = slots[0] if slots else (0, p.steps[0].name)
        robust = (self.loo_annotate and len(self.smt.facts) >= 2 and all(
            self.smt.check_pipeline(p, with_params=False, drop_pair=i).refuted
            for i in range(len(self.smt.facts))))
        nl = (f"any pipeline with `{prim}` at step {pos + 1} of this shape is "
              f"impossible with ANY parameters — it contradicts the training "
              f"pairs' {_facts_nl(core)}")
        return self._add(Clause("at_pos", prim, pos, core, nl, loo_robust=robust))

    def _loo_exact(self, p: Pipeline) -> bool:
        if not self.loo_annotate or len(self.smt.facts) < 2:
            return False
        return all(self.smt.check_pipeline(p, drop_pair=i).refuted
                   for i in range(len(self.smt.facts)))

    def add_concrete_block(self, p: Pipeline) -> Clause:
        """A candidate that PASSED the abstract check but failed concrete
        execution: block the exact instance (concretely disproved)."""
        nl = f"`{p.pretty()}` was executed and does NOT reproduce the training outputs"
        return self._add(Clause("concrete", p.pretty(), None, (), nl, loo_robust=True))

    # ---------------------------------------------------------------- render
    def render_for_prompt(self, max_clauses: int = 16) -> str:
        if not self.clauses:
            return ""
        # Most general first; concrete one-offs last.
        order = {"anywhere": 0, "at_pos": 1, "exact": 2, "concrete": 3}
        ranked = sorted(self.clauses, key=lambda c: order[c.kind])[:max_clauses]
        lines = ["PROVABLY REFUTED APPROACHES (machine-checked against ALL training "
                 "examples — do not emit these):"]
        lines += [f"  - {c.nl}" for c in ranked]
        return "\n".join(lines)
