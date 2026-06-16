"""σ-invariant dual — the verified grid-level invariants (dims, palette, counts,
symmetry) from clarc.spec. Coarser than the object dual, but cheap and orthogonal;
contributes additional constraints + counterexamples."""

from __future__ import annotations

from typing import Optional

import numpy as np

from clarc.dual.base import Counterexample, Dual, Pair
from clarc.spec import extract_spec


class SigmaDual:
    name = "sigma"

    def __init__(self) -> None:
        self.spec = None

    def extract(self, train_pairs: list[Pair]) -> None:
        ti = [np.asarray(gi) for gi, _ in train_pairs]
        to = [np.asarray(go) for _, go in train_pairs]
        try:
            self.spec = extract_spec(ti, to, None)
        except (ValueError, KeyError):
            self.spec = None

    def prompt_block(self) -> str:
        if self.spec is None or self.spec.is_empty():
            return ""
        return self.spec.render_for_prompt()

    def active_names(self) -> list[str]:
        """Invariant names actually injected into the prompt (for per-iteration logs)."""
        if self.spec is None or self.spec.is_empty():
            return []
        return [c.name for c in self.spec.contracts]

    def refute(self, cand_pairs: list[Pair]) -> Optional[Counterexample]:
        if self.spec is None or self.spec.is_empty():
            return None
        ti = [np.asarray(gi) for gi, _ in cand_pairs]
        produced = [None if co is None else np.asarray(co) for _, co in cand_pairs]
        violated = self.spec.violations(ti, produced)
        if violated:
            detail = "; ".join(v.descr for v in violated)   # all violated invariants, not just one
            return Counterexample("invariant", f"your output violates: {detail}")
        return None


assert isinstance(SigmaDual(), Dual)
