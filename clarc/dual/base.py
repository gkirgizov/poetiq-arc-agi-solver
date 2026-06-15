"""The logical-dual layer — orthogonal invariant sources that GUIDE the poetiq
code-gen generator (they never replace it).

Each `Dual` extracts a logical representation from the train examples and exposes
it three ways, matching the proven neurosymbolic refinement pattern (LLM proposes,
symbolic layer constrains + returns counterexamples):

  1. `prompt_block()`  — invariants to inject so the generator searches a smaller
                         space (invariants reduce program-space entropy).
  2. `refute(...)`     — a candidate's train outputs that violate the invariants
                         yield a COUNTEREXAMPLE, fed back to repair the next attempt.

A Dual that finds no usable structure returns "" / None and is a pure no-op — so
the solver with no (or silent) duals is EXACTLY poetiq A0, and any guidance can
only help. Strictly-≥-A0 by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

Pair = tuple[np.ndarray, np.ndarray]


@dataclass
class Counterexample:
    invariant: str          # which induced invariant the candidate violated
    detail: str             # specifics: which example/object/value (for repair)

    def render(self) -> str:
        return f"{self.invariant} — {self.detail}"


@runtime_checkable
class Dual(Protocol):
    name: str

    def extract(self, train_pairs: list[Pair]) -> None:
        """Induce the dual representation from the train (input, output) pairs."""
        ...

    def prompt_block(self) -> str:
        """Invariants to inject into the generator prompt ("" if none)."""
        ...

    def refute(self, cand_pairs: list[Pair]) -> Optional[Counterexample]:
        """`cand_pairs` = (train_input, CANDIDATE output). Return a counterexample
        if the candidate violates an induced invariant, else None."""
        ...
