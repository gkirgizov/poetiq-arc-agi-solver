"""LearnedStore — the CDCL clause database.

In the minimal core the learned clauses ARE the verified spec contracts (each
provably holds on all train pairs, so admitting them can never introduce false
knowledge). The store adds the *dynamic* CDCL machinery on top:

- violation activity counters (VSIDS-lite): which contracts the generator keeps
  violating across iterations;
- earned escalation: a contract violated >= ESCALATE_AT times is rendered as
  MANDATORY and surfaced first (the "inject only after repeated conflict" Risk-2
  mitigation);
- `add()` enforces a `verified` gate so any future (e.g. LLM-proposed) clause can
  only enter once Python has checked it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from clarc.contracts import Contract
from clarc.types import LearnedContract

ESCALATE_AT = 2  # a contract violated this many times becomes MANDATORY in the prompt


@dataclass
class LearnedStore:
    contracts: list[Contract] = field(default_factory=list)   # verified positive contracts (fixed basis)
    learned: list[LearnedContract] = field(default_factory=list)  # induced/reused (open vocabulary)
    trusted: set[str] = field(default_factory=set)            # names safe for hard pruning (LOO)
    violations: Counter = field(default_factory=Counter)      # name -> times a candidate violated it
    conflicts: int = 0                                        # iterations with >=1 violation

    # ---- mutation ----
    def add(self, contract: Contract, *, verified: bool) -> bool:
        """Admit a clause ONLY if Python verified it. Returns whether it was added."""
        if not verified:
            return False
        if any(c.name == contract.name for c in self.contracts):
            return False
        self.contracts.append(contract)
        return True

    def add_learned(self, lc: LearnedContract) -> bool:
        """Admit an induced/reused contract (already verified upstream by the gate)."""
        norm = " ".join(lc.code.split())
        if any(" ".join(e.code.split()) == norm for e in self.learned):
            return False
        self.learned.append(lc)
        return True

    def note(self, violated: list[Contract]) -> None:
        if violated:
            self.conflicts += 1
        for c in violated:
            self.violations[c.name] += 1

    # ---- rendering (injection) ----
    def render_for_prompt(self) -> str:
        if not self.contracts and not self.learned:
            return ""
        lines: list[str] = []
        if self.contracts:
            ordered = sorted(self.contracts, key=lambda c: -self.violations[c.name])
            lines.append(
                "VERIFIED INVARIANTS (these provably hold on ALL training examples; a "
                "correct `transform` MUST satisfy every one):"
            )
            for c in ordered:
                v = self.violations[c.name]
                tag = f"   [⚠ violated {v}× — MANDATORY]" if v >= ESCALATE_AT else ""
                lines.append(f"  - {c.descr}{tag}")
        if self.learned:
            lines.append(
                "\nDISCOVERED INVARIANTS (learned from earlier failures, verified on all "
                "training pairs — your output MUST satisfy these too):"
            )
            for lc in self.learned:
                lines.append(f"  - {lc.descr}")
        return "\n".join(lines)
