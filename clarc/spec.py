"""Spec extraction — the target refinement type Grid[C_I] -> Grid[C_O].

`extract_spec` evaluates the contract vocabulary across ALL train pairs and keeps
(after subsumption) the invariants that hold everywhere. By construction every
contract in the returned Spec re-verifies on the training data — that soundness
is asserted in the tests.

This is the "motivated observation" step in its minimal form: a fixed, cheap
attribute basis (grids are tiny, so eager evaluation is fine for the core).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from clarc.contracts import BUILDERS, Contract, Pair, subsume

GridList = list[list[int]]


@dataclass
class Spec:
    task_id: str | None
    contracts: list[Contract]
    pairs: list[Pair] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.contracts

    def render_for_prompt(self) -> str:
        if not self.contracts:
            return ""
        lines = [
            "VERIFIED INVARIANTS (these hold on ALL training examples; a correct "
            "`transform` MUST satisfy them):"
        ]
        for c in self.contracts:
            lines.append(f"  - {c.descr}")
        return "\n".join(lines)

    def violations(
        self, train_in: list[GridList], produced_out: list[np.ndarray | None]
    ) -> list[Contract]:
        """Contracts violated by a candidate's produced outputs on the train inputs.

        `produced_out[i]` is the candidate's output for `train_in[i]` (or None if it
        failed to run / parse). A None output counts as violating every contract.
        Used by Phase 2 for cheap, structured, pre-exact-check feedback/pruning.
        """
        bad: list[Contract] = []
        for c in self.contracts:
            ok = True
            for gi_list, go in zip(train_in, produced_out):
                if go is None:
                    ok = False
                    break
                gi = np.asarray(gi_list, dtype=int)
                try:
                    if not c.check(gi, go):
                        ok = False
                        break
                except (ValueError, IndexError, KeyError, TypeError, ZeroDivisionError, AttributeError):
                    ok = False
                    break
            if not ok:
                bad.append(c)
        return bad


def loo_trusted(
    train_in: list[GridList], train_out: list[GridList], task_id: str | None = None
) -> set[str]:
    """Contract names robust to leave-one-out — safe for HARD pruning.

    A contract is trusted iff, for every held-out pair, it is (a) still discovered
    from the remaining pairs AND (b) correctly predicts the held-out pair. This
    filters coincidences on tiny train sets (the Risk-5 mitigation). With <2 pairs
    nothing can be trusted for pruning.
    """
    full = extract_spec(train_in, train_out, task_id).contracts
    if len(train_in) < 2 or not full:
        return set()
    surviving = {c.name for c in full}
    for i in range(len(train_in)):
        sub_in = train_in[:i] + train_in[i + 1:]
        sub_out = train_out[:i] + train_out[i + 1:]
        sub_names = {c.name for c in extract_spec(sub_in, sub_out, task_id).contracts}
        held_in = np.asarray(train_in[i], dtype=int)
        held_out = np.asarray(train_out[i], dtype=int)
        ok = set()
        for c in full:
            try:
                if c.name in sub_names and c.check(held_in, held_out):
                    ok.add(c.name)
            except (ValueError, IndexError, KeyError, TypeError, ZeroDivisionError, AttributeError):
                pass
        surviving &= ok
    return surviving


def extract_spec(
    train_in: list[GridList], train_out: list[GridList], task_id: str | None = None
) -> Spec:
    pairs: list[Pair] = [
        (np.asarray(gi, dtype=int), np.asarray(go, dtype=int))
        for gi, go in zip(train_in, train_out)
    ]
    if not pairs:
        return Spec(task_id=task_id, contracts=[], pairs=[])
    found: list[Contract] = []
    for builder in BUILDERS:
        try:
            c = builder(pairs)
        except (ValueError, IndexError, KeyError, TypeError, ZeroDivisionError, AttributeError):
            c = None
        if c is not None:
            found.append(c)
    return Spec(task_id=task_id, contracts=subsume(found), pairs=pairs)
