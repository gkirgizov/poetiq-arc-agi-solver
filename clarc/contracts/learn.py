"""Open-vocabulary contract induction.

When a candidate fails the exact check yet violates NO known invariant (a
*semantic* conflict — "something is missing"), we ask the generator to propose a
new invariant as a pure predicate `holds(inp, out) -> bool`. Python then applies
the VERIFICATION GATE:

  1. soundness  : holds(inp, out) is True for EVERY train pair (it's a real invariant);
  2. relevance  : holds(inp, failed_out) is False for >=1 failed pair (it explains
                  the gap — motivated observation, not a random true fact).

Only predicates passing both are admitted. Execution is sandboxed
(predicate_sandbox), so no model code runs in-process. This is where the contract
vocabulary genuinely GROWS, adaptively, from real failures.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from arc_agi.solve_coding import _example_to_diagram

from clarc.contracts.sandbox import Pair, verify_predicate
from clarc.common.types import GenOutput, Generator, LearnedContract

_PROPOSAL_PROMPT = """You are analyzing an ARC grid task to discover an INVARIANT of the transformation.

Below are correct input -> output training pairs, then an INCORRECT output a program produced (it has the right gross structure but is wrong). Your job: find a property that ALL correct outputs satisfy relative to their input, but which the incorrect output VIOLATES.

Express it as a pure Python predicate:

```python
import numpy as np
DESCRIPTION = "<one short sentence describing the invariant>"
def holds(inp: np.ndarray, out: np.ndarray) -> bool:
    # True iff `out` is a valid output for input `inp` w.r.t. this invariant.
    ...
```

Rules: `holds` must be PURE and TOTAL (never raise; return False on anything odd), use only numpy + stdlib, and be a genuine invariant of the CORRECT pairs (not a description of the specific wrong output). Prefer simple, general properties (counts, relations between input and output, structure) over memorizing values. Return ONLY the code block.

CORRECT TRAINING PAIRS:
{pairs}

AN INCORRECT OUTPUT (for the first training input) that your invariant should rule out:
{failed}
"""


def build_proposal_prompt(train_pairs: list[Pair], failed_out: np.ndarray) -> str:
    blocks = []
    for i, (gi, go) in enumerate(train_pairs, 1):
        blocks.append(f"Pair {i} input:\n{_example_to_diagram(gi.tolist())}\n"
                      f"Pair {i} output:\n{_example_to_diagram(go.tolist())}")
    failed = _example_to_diagram(failed_out.tolist())
    return _PROPOSAL_PROMPT.format(pairs="\n\n".join(blocks), failed=failed)


def parse_proposal(code: str) -> tuple[str, Optional[str]]:
    """Extract (description, code) from a proposal code block."""
    if not code or "def holds" not in code:
        return "", None
    m = re.search(r'DESCRIPTION\s*=\s*["\'](.+?)["\']', code)
    descr = m.group(1) if m else "induced invariant"
    return descr, code


async def propose_contract(
    generator: Generator,
    train_pairs: list[Pair],
    failed_outputs: list[Optional[np.ndarray]],
    *,
    idx: int,
    seed: int,
    timeout_s: float = 5.0,
    report: Optional[dict] = None,
) -> Optional[LearnedContract]:
    """Ask, then GATE. Returns an admitted LearnedContract or None.

    `report`, when given, is filled with the proposal's LLM cost, the proposed
    descr, and where it died ("no-parse" | "gate1-soundness" | "gate2-relevance"
    | "admitted") — the raw material for tuning the induction loop offline.
    """
    if report is None:
        report = {}
    failed = next((f for f in failed_outputs if f is not None), None)
    if failed is None:
        report.update(stage="no-failed-output", cost_usd=0.0, descr=None)
        return None
    g: GenOutput = await generator.generate(
        build_proposal_prompt(train_pairs, failed), seed=seed
    )
    descr, code = parse_proposal(g.code or "")
    report.update(stage="no-parse", cost_usd=g.cost_usd, descr=descr or None)
    if code is None:
        return None

    # Gate 1: soundness — holds on every train pair.
    res_train = await verify_predicate(code, train_pairs, timeout_s=timeout_s)
    if res_train is None or not all(r is True for r in res_train):
        report["stage"] = "gate1-soundness"
        return None

    # Gate 2: relevance — violated by >=1 failed (input, produced-output).
    fail_pairs = [(gi, fo) for (gi, _), fo in zip(train_pairs, failed_outputs) if fo is not None]
    res_fail = await verify_predicate(code, fail_pairs, timeout_s=timeout_s)
    if res_fail is None or not any(r is False for r in res_fail):
        report["stage"] = "gate2-relevance"
        return None

    report["stage"] = "admitted"
    return LearnedContract(name=f"induced_{idx}", descr=descr, code=code, origin="induced")


async def verify_on_pairs(code: str, train_pairs: list[Pair], *, timeout_s: float = 5.0) -> bool:
    """Sound-reuse gate for a library candidate: does it hold on THIS task?"""
    res = await verify_predicate(code, train_pairs, timeout_s=timeout_s)
    return res is not None and len(res) > 0 and all(r is True for r in res)


async def induced_violations(
    learned: list[LearnedContract],
    train_in: list,
    produced: list[Optional[np.ndarray]],
    *,
    timeout_s: float = 5.0,
) -> list[LearnedContract]:
    """Which induced contracts a candidate's outputs violate (sandboxed, batched).

    Evaluated only on inputs the candidate actually produced a grid for; a contract
    is violated if its predicate is not True on every such (input, produced-output)
    pair. Since each induced predicate holds on all TRAIN pairs (the gate), a
    violation proves the candidate is train-wrong — so it is sound for pruning.
    """
    pairs = [(np.asarray(gi, dtype=int), po)
             for gi, po in zip(train_in, produced) if po is not None]
    if not pairs or not learned:
        return []
    bad: list[LearnedContract] = []
    for lc in learned:
        res = await verify_predicate(lc.code, pairs, timeout_s=timeout_s)
        if res is None or not all(r is True for r in res):
            bad.append(lc)
    return bad
