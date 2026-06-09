"""Conflict analysis — the CDCL "learn from a conflict" step.

Minimal core: a failed candidate is checked against the verified spec contracts
(`spec.violations`). The outcome classifies the conflict:

- structural: the candidate's output violates >=1 verified invariant. We know it is
  wrong WITHOUT comparing to ground truth, and we can say exactly why — that
  precise reason is the feedback that "subsumes" poetiq's generic cell diff, and
  (when trusted) licenses pruning.
- semantic: the output satisfies every invariant but is still wrong (a content
  bug). Here the fixed vocabulary has nothing to bite on and we fall back to the
  poetiq diff. The structural:semantic ratio is the key instrument for whether the
  CDCL framing applies to a task.

The verification GATE (`admit_clause`) is the soundness linchpin: any proposed
clause — Python- or (future) LLM-proposed — enters the store only if its predicate
provably holds across all train pairs. Unverifiable "lessons" are rejected.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from clarc.contracts import Contract, Pair
from clarc.types import RunResult


def parse_grid(run_result: RunResult) -> Optional[np.ndarray]:
    """Parse a candidate's produced output into a 2D int grid, or None if invalid."""
    out = run_result.get("output")
    if not out:
        return None
    try:
        arr = np.array(json.loads(out))
    except Exception:
        return None
    if arr.ndim != 2 or arr.size == 0:
        return None
    try:
        return arr.astype(int, copy=False)
    except Exception:
        return None


def classify_conflict(violated: list[Contract]) -> str:
    return "structural" if violated else "semantic"


def render_violation_feedback(violated: list[Contract]) -> str:
    """Sharp, structured reason a candidate is wrong — distinct from a cell diff."""
    if not violated:
        return ""
    lines = [
        "Your output VIOLATED these verified invariants, so it cannot be correct. "
        "Fix these first:"
    ]
    for c in violated:
        lines.append(f"  - {c.descr}")
    return "\n".join(lines)


def admit_clause(candidate: Contract, pairs: list[Pair]) -> bool:
    """The verification gate: admit a clause iff it holds on EVERY train pair.

    Works for any source of `candidate` (deterministic miner or, as an extension,
    an LLM proposal parsed from a `--json-schema` response). The minimal core does
    not mine new clauses beyond the fixed vocabulary (extract_spec already runs the
    whole vocabulary), so this guards the extension path.
    """
    try:
        return candidate.holds_on(pairs)
    except Exception:
        return False


# --- Extension point (documented, not built in the minimal core) -------------
# def propose_clauses_via_llm(generator, spec, failed_code, pairs) -> list[Contract]:
#     """Ask the generator (claude -p --json-schema) to pick, from a fixed
#     predicate vocabulary, an invariant that explains a *semantic* conflict;
#     parse the JSON, build a concrete Contract, and gate it through admit_clause.
#     This is where new knowledge beyond the fixed builders would enter. Deferred:
#     it adds a second LLM call per conflict and a richer vocabulary, neither of
#     which is needed to demonstrate the verifiable-clause loop."""
