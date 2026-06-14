"""Core clarc data structures.

Phase 0 keeps this intentionally small: the generator interface, the per-run
config (with ingredient flags for the A0-A5 ablations, all OFF by default), and
re-exports of the poetiq result TypedDicts so clarc results flow straight into
the existing ensemble voting / io / scoring code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

# Reuse poetiq's boundary types verbatim so solve_parallel_coding / io / scoring
# accept clarc output unchanged.
from arc_agi.types import ARCAGIResult, RunResult  # noqa: F401  (re-exported)

# A grid is a 2D int array at runtime; nested lists at the JSON boundary.
Grid = np.ndarray
GridList = list[list[int]]


@dataclass
class GenOutput:
    """Result of a single generator call."""

    code: Optional[str]            # parsed `transform` source, or None on failure
    raw: str                       # full assistant text
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None    # set on subprocess/timeout/JSON/CLI failure


@dataclass(frozen=True)
class LearnedContract:
    """An invariant DISCOVERED at runtime (not from the fixed vocabulary).

    `code` defines a pure predicate `holds(inp, out) -> bool`. It is admitted only
    after Python verifies it holds on every train pair (and is discriminative w.r.t.
    a failed candidate). Persisted to the cross-task library for sound reuse.
    """

    name: str
    descr: str
    code: str
    origin: str = "induced"     # "induced" (this task) | "library" (reused)


class Generator(Protocol):
    """Same ROLE as poetiq's llm(): turn a prompt into a candidate program.

    The loop owns iteration; a Generator is a stateless one-shot code producer.
    """

    async def generate(self, prompt: str, *, seed: int) -> GenOutput: ...


@dataclass
class ClarcConfig:
    """Per-run configuration. Ingredient flags select the ablation arm.

    A0 (baseline)  : all flags False  -> poetiq-style loop through the CLI.
    A5 (spec-only) : spec_inject=True, rest False.
    A1 (full)      : spec_inject, clause_learn, clause_inject all True (+ optional clause_prune).
    A4 (DSL tax)   : dsl_required=True.
    """

    # --- generation ---
    model: Optional[str] = None            # CLI model alias; None = CLI default
    temperature: float = 1.0
    max_iterations: int = 10
    request_timeout_s: float = 300.0
    seed: int = 0
    shuffle_examples: bool = True
    return_best_result: bool = True
    lean_prompt: bool = False          # use a terse single-attempt prompt (curbs CLI over-thinking)
    # poetiq-style feedback memory (used when contract machinery is off)
    max_solutions: int = 5
    selection_probability: float = 1.0
    improving_order: bool = True
    # --- sandbox ---
    timeout_sandbox_s: float = 5.0
    # --- contract ingredient flags (Phases 1-2; OFF for A0) ---
    spec_inject: bool = False
    clause_learn: bool = False
    clause_inject: bool = False
    clause_prune: bool = False
    dsl_required: bool = False
    # --- DSL ⇄ SMT dual (Phase 5; D-arms; needs dsl_required) ---
    z3_refute: bool = False        # pre-execution CHECK; refuted candidates never run
    z3_learn: bool = False         # conflicts -> generalized blocking clauses
    z3_inject: bool = True         # render learned clauses into the prompt
    dsl_depth_max: int = 4         # bound for SYNTH/position-lift queries
    z3_timeout_ms: int = 2000      # solver cap; timeout degrades to "not refuted"
    max_clauses: int = 16          # prompt block cap (solver always sees all)
    # --- self-extending DSL (M6; arm E0) ---
    induce_prims: bool = False     # induce new primitives when the library is insufficient
    induce_on_unsat: bool = True   # trigger induction when the SMT oracle says UNSAT@depth
    max_induced_prims: int = 3     # cap induced prims per task
    prim_use_library: bool = True  # load/persist the cross-task induced-primitive library
    # --- adaptive contract learning (Phase 4) ---
    learn_contracts: bool = False     # induce new invariants from semantic conflicts
    max_learned: int = 3              # cap induced contracts per task
    propose_every: int = 1            # attempt a proposal every N semantic-conflict iters
    use_library: bool = True          # load/persist the cross-task contract library
    # --- bookkeeping ---
    problem_id: Optional[str] = None
    log_dir: Optional[str] = None
    metadata: dict = field(default_factory=dict)
