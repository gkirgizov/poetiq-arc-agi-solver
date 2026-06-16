"""Per-iteration instrumentation for the head-to-head and qualitative inspection.

Kept deliberately small: the loop appends an IterRecord per iteration; RunLog
summarises the thesis-critical signals (clause yield, structural:semantic ratio,
prune/harmful counts, iterations-to-solve, cost).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class IterRecord:
    iteration: int
    stage: str                       # parse_fail | exec_error | structural | semantic | solved
    soft_score: float = 0.0
    violated: list[str] = field(default_factory=list)
    conflict_type: Optional[str] = None   # structural | semantic | None (solved/parse_fail)
    pruned: bool = False
    harmful_prune: bool = False
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None      # generator error (timeout / no-code-block / cli-error)
    # --- predicate-loop history (for offline analysis of the contract machinery) ---
    active: list[str] = field(default_factory=list)  # contract names injected this iter
    proposed: Optional[str] = None   # induction outcome: proposal descr, or rejection
                                     # stage ("no-parse" | "gate1-soundness" | "gate2-relevance")
    prop_admitted: Optional[bool] = None  # None = no induction attempted this iter
    prop_cost_usd: float = 0.0       # LLM cost of the proposal call (separate from solver gen)
    # --- DSL ⇄ SMT dual (D-arms; stages "dsl_invalid" | "dup" | "refuted") ---
    dsl_text: Optional[str] = None   # normalized pipeline text (or raw snippet on parse fail)
    executed: bool = False           # sandbox actually ran this candidate
    dup_hit: bool = False            # exact duplicate of an earlier candidate
    refuted: bool = False            # pre-execution refutation (clause or z3)
    refutation_core: list[str] = field(default_factory=list)
    clause_fired: list[str] = field(default_factory=list)   # stored clauses that matched
    clause_learned: Optional[str] = None                    # NL of a newly learned clause
    class_pruned: int = 0            # skeletons blocked by the learned clause (depth-bounded)
    solver_ms: float = 0.0
    abs_weak: list[str] = field(default_factory=list)  # σ components the abstraction missed
    # --- guided code-gen dual (G-arms) ---
    ce: Optional[dict] = None        # structured counterexample payload {invariant, viols:[...]}
                                     # fed back this iter (None if no dual refuted the candidate)
    produced: Optional[list] = None  # candidate's per-train-example output grids (list|None each),
                                     # so CE actionability/repair can be recomputed offline


@dataclass
class RunLog:
    task_id: Optional[str]
    arm: str = "A0"
    seed: int = 0
    records: list[IterRecord] = field(default_factory=list)
    solved: bool = False
    iterations_to_solve: Optional[int] = None
    learned: list[str] = field(default_factory=list)   # descriptions of induced/reused contracts

    def add(self, rec: IterRecord) -> None:
        self.records.append(rec)

    # ---- thesis-critical rollups ----
    def total_cost(self) -> float:
        # Includes induction-proposal calls: A1L's true cost, not just solver gens.
        return sum(r.cost_usd + r.prop_cost_usd for r in self.records)

    def n_conflicts(self) -> int:
        return sum(1 for r in self.records if r.conflict_type in ("structural", "semantic"))

    def clause_yield(self) -> float:
        """Fraction of conflicts that were structural (a verified clause bit)."""
        n = self.n_conflicts()
        if not n:
            return 0.0
        return sum(1 for r in self.records if r.conflict_type == "structural") / n

    def n_pruned(self) -> int:
        return sum(1 for r in self.records if r.pruned)

    def n_harmful(self) -> int:
        return sum(1 for r in self.records if r.harmful_prune)

    def n_gen_failures(self) -> int:
        return sum(1 for r in self.records
                   if r.stage in ("gen_timeout", "gen_error", "parse_fail"))

    # ---- DSL ⇄ SMT rollups (zero for non-D arms) ----
    def n_refuted(self) -> int:
        return sum(1 for r in self.records if r.refuted)

    def n_executed(self) -> int:
        return sum(1 for r in self.records if r.executed)

    def n_dsl_invalid(self) -> int:
        return sum(1 for r in self.records if r.stage == "dsl_invalid")

    def n_dup(self) -> int:
        return sum(1 for r in self.records if r.dup_hit)

    def clause_reuse(self) -> int:
        """Refutations served by an ALREADY-stored clause (no fresh z3 call)."""
        return sum(1 for r in self.records if r.clause_fired)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "arm": self.arm,
            "seed": self.seed,
            "solved": self.solved,
            "iterations_to_solve": self.iterations_to_solve,
            "n_learned": len(self.learned),
            "learned": self.learned,
            "n_iters": len(self.records),
            "n_conflicts": self.n_conflicts(),
            "clause_yield": round(self.clause_yield(), 3),
            "n_pruned": self.n_pruned(),
            "n_harmful": self.n_harmful(),
            "n_gen_failures": self.n_gen_failures(),
            "n_refuted": self.n_refuted(),
            "n_executed": self.n_executed(),
            "n_dsl_invalid": self.n_dsl_invalid(),
            "n_dup": self.n_dup(),
            "clause_reuse": self.clause_reuse(),
            "solver_ms_total": round(sum(r.solver_ms for r in self.records), 1),
            "total_cost_usd": round(self.total_cost(), 4),
            "records": [asdict(r) for r in self.records],
        }
