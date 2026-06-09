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
        return sum(r.cost_usd for r in self.records)

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
            "total_cost_usd": round(self.total_cost(), 4),
            "records": [asdict(r) for r in self.records],
        }
