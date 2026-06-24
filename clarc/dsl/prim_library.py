"""Cross-task library of INDUCED primitives — the "between tasks" half of the
self-extending DSL (twin of clarc/library.py, which stores induced predicates).

Each entry persists the LLM's transform source + the auto-derived template
contract + utility stats. On reuse at a new task, the contract is RE-VERIFIED by
re-deriving from fresh σ-samples of the stored code (sound reuse — never trust a
stored contract blindly), mirroring the predicate library's re-verification.

Stored as JSON so it accumulates across runs.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from clarc.dsl.induce import InducedPrimitive, TemplateContract
from clarc.common.paths import PKG_ROOT

_DEFAULT_PATH = os.path.join(PKG_ROOT, "prim_library.json")


@dataclass
class PrimEntry:
    name: str
    descr: str
    code: str
    int_tmpl: dict          # serialized TemplateContract (templates as lists)
    bool_tmpl: dict
    proposed: int = 0       # times induced/loaded
    seen_tasks: int = 0     # tasks where it re-verified sound
    solved_after: int = 0   # tasks solved after it was active

    def utility(self) -> float:
        return (self.solved_after + 1) / (self.seen_tasks + 2)   # Laplace-smoothed

    def contract(self) -> TemplateContract:
        return TemplateContract(
            int_tmpl={k: tuple(v) for k, v in self.int_tmpl.items()},
            bool_tmpl={k: tuple(v) for k, v in self.bool_tmpl.items()})

    def induced(self) -> InducedPrimitive:
        return InducedPrimitive(self.name, self.descr, self.code, self.contract())


@dataclass
class PrimLibrary:
    path: str = _DEFAULT_PATH
    entries: list[PrimEntry] = field(default_factory=list)

    @classmethod
    def load(cls, path: str = _DEFAULT_PATH) -> "PrimLibrary":
        lib = cls(path=path)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
                lib.entries = [PrimEntry(**e) for e in raw.get("entries", [])]
            except (json.JSONDecodeError, TypeError, KeyError):
                lib.entries = []
        return lib

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"entries": [asdict(e) for e in self.entries]}, f, indent=2)
        except OSError:
            pass

    def _norm(self, code: str) -> str:
        return " ".join(code.split())

    def add(self, ind: InducedPrimitive) -> PrimEntry:
        n = self._norm(ind.code)
        for e in self.entries:
            if self._norm(e.code) == n:
                e.proposed += 1
                return e
        entry = PrimEntry(name=ind.name, descr=ind.descr, code=ind.code,
                          int_tmpl={k: list(v) for k, v in ind.contract.int_tmpl.items()},
                          bool_tmpl={k: list(v) for k, v in ind.contract.bool_tmpl.items()},
                          proposed=1)
        self.entries.append(entry)
        return entry

    def candidates(self) -> list[PrimEntry]:
        """Entries ranked by past utility (best prior first)."""
        return sorted(self.entries, key=lambda e: -e.utility())

    def record_use(self, code: str, *, verified: bool, solved: bool) -> None:
        n = self._norm(code)
        for e in self.entries:
            if self._norm(e.code) == n:
                if verified:
                    e.seen_tasks += 1
                if solved:
                    e.solved_after += 1
                return
