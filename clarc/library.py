"""Cross-task contract library — the "between tasks" learning.

Admitted induced contracts persist here (code + description + utility stats). At a
new task they are RE-VERIFIED on that task's train pairs before use (sound reuse:
a library contract applies only if it actually holds for the new task), and ranked
by past utility (a frequentist prior — the sketch's "learnt statistics").

Stored as a small JSON file so it accumulates across runs/sessions.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PATH = os.path.join(_HERE, "contract_library.json")


@dataclass
class LibraryEntry:
    name: str
    descr: str
    code: str
    seen_tasks: int = 0      # tasks where it was re-verified True
    proposed: int = 0        # times induced/loaded
    solved_after: int = 0    # tasks solved after this contract was active (utility)

    def utility(self) -> float:
        return (self.solved_after + 1) / (self.seen_tasks + 2)  # Laplace-smoothed


@dataclass
class ContractLibrary:
    path: str = _DEFAULT_PATH
    entries: list[LibraryEntry] = field(default_factory=list)

    @classmethod
    def load(cls, path: str = _DEFAULT_PATH) -> "ContractLibrary":
        lib = cls(path=path)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
                lib.entries = [LibraryEntry(**e) for e in raw.get("entries", [])]
            except (json.JSONDecodeError, OSError, TypeError, KeyError, ValueError):
                lib.entries = []   # corrupt/old-schema file → start empty
        return lib

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"entries": [asdict(e) for e in self.entries]}, f, indent=2)
        except OSError:
            pass

    def has(self, code: str) -> bool:
        norm = " ".join(code.split())
        return any(" ".join(e.code.split()) == norm for e in self.entries)

    def add(self, name: str, descr: str, code: str) -> LibraryEntry:
        for e in self.entries:
            if " ".join(e.code.split()) == " ".join(code.split()):
                e.proposed += 1
                return e
        entry = LibraryEntry(name=name, descr=descr, code=code, proposed=1)
        self.entries.append(entry)
        return entry

    def candidates(self) -> list[LibraryEntry]:
        """Library entries ranked by utility (best prior first)."""
        return sorted(self.entries, key=lambda e: -e.utility())

    def record_use(self, code: str, *, verified: bool, solved: bool) -> None:
        norm = " ".join(code.split())
        for e in self.entries:
            if " ".join(e.code.split()) == norm:
                if verified:
                    e.seen_tasks += 1
                if solved:
                    e.solved_after += 1
                return
