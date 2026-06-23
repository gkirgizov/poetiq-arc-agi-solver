"""ARC dataset loading — deliberately dependency-free (no generator/LLM import).

Kept separate from ``clarc.run`` so the $0 probes (``synth_coverage``, ``probe_dsl``,
…) can load tasks WITHOUT importing the generator subsystem. ``load`` returns
``(challenges, solutions | None)``; ``solutions`` is ``None`` when the split ships no
answer key (e.g. the held-out test set).
"""

from __future__ import annotations

import json
import os
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")

DATASETS = {
    "2024-train": ("arc-prize-2024", "training"),
    "2024-eval": ("arc-prize-2024", "evaluation"),
    "2025-train": ("arc-prize-2025", "training"),
    "2025-eval": ("arc-prize-2025", "evaluation"),
}


def load(dataset: str) -> tuple[dict, Optional[dict]]:
    """Load an ARC challenge split → (challenges, solutions|None)."""
    folder, split = DATASETS[dataset]
    base = os.path.join(_DATA, folder)
    with open(os.path.join(base, f"arc-agi_{split}_challenges.json"), encoding="utf-8") as f:
        challenges = json.load(f)
    solutions = None
    sol_path = os.path.join(base, f"arc-agi_{split}_solutions.json")
    if os.path.exists(sol_path):
        with open(sol_path, encoding="utf-8") as f:
            solutions = json.load(f)
    return challenges, solutions
