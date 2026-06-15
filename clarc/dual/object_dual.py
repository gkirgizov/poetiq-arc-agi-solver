"""Object-correspondence dual — the STRONG information extractor.

From the train examples it induces (via the SMT matching in objsmt) the structural
contract of the transformation: how objects correspond and what is preserved or
mapped (count / shape / color / position / size). This is far richer guidance than
σ-count invariants — it tells the generator the object-level shape of the answer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from clarc.dual.base import Counterexample, Dual, Pair
from clarc.objects import segment
from clarc.objsmt import ObjectContracts, check_output, induce_object_contracts


def _nl(c: ObjectContracts) -> list[str]:
    lines = []
    seg = {"connected4": "connected regions", "connected8": "connected regions (incl. diagonals)",
           "samecolor4": "connected same-color regions", "by_color": "color groups",
           "by_row": "non-empty rows", "by_col": "non-empty columns"}.get(c.segmentation, c.segmentation)
    lines.append(f"Viewing objects as {seg}, the output has the SAME number of objects as the input.")
    for u in c.used:
        if u == "shape_exact":
            lines.append("Each object keeps its EXACT shape.")
        elif u == "shape_canon":
            lines.append("Each object keeps its shape (possibly rotated/reflected).")
        elif u == "color_preserved":
            lines.append("Each object keeps its color.")
        elif u == "color_map":
            m = {k: c.pi[k] for k in range(10) if c.pi[k] != k}
            if m:
                lines.append("Object colors are remapped: " + ", ".join(f"{a}→{b}" for a, b in m.items()) + ".")
        elif u == "size_preserved":
            lines.append("Each object keeps its size (cell count).")
        elif u == "size_scale" and c.ksz != 1:
            lines.append(f"Each object is scaled by {c.ksz}×.")
        elif u == "pos_preserved":
            lines.append("Every object stays in the same position.")
        elif u == "pos_shift" and (c.dr, c.dc) != (0, 0):
            lines.append(f"Every object moves by ({c.dr} rows, {c.dc} cols).")
        elif u == "bbox_preserved":
            lines.append("Each object keeps its bounding-box dimensions.")
    return lines


class ObjectDual:
    name = "object"

    def __init__(self) -> None:
        self.contracts: Optional[ObjectContracts] = None

    def extract(self, train_pairs: list[Pair]) -> None:
        try:
            self.contracts = induce_object_contracts(
                [(np.asarray(gi), np.asarray(go)) for gi, go in train_pairs])
        except (ValueError, KeyError):
            self.contracts = None

    def prompt_block(self) -> str:
        if self.contracts is None or not self.contracts.used:
            return ""
        lines = _nl(self.contracts)
        if len(lines) <= 1:                       # only the count line ⇒ too weak to guide
            return ""
        return ("LEARNED OBJECT STRUCTURE (the correct output MUST satisfy ALL of these — "
                "verified on every training example):\n" + "\n".join("  - " + s for s in lines))

    def refute(self, cand_pairs: list[Pair]) -> Optional[Counterexample]:
        c = self.contracts
        if c is None or not c.used:
            return None
        for k, (gi, co) in enumerate(cand_pairs, 1):
            if co is None:
                continue
            in_objs = segment(np.asarray(gi), c.segmentation)
            out_objs = segment(np.asarray(co), c.segmentation)
            if len(in_objs) != len(out_objs):
                return Counterexample(
                    "object count", f"on example {k}, your output has {len(out_objs)} "
                    f"objects but the rule keeps the input's {len(in_objs)} "
                    f"(viewing objects as {c.segmentation})")
            if not check_output(in_objs, out_objs, c):
                return Counterexample(
                    "object structure", f"on example {k}, your output's objects do not "
                    f"match the learned correspondence ({c.render()})")
        return None


assert isinstance(ObjectDual(), Dual)   # structural conformance
