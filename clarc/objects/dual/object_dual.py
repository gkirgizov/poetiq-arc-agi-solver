"""Object-correspondence dual — the STRONG information extractor.

From the train examples it induces (via the SMT matching in objsmt) the structural
contract of the transformation: how objects correspond and what is preserved or
mapped (count / shape / color / position / size). This is far richer guidance than
σ-count invariants — it tells the generator the object-level shape of the answer.

STATUS — settled NEGATIVE result (see docs/notebook/RD_NEXT.md): across every paid
regime the counterexample channel fires but does NOT guide a code-gen solver (the
contracts justified by 3-4 examples can't see content errors; the ones that can are
coincidental and overfit). The dual is kept only as the PORTFOLIO FLOOR (G0 always
runs, so the union of train-verified solves is ≥ A0 by construction) and as the
verified-selection signal (arm G5). The gate / strong-CE / soft-prompt paths (arms
G3/G4) are retained as the recorded experiments behind that conclusion; do not invest
further in the CE channel for this generator class.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from clarc.objects.dual.base import Counterexample, Dual, Pair
from clarc.objects.conf import forward_solves
from clarc.objects.conf import gate as gate_contracts
from clarc.objects.conf import tier_names
from clarc.objects.base import segment
from clarc.objects.smt import (ObjectContracts, TermViolation, check_output, diagnose_output,
                          induce_object_contracts)
from clarc.objects.solve import apply_contracts

_TIER_RANK = {"hard": 2, "soft": 1, "drop": 0}


def _count_line(c: ObjectContracts) -> str:
    seg = {"connected4": "connected regions", "connected8": "connected regions (incl. diagonals)",
           "samecolor4": "connected same-color regions", "by_color": "color groups",
           "by_row": "non-empty rows", "by_col": "non-empty columns"}.get(c.segmentation, c.segmentation)
    return f"Viewing objects as {seg}, the output has the SAME number of objects as the input."


def _term_line(u: str, c: ObjectContracts) -> Optional[str]:
    """NL for one menu term, or None when it carries no information (identity params)."""
    if u == "shape_exact":
        return "Each object keeps its EXACT shape."
    if u == "shape_canon":
        return "Each object keeps its shape (possibly rotated/reflected)."
    if u == "color_preserved":
        return "Each object keeps its color."
    if u == "color_map":
        m = {k: c.pi[k] for k in range(10) if c.pi[k] != k}
        return ("Object colors are remapped: "
                + ", ".join(f"{a}→{b}" for a, b in m.items()) + ".") if m else None
    if u == "size_preserved":
        return "Each object keeps its size (cell count)."
    if u == "size_scale":
        return f"Each object is scaled by {c.ksz}×." if c.ksz != 1 else None
    if u == "pos_preserved":
        return "Every object stays in the same position."
    if u == "pos_shift":
        return (f"Every object moves by ({c.dr} rows, {c.dc} cols)."
                if (c.dr, c.dc) != (0, 0) else None)
    if u == "bbox_preserved":
        return "Each object keeps its bounding-box dimensions."
    return None


def _nl(c: ObjectContracts, names: Optional[list[str]] = None) -> list[str]:
    """Count line + a term line per (informative) contract in `names` (default: all)."""
    lines = [_count_line(c)]
    for u in (c.used if names is None else names):
        s = _term_line(u, c)
        if s:
            lines.append(s)
    return lines


_HARD_HEADER = ("LEARNED OBJECT STRUCTURE (the correct output MUST satisfy ALL of these — "
                "verified on every training example):")
# Example-deferring phrasing: a coincidental contract that contradicts the visible
# examples should not override them. Genuine constraints never conflict, so this is
# ~zero-risk for real wins while curbing the over-constraining overfit (e.g. 1c56ad9f).
_SOFT_HEADER = ("LEARNED OBJECT STRUCTURE (holds on every training example; the output "
                "should satisfy these. If any ever conflicts with the actual input→output "
                "examples, TRUST THE EXAMPLES):")
# Hint block for low-confidence (soft-tier) contracts: present the structure but make
# clear it may be incidental on tiny train sets — the mode-2 over-constraining fix.
_HINT_HEADER = ("OBSERVED OBJECT STRUCTURE (held on every training example, but may be "
                "incidental on so few examples — use only as a weak hint, and TRUST THE "
                "EXAMPLES if they disagree):")


class ObjectDual:
    name = "object"

    def __init__(self, *, gate: bool = False, strong_ce: bool = False,
                 soft_prompt: bool = False, ce_tier_min: str = "hard") -> None:
        self.contracts: Optional[ObjectContracts] = None
        self.gate = gate                 # confidence-tier the contracts (G3); False = ungated G1
        self.strong_ce = strong_ce       # witness-decoding refutation (G3); False = vague G1 CE
        self.soft_prompt = soft_prompt    # example-deferring header (G3); False keeps G1 verbatim
        self.ce_tier_min = ce_tier_min    # lowest tier the general CE path may refute on
        self.train_pairs: list[Pair] = []
        self._solves = False              # full contracts forward-solve train (sound cell-diff CE)

    def extract(self, train_pairs: list[Pair]) -> None:
        self.train_pairs = [(np.asarray(gi), np.asarray(go)) for gi, go in train_pairs]
        self._solves = False
        try:
            self.contracts = induce_object_contracts(self.train_pairs)
        except (ValueError, KeyError):
            self.contracts = None
        if self.contracts is not None:
            if self.gate:
                gate_contracts(self.contracts, self.train_pairs)
            self._solves = forward_solves(self.contracts, self.train_pairs)

    def _block(self, header: str, names: list[str], with_count: bool) -> Optional[str]:
        terms = [s for u in names if (s := _term_line(u, self.contracts))]
        if not terms:
            return None
        lines = ([_count_line(self.contracts)] if with_count else []) + terms
        return header + "\n" + "\n".join("  - " + s for s in lines)

    def prompt_block(self) -> str:
        c = self.contracts
        if c is None or not c.used:
            return ""
        if not (self.gate and c.tiers):              # ungated: byte-identical to G1
            lines = _nl(c)
            if len(lines) <= 1:                       # only the count line ⇒ too weak to guide
                return ""
            header = _SOFT_HEADER if self.soft_prompt else _HARD_HEADER
            return header + "\n" + "\n".join("  - " + s for s in lines)
        # gated: a firm HARD block (confident) above a deferring HINT block (weak)
        h_header = _SOFT_HEADER if self.soft_prompt else _HARD_HEADER
        hard = self._block(h_header, tier_names(c, "hard"), with_count=True)
        soft = self._block(_HINT_HEADER, tier_names(c, "soft"), with_count=hard is None)
        return "\n\n".join(b for b in (hard, soft) if b)

    def active_names(self) -> list[str]:
        """Invariant names actually injected into the prompt (for per-iteration logs)."""
        c = self.contracts
        if c is None or not self.prompt_block():
            return []
        if self.gate and c.tiers:
            return tier_names(c, "hard") + tier_names(c, "soft")
        return list(c.used)

    def _refute_used(self) -> list[str]:
        """Contracts the GENERAL witness path may refute on (≥ ce_tier_min tier). On a
        coincidental (soft) contract a CE would push the LLM toward overfit, so by
        default only HARD-tier contracts drive general refutation."""
        c = self.contracts
        if not (self.gate and c.tiers):
            return list(c.used)
        floor = _TIER_RANK.get(self.ce_tier_min, 2)
        return [n for n in c.used if _TIER_RANK.get(c.tiers.get(n), 0) >= floor]

    def refute(self, cand_pairs: list[Pair]) -> Optional[Counterexample]:
        c = self.contracts
        if c is None or not c.used:
            return None
        if not self.strong_ce:
            return self._refute_vague(cand_pairs)
        used = self._refute_used()
        viols: list[TermViolation] = []
        for k, (gi, co) in enumerate(cand_pairs, 1):
            if co is None:
                continue
            gi, co = np.asarray(gi), np.asarray(co)
            in_objs = segment(gi, c.segmentation)
            out_objs = segment(co, c.segmentation)
            if len(in_objs) != len(out_objs):          # count mismatch: sound, name the orphans
                viols += diagnose_output(in_objs, out_objs, c, example=k, names=used)
                continue
            if self._solves:                            # contracts construct the verified answer
                fwd = apply_contracts(gi, c)
                if fwd is not None and fwd.shape == co.shape and not np.array_equal(fwd, co):
                    viols.append(_cell_witness(k, fwd, co))
                    continue
            if used:                                    # general witness on confident contracts only
                rc = _restrict(c, used)
                if not check_output(in_objs, out_objs, rc):
                    viols += diagnose_output(in_objs, out_objs, rc, example=k, names=used)
        if not viols:
            return None
        return Counterexample("object structure", _render_violations(viols), violations=viols)

    def _refute_vague(self, cand_pairs: list[Pair]) -> Optional[Counterexample]:
        """The original first-violation, unlocalized CE — kept verbatim so G1 is a faithful
        control (no tiering, no witness decoding)."""
        c = self.contracts
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


def _restrict(c: ObjectContracts, names: list[str]) -> ObjectContracts:
    return ObjectContracts(segmentation=c.segmentation, used=list(names), pi=list(c.pi),
                           dr=c.dr, dc=c.dc, ksz=c.ksz, tiers=c.tiers)


def _cell_witness(k: int, expected: np.ndarray, actual: np.ndarray) -> TermViolation:
    diff = np.argwhere(expected != actual)
    r, col = int(diff[0][0]), int(diff[0][1])
    return TermViolation(k, "cell", "cell", f"color {int(expected[r, col])}",
                         f"color {int(actual[r, col])}", None, (r, col), n_objects_hit=len(diff))


def _render_violations(viols: list[TermViolation], max_lines: int = 3) -> str:
    lines = []
    for v in viols[:max_lines]:
        if v.kind == "cell":
            lines.append(f"Example {v.example}: cell {tuple(v.out_pos)} is {v.actual} but should "
                         f"be {v.expected}.")
        elif v.kind == "term":
            lines.append(f"Example {v.example}: the object at {tuple(v.out_pos)} is {v.actual} but "
                         f"should be {v.expected}.")
        elif v.kind == "extra":
            lines.append(f"Example {v.example}: your output has {v.actual}, which the rule does "
                         f"not produce.")
        elif v.kind == "missing":
            lines.append(f"Example {v.example}: {v.actual} — the rule expects {v.expected}.")
    return "\n".join(lines)


assert isinstance(ObjectDual(), Dual)   # structural conformance
