"""Primitive induction (M6) — the self-extending DSL.

The transformation-space twin of `clarc/learn.py` (which induces predicates).
When the SMT sufficiency oracle reports the current library can't reach a task's
output (synth UNSAT), we ask the LLM for a NEW primitive — but ONLY its concrete
numpy `transform(grid)->grid`. The abstract z3 contract is **auto-derived**, never
written by the model: we sample σ-deltas over random grids and pick, per σ
component, the strongest relation template (eq / mul-k / const / ≤ / ≥) that holds
on ALL samples, defaulting to havoc. This makes an LLM-authored *unsound* contract
structurally impossible — the single most dangerous artifact is removed.

The gate (all automatic): (a) the code runs in the sandbox; (b) the auto-derived
contract is RE-VERIFIED on a FRESH independent random batch (different seed than
derivation — any contradiction ⇒ reject the over-strong contract); (c) it must
help (handled by the caller via the oracle). Bias is toward weak templates: a
too-weak contract only refutes less, never wrongly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import z3

from clarc.absdomain import Sigma, ZState, sigma_of
from clarc.codeparse import parse_transform_code
from clarc.dsl import Primitive
from clarc.dsltypes import Ty
from clarc.predicate_sandbox import run_transform_batch
from clarc.types import GenOutput, Generator

# σ component accessors (parallel on Sigma and ZState — same attribute names).
_INT_NAMES = (["h", "w", "n_obj", "bbox_h", "bbox_w", "n_holed", "n_border"]
              + [f"cnt{i}" for i in range(10)] + [f"osz{i}" for i in range(6)]
              + [f"ocol{i}" for i in range(10)] + [f"oshape{i}" for i in range(5)])
_BOOL_NAMES = [f"sym{i}" for i in range(5)]
_MUL_K = (2, 3, 4)


def _ival(x, name):
    if name in ("h", "w", "n_obj", "bbox_h", "bbox_w", "n_holed", "n_border"):
        return getattr(x, name)
    fam = "".join(c for c in name if not c.isdigit())   # cnt / osz / ocol / oshape
    idx = int("".join(c for c in name if c.isdigit()))
    return getattr(x, fam)[idx]


def _bval(x, name):
    return x.sym[int(name[3:])]


@dataclass(frozen=True)
class TemplateContract:
    """Per-σ-component relation templates, auto-derived from samples."""

    int_tmpl: dict      # name -> ('eq',)|('mul',k)|('const',v)|('le',)|('ge',)|('free',)
    bool_tmpl: dict     # name -> ('eq',)|('const',b)|('free',)

    def render(self) -> str:
        nontrivial = {n: t for n, t in {**self.int_tmpl, **self.bool_tmpl}.items()
                      if t[0] != "free"}
        return ", ".join(f"{n}:{t[0]}{t[1] if len(t) > 1 else ''}"
                         for n, t in sorted(nontrivial.items())) or "(all free)"


def _pick_int(ins: list[int], outs: list[int]) -> tuple:
    if all(o == i for i, o in zip(ins, outs)):
        return ("eq",)
    for k in _MUL_K:
        if all(o == k * i for i, o in zip(ins, outs)):
            return ("mul", k)
    if len(set(outs)) == 1:
        return ("const", outs[0])
    if all(o <= i for i, o in zip(ins, outs)):
        return ("le",)
    if all(o >= i for i, o in zip(ins, outs)):
        return ("ge",)
    return ("free",)


def _pick_bool(ins: list[bool], outs: list[bool]) -> tuple:
    if all(o == i for i, o in zip(ins, outs)):
        return ("eq",)
    if len(set(outs)) == 1:
        return ("const", outs[0])
    return ("free",)


def derive_contract(samples: list[tuple[Sigma, Sigma]]) -> TemplateContract:
    """Pick the strongest per-component template consistent with ALL samples."""
    int_tmpl = {n: _pick_int([_ival(si, n) for si, _ in samples],
                             [_ival(so, n) for _, so in samples]) for n in _INT_NAMES}
    bool_tmpl = {n: _pick_bool([_bval(si, n) for si, _ in samples],
                               [_bval(so, n) for _, so in samples]) for n in _BOOL_NAMES}
    return TemplateContract(int_tmpl, bool_tmpl)


def _int_holds(t: tuple, a: int, b: int) -> bool:
    return {"eq": b == a, "mul": t[0] == "mul" and b == t[1] * a,
            "const": t[0] == "const" and b == t[1],
            "le": b <= a, "ge": b >= a, "free": True}[t[0]]


def _bool_holds(t: tuple, a: bool, b: bool) -> bool:
    return {"eq": b == a, "const": t[0] == "const" and b == bool(t[1]),
            "free": True}[t[0]]


def derive_contract_verified(deriv: list[tuple[Sigma, Sigma]],
                             fresh: list[tuple[Sigma, Sigma]]) -> TemplateContract:
    """Strongest templates over `deriv`, then DOWNGRADE to free any component a
    `fresh` sample contradicts. The result is sound on deriv ∪ fresh by
    construction (downgrading to free can only remove violations), so a degenerate
    derivation batch yields a weaker-but-sound contract instead of a rejection."""
    c = derive_contract(deriv)
    int_t, bool_t = dict(c.int_tmpl), dict(c.bool_tmpl)
    for n in _INT_NAMES:
        if any(not _int_holds(int_t[n], _ival(si, n), _ival(so, n)) for si, so in fresh):
            int_t[n] = ("free",)
    for n in _BOOL_NAMES:
        if any(not _bool_holds(bool_t[n], _bval(si, n), _bval(so, n)) for si, so in fresh):
            bool_t[n] = ("free",)
    return TemplateContract(int_t, bool_t)


def make_encode(contract: TemplateContract):
    """Build a z3 encode(si, so, P) from the derived templates."""
    def encode(si: ZState, so: ZState, P) -> list[z3.BoolRef]:
        cs: list[z3.BoolRef] = []
        for name, t in contract.int_tmpl.items():
            a, b = _ival(si, name), _ival(so, name)
            if t[0] == "eq":
                cs.append(b == a)
            elif t[0] == "mul":
                cs.append(b == t[1] * a)
            elif t[0] == "const":
                cs.append(b == t[1])
            elif t[0] == "le":
                cs.append(b <= a)
            elif t[0] == "ge":
                cs.append(b >= a)
        for name, t in contract.bool_tmpl.items():
            a, b = _bval(si, name), _bval(so, name)
            if t[0] == "eq":
                cs.append(b == a)
            elif t[0] == "const":
                cs.append(b == z3.BoolVal(bool(t[1])))
        return cs
    return encode


def verify_contract(contract: TemplateContract,
                    samples: list[tuple[Sigma, Sigma]]) -> bool:
    """Soundness gate: the derived contract must be SAT against every (concrete)
    sample. Run on a FRESH batch — a contradiction means the template over-claims."""
    enc = make_encode(contract)
    for si_c, so_c in samples:
        s = z3.Solver()
        si, so = ZState("vi"), ZState("vo")
        s.add(*si.wf(), *so.wf())
        for grp in si.eq_concrete(si_c).values():
            s.add(grp)
        for grp in so.eq_concrete(so_c).values():
            s.add(grp)
        s.add(*enc(si, so, {}))
        if s.check() != z3.sat:
            return False
    return True


def _rand_grids(rng, n: int) -> list[np.ndarray]:
    """Varied grids that EXERCISE the σ components: some with many objects (to
    populate osz3..5 / oshape), some with holes, varied sizes/colors."""
    grids = []
    for t in range(n):
        h = int(rng.integers(4, 16))
        # ~half square (so domain-restricted transforms, e.g. transpose-based,
        # get enough in-domain samples), the rest rectangular.
        w = h if rng.random() < 0.5 else int(rng.integers(4, 16))
        g = np.zeros((h, w), dtype=int)
        n_obj = int(rng.integers(1, 9)) if t % 2 else int(rng.integers(5, 12))
        for _ in range(n_obj):
            i, j = int(rng.integers(0, h)), int(rng.integers(0, w))
            col = int(rng.integers(1, 10))
            if rng.random() < 0.25 and i + 2 < h and j + 2 < w:   # holed square
                g[i:i + 3, j:j + 3] = col
                g[i + 1, j + 1] = 0
            else:
                g[i:i + int(rng.integers(1, 4)), j:j + int(rng.integers(1, 4))] = col
        grids.append(g)
    return grids


async def _samples(code: str, grids: list[np.ndarray], timeout_s: float):
    outs = await run_transform_batch(code, grids, timeout_s=timeout_s)
    if outs is None:
        return None
    return [(sigma_of(gi), sigma_of(go))
            for gi, go in zip(grids, outs) if go is not None]


@dataclass
class InducedPrimitive:
    name: str
    descr: str
    code: str               # the LLM's transform(grid) source
    contract: TemplateContract

    def to_primitive(self) -> Primitive:
        return Primitive(self.name, "induced", (), self.descr,
                         _make_apply(self.code), make_encode(self.contract),
                         in_type=Ty.GRID, out_type=Ty.GRID, code=self.code)


def _make_apply(code: str):
    """In-process apply for the induced prim (used by run_pipeline outside the
    sandbox, e.g. oracle model-decode / probes). Heavy/untrusted execution in the
    solve loop still goes through compile_pipeline -> sandbox. Compiled once."""
    ns: dict = {"np": np}
    exec(compile(code, "<induced>", "exec"), ns)   # noqa: S102 — research tool, code already sandbox-vetted
    fn = ns.get("transform")

    def apply(grid, params):
        return np.asarray(fn(np.asarray(grid, dtype=int)), dtype=int)
    return apply


async def induce_primitive(
    generator: Generator,
    train_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    name: str,
    seed: int,
    n_derive: int = 24,
    n_verify: int = 24,
    min_samples: int = 8,
    timeout_s: float = 10.0,
    report: Optional[dict] = None,
) -> Optional[InducedPrimitive]:
    """Ask for a transform, AUTO-DERIVE its contract, GATE it. Returns an
    InducedPrimitive or None. `report` (if given) records where it died."""
    if report is None:
        report = {}
    g: GenOutput = await generator.generate(build_induce_prompt(train_pairs), seed=seed)
    code = parse_transform_code(g.raw or "")
    report.update(stage="no-code", cost_usd=g.cost_usd, descr=None)
    if not code or "def transform" not in code:
        return None
    descr = _first_doc(g.raw or "") or name
    return await gate_code(code, name, descr, train_pairs, seed=seed,
                           n_derive=n_derive, n_verify=n_verify,
                           min_samples=min_samples, timeout_s=timeout_s, report=report)


async def gate_code(
    code: str, name: str, descr: str,
    train_pairs: list[tuple[np.ndarray, np.ndarray]], *,
    seed: int, n_derive: int = 24, n_verify: int = 24, min_samples: int = 8,
    timeout_s: float = 10.0, report: Optional[dict] = None,
) -> Optional[InducedPrimitive]:
    """Derive + GATE a transform's contract (shared by fresh induction and
    library reuse). The LLM never enters here — only its proposed `code`."""
    if report is None:
        report = {}
    # Derivation batch = rich random grids + the actual train inputs.
    drng = np.random.default_rng(seed)
    deriv = await _samples(code, _rand_grids(drng, n_derive) + [gi for gi, _ in train_pairs],
                           timeout_s)
    report["stage"] = "exec-fail"
    if deriv is None or len(deriv) < min_samples:
        return None
    # Fresh, independent batch (different seed). Strongest templates over `deriv`,
    # then DOWNGRADE any component `fresh` contradicts (anti-overfit: an
    # over-claimed template never survives to refute; the prim is still admitted
    # with its sound, weaker contract).
    frng = np.random.default_rng(seed + 9973)
    fresh = await _samples(code, _rand_grids(frng, n_verify), timeout_s)
    report["stage"] = "verify-fail"
    if fresh is None or len(fresh) < min_samples:
        return None
    contract = derive_contract_verified(deriv, fresh)
    if not verify_contract(contract, fresh):       # belt-and-suspenders (z3)
        return None
    report.update(stage="admitted", descr=descr, contract=contract.render())
    return InducedPrimitive(name=name, descr=descr, code=code, contract=contract)


def _first_doc(text: str) -> Optional[str]:
    import re
    m = re.search(r'DESCRIPTION\s*=\s*["\'](.+?)["\']', text)
    return m.group(1) if m else None


_INDUCE_PROMPT = """You are extending an ARC solver's library with ONE new grid transformation.

The existing primitives cannot express this task. Study the input->output training
pairs and write ONE Python function capturing the GENERAL transformation rule:

```python
import numpy as np
DESCRIPTION = "<short description of the rule>"
def transform(grid: np.ndarray) -> np.ndarray:
    # general, not memorized; works for any grid following the same rule
    ...
```

Rules: pure function, numpy/scipy only, no I/O. Make it GENERAL (it will be tested
on other grids) — do not hard-code these specific grids' sizes or colors. Return
ONLY the code block.

TRAINING PAIRS:
{pairs}
"""


def build_induce_prompt(train_pairs) -> str:
    from arc_agi.solve_coding import _example_to_diagram
    blocks = []
    for i, (gi, go) in enumerate(train_pairs, 1):
        blocks.append(f"Pair {i} input:\n{_example_to_diagram(np.asarray(gi).tolist())}\n"
                      f"Pair {i} output:\n{_example_to_diagram(np.asarray(go).tolist())}")
    return _INDUCE_PROMPT.format(pairs="\n\n".join(blocks))
