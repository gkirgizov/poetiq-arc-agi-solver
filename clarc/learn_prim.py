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
    code: str               # FULL scaffolded transform source (typed detect→rule→render)
    contract: TemplateContract
    kind: str = "recolor"   # which typed slot: recolor | select | (future: map)

    def to_primitive(self) -> Primitive:
        return Primitive(self.name, f"induced-{self.kind}", (), self.descr,
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
    report.update(stage="no-rule", cost_usd=g.cost_usd, descr=None)
    kind, rule_code, descr = parse_rule(g.raw or "")
    if kind is None:
        return None
    # Wrap the narrow per-object rule in the typed detect→rule→render scaffold.
    code = scaffold_source(kind, rule_code)
    report["kind"] = kind
    return await gate_code(code, name, descr or name, train_pairs, seed=seed,
                           n_derive=n_derive, n_verify=n_verify, kind=kind,
                           min_samples=min_samples, timeout_s=timeout_s, report=report)


async def gate_code(
    code: str, name: str, descr: str,
    train_pairs: list[tuple[np.ndarray, np.ndarray]], *,
    seed: int, n_derive: int = 24, n_verify: int = 24, min_samples: int = 8,
    timeout_s: float = 10.0, kind: str = "recolor", report: Optional[dict] = None,
) -> Optional[InducedPrimitive]:
    """Derive + GATE a scaffolded primitive's contract (shared by fresh induction
    and library reuse). The LLM never enters here — only the scaffolded `code`."""
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
    return InducedPrimitive(name=name, descr=descr, code=code, contract=contract, kind=kind)


# --------------------------------------------------------------------------- #
# Typed object-decomposition scaffolding (M6, the FAITHFUL induction).
#
# The system owns object DETECTION and RENDER (the typed Grid<->Selection
# bridge); the LLM induces only a NARROW per-object rule over the generating
# basis (each object's local attributes). This enforces the decomposition
# hypothesis — the induced piece is a typed, composable building block, NOT a
# free-form whole-grid transform (which would just be code-gen). The auto-derived
# contract is then the contract of a per-object operation: a clean domain
# constraint, reusable and refutation-bearing.
# --------------------------------------------------------------------------- #

_OBJVIEW_SRC = '''
import numpy as np
from scipy import ndimage as _ndi

def _bg(g):
    v, c = np.unique(g, return_counts=True)
    return int(v[np.flatnonzero(c == c.max()).min()])

def _detect(g):
    """Detect 4-connected non-background objects; expose each one's LOCAL
    attributes (the generating basis) the induced rule reasons over."""
    bg = _bg(g)
    lab, n = _ndi.label(g != bg, structure=_ndi.generate_binary_structure(2, 1))
    H, W = g.shape
    scene = []
    for k in range(1, n + 1):
        m = lab == k
        rows = np.flatnonzero(m.any(1)); cols = np.flatnonzero(m.any(0))
        t, l, b, r = int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])
        size = int(m.sum()); bh, bw = b - t + 1, r - l + 1
        vals, cnts = np.unique(g[m], return_counts=True)
        dom = int(vals[np.flatnonzero(cnts == cnts.max()).min()])
        holed = int(_ndi.binary_fill_holes(m).sum()) > size
        shape = ('dot' if size == 1 else 'hline' if bh == 1 else 'vline' if bw == 1
                 else 'rect' if size == bh * bw else 'other')
        scene.append({'color': dom, 'size': size, 'h': bh, 'w': bw, 'top': t,
                      'left': l, 'n_holes': int(holed),
                      'is_border': bool(t == 0 or l == 0 or b == H - 1 or r == W - 1),
                      'shape': shape, 'mask': m, 'cells': g[t:b+1, l:r+1].copy()})
    return scene, bg
'''

_SCAFFOLD = {
    "recolor": '''
def transform(grid):
    g = np.asarray(grid, dtype=int)
    scene, bg = _detect(g)
    out = np.full(g.shape, bg, dtype=int)
    for o in scene:
        c = color_of(o, scene)
        out[o['mask']] = bg if c is None else int(c)
    return out
''',
    "select": '''
def transform(grid):
    g = np.asarray(grid, dtype=int)
    scene, bg = _detect(g)
    out = np.full(g.shape, bg, dtype=int)
    for o in scene:
        if keep(o, scene):
            out[o['mask']] = g[o['mask']]
    return out
''',
}

_RULE_FN = {"recolor": "color_of", "select": "keep"}


def scaffold_source(kind: str, rule_code: str) -> str:
    """Wrap an induced per-object rule in the typed detect→rule→render scaffold,
    producing a self-contained transform() (so it inlines into the sandbox)."""
    return _OBJVIEW_SRC + "\n" + rule_code.strip() + "\n" + _SCAFFOLD[kind]


# --------------------------------------------------------------------------- #
# M7c: unified object-rule over DYNAMICALLY-segmented objects with rich typed
# attributes. The recognizer (LLM) chooses HOW to see objects (the segmentation)
# AND writes one rule over the generating basis; the rule returns a per-object
# spec — recolor / move / drop — so it covers marker-erase, object movement and
# conditional recoloring that the fixed-4-conn recolor/select kinds cannot.
# --------------------------------------------------------------------------- #

_OBJ_RULE_SCAFFOLD = '''
import numpy as np
from clarc.objects import segment as _segment
from clarc.contracts import bg as _bgf
{rule_code}
def transform(grid):
    g = np.asarray(grid, dtype=int)
    bg = int(_bgf(g)); H, W = g.shape
    objs = _segment(g, "{strategy}")
    scene = [dict(color=o.color, size=o.size, top=o.top, left=o.left, h=o.bh, w=o.bw,
                  shape=o.shape, n_holes=o.n_holes, is_border=o.is_border,
                  cells=o.cells.copy()) for o in objs]
    out = np.full((H, W), bg, dtype=int)
    for o, sc in zip(objs, scene):
        spec = rule(sc, scene)
        if spec is None:
            continue                       # drop this object
        if isinstance(spec, dict):
            color = spec.get("color", sc["color"]); dr = spec.get("dr", 0); dc = spec.get("dc", 0)
        else:
            color = spec; dr = dc = 0       # bare int = recolor in place
        for i, j in zip(*np.nonzero(o.mask)):
            r, c = int(i) + dr, int(j) + dc
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = int(color)
    return out
'''


def scaffold_object_rule(strategy: str, rule_code: str) -> str:
    return _OBJ_RULE_SCAFFOLD.replace("{rule_code}", rule_code.strip()).replace(
        "{strategy}", strategy)


def parse_object_rule(raw: str):
    """Extract (segmentation, rule_code, descr) from an object-rule reply."""
    import re
    from clarc.objects import SEGMENTERS
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", raw or "", re.S | re.I)
    body = blocks[-1] if blocks else (raw or "")
    if "def rule" not in body:
        return None, None, None
    sm = re.search(r'SEGMENT\s*=\s*["\'](\w+)["\']', body)
    strategy = sm.group(1) if sm and sm.group(1) in SEGMENTERS else "connected4"
    dm = re.search(r'DESCRIPTION\s*=\s*["\'](.+?)["\']', body)
    return strategy, body, (dm.group(1) if dm else None)


def parse_rule(raw: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract (kind, rule_code, descr) from an induction reply. Returns
    (None, None, None) if no valid rule of a known kind is present."""
    import re
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", raw or "", re.S | re.I)
    body = blocks[-1] if blocks else (raw or "")
    km = re.search(r'KIND\s*=\s*["\'](\w+)["\']', body)
    kind = km.group(1).lower() if km else None
    if kind not in _SCAFFOLD:
        # infer from the function the model wrote
        kind = next((k for k, fn in _RULE_FN.items() if f"def {fn}" in body), None)
    if kind is None or f"def {_RULE_FN[kind]}" not in body:
        return None, None, None
    dm = re.search(r'DESCRIPTION\s*=\s*["\'](.+?)["\']', body)
    return kind, body, (dm.group(1) if dm else None)


_INDUCE_PROMPT = """You are extending a TYPED, COMPOSITIONAL ARC solver. The solver already
DETECTS the objects in a grid and exposes each object's local attributes; your job is
NOT to write a whole-grid transform, but to supply ONE narrow rule that operates over
those detected objects — the missing typed building block.

Each detected object `o` is a dict with: o['color'] (dominant color), o['size'],
o['h'], o['w'] (bbox dims), o['top'], o['left'] (bbox position), o['shape']
('dot'|'hline'|'vline'|'rect'|'other'), o['n_holes'], o['is_border'], and
o['cells'] (the object's bbox as a numpy array). `scene` is the list of all objects
(use it for relational rules, e.g. ranking by size or finding a marker object).

Choose EXACTLY ONE kind of rule and write it (reason ONLY over object attributes; do
NOT re-detect objects or manipulate raw pixels):

  KIND = "recolor"   →  def color_of(o, scene): return <new color int for this object>   (None = erase to background)
  KIND = "select"    →  def keep(o, scene): return <True to keep this object, False to drop>

Reply with ONE code block: a KIND line, a DESCRIPTION line, and the function. Make it
GENERAL (it is tested on other grids and reused on other tasks) — never hard-code these
grids' specific sizes. Example:

```python
KIND = "recolor"
DESCRIPTION = "recolor each object by its size rank (largest gets 2, rest 1)"
def color_of(o, scene):
    rank = sorted({s['size'] for s in scene}, reverse=True).index(o['size'])
    return 2 if rank == 0 else 1
```

TRAINING PAIRS (with the objects detected in each input):
{pairs}
"""


def _render_objects(g: np.ndarray) -> str:
    """Show the detected objects + attributes for a train input, so the model
    reasons at the object level (not pixels)."""
    from clarc.absdomain import _object_features  # noqa: F401  (ensure import ok)
    from clarc.dslobj import grid_to_selection
    sel = grid_to_selection(g)
    lines = []
    for i, o in enumerate(sel.objs):
        bb = o.bbox
        lines.append(f"    obj{i}: color={o.color} size={o.size} "
                     f"bbox=({bb[2]-bb[0]+1}x{bb[3]-bb[1]+1})@({bb[0]},{bb[1]})")
    return "\n".join(lines) if lines else "    (no non-background objects)"


def build_induce_prompt(train_pairs) -> str:
    from arc_agi.solve_coding import _example_to_diagram
    blocks = []
    for i, (gi, go) in enumerate(train_pairs, 1):
        gi = np.asarray(gi)
        blocks.append(f"Pair {i} input:\n{_example_to_diagram(gi.tolist())}\n"
                      f"Pair {i} input objects:\n{_render_objects(gi)}\n"
                      f"Pair {i} output:\n{_example_to_diagram(np.asarray(go).tolist())}")
    # .replace (not .format): the prompt's example contains literal { } braces.
    return _INDUCE_PROMPT.replace("{pairs}", "\n\n".join(blocks))


_OBJ_PROMPT = """You are extending a TYPED, COMPOSITIONAL ARC solver. The solver can SEGMENT a
grid into objects in several ways; you choose the segmentation that makes the rule
cleanest, then write ONE rule over the detected objects' attributes.

Segmentations (pick one via SEGMENT="..."): connected4, connected8, samecolor4
(split by color too), by_color (one object per color, even if scattered),
by_row, by_col.

Each object `o` is a dict: o['color'], o['size'], o['top'], o['left'], o['h'],
o['w'] (bbox), o['shape'] ('dot'|'hline'|'vline'|'rect'|'other'), o['n_holes'],
o['is_border'], o['cells'] (numpy bbox). `scene` is all objects (for relational
rules: rank by size, find a marker, count, etc.).

Write `rule(o, scene)` returning the per-object action — reason ONLY over the
attributes, never raw pixels:
  - an int            → recolor the whole object to that color
  - {"color": c, "dr": dr, "dc": dc} → recolor to c and move by (dr rows, dc cols)
  - None              → delete the object (erase to background)

Reply with ONE code block containing SEGMENT, DESCRIPTION and def rule. Make it
GENERAL (tested on other grids, reused on other tasks). Example:

```python
SEGMENT = "by_color"
DESCRIPTION = "delete the marker color (1); recolor every other object to it"
def rule(o, scene):
    if o['color'] == 1:
        return None
    return 1
```

TRAINING PAIRS (with objects under connected4 and by_color):
{pairs}
"""


def build_object_prompt(train_pairs) -> str:
    from arc_agi.solve_coding import _example_to_diagram
    from clarc.objects import segment as _seg
    blocks = []
    for i, (gi, go) in enumerate(train_pairs, 1):
        gi = np.asarray(gi)
        views = []
        for strat in ("connected4", "by_color"):
            objs = _seg(gi, strat)
            ds = "; ".join(f"{o.color}:{o.shape}@({o.top},{o.left}) sz{o.size}"
                           for o in objs[:10])
            views.append(f"    [{strat}] {ds}")
        blocks.append(f"Pair {i} input:\n{_example_to_diagram(gi.tolist())}\n"
                      f"Pair {i} objects:\n" + "\n".join(views) + "\n"
                      f"Pair {i} output:\n{_example_to_diagram(np.asarray(go).tolist())}")
    return _OBJ_PROMPT.replace("{pairs}", "\n\n".join(blocks))


async def induce_object_rule(
    generator: Generator,
    train_pairs: list[tuple[np.ndarray, np.ndarray]], *,
    name: str, seed: int, n_derive: int = 24, n_verify: int = 24,
    min_samples: int = 8, timeout_s: float = 10.0, report: Optional[dict] = None,
) -> Optional[InducedPrimitive]:
    """M7c: induce a rule over DYNAMICALLY-segmented objects (recolor/move/drop).
    Gated by the auto-derived σ contract; the richer SMT object-contract dual is
    attached for interpretability/reuse."""
    if report is None:
        report = {}
    g: GenOutput = await generator.generate(build_object_prompt(train_pairs), seed=seed)
    report.update(stage="no-rule", cost_usd=g.cost_usd, descr=None)
    strategy, rule_code, descr = parse_object_rule(g.raw or "")
    if strategy is None:
        return None
    code = scaffold_object_rule(strategy, rule_code)
    report.update(kind="object", segmentation=strategy)
    ind = await gate_code(code, name, descr or name, train_pairs, seed=seed,
                          n_derive=n_derive, n_verify=n_verify, kind="object",
                          min_samples=min_samples, timeout_s=timeout_s, report=report)
    if ind is not None:
        # attach the SMT object-correspondence contract (the logical dual) for the log
        try:
            from clarc.objsmt import induce_object_contracts
            oc = induce_object_contracts([(np.asarray(gi), np.asarray(go))
                                          for gi, go in train_pairs])
            report["obj_contract"] = oc.render() if oc else None
        except (ValueError, KeyError):
            report["obj_contract"] = None
    return ind
