"""M6a: auto-derived contracts for induced primitives + the soundness gate."""

from __future__ import annotations

import numpy as np
import pytest
import z3

from clarc.absdomain import ZState, sigma_of
from clarc.generator import StubGenerator
from clarc.learn_prim import (
    TemplateContract,
    derive_contract,
    derive_contract_verified,
    induce_primitive,
    make_encode,
    verify_contract,
)
from clarc.smt import TaskSMT

rng = np.random.default_rng(0)


def _grids(n):
    out = []
    for _ in range(n):
        h, w = int(rng.integers(3, 12)), int(rng.integers(3, 12))
        g = np.zeros((h, w), dtype=int)
        for _ in range(int(rng.integers(1, 6))):
            i, j = int(rng.integers(0, h)), int(rng.integers(0, w))
            g[i:i + int(rng.integers(1, 3)), j:j + int(rng.integers(1, 3))] = int(rng.integers(1, 10))
        out.append(g)
    return out


def _samples(fn):
    gs = _grids(30)
    return [(sigma_of(g), sigma_of(fn(g))) for g in gs]


def test_derive_identity_is_eq():
    c = derive_contract(_samples(lambda g: g.copy()))
    assert c.int_tmpl["h"] == ("eq",) and c.int_tmpl["w"] == ("eq",)
    assert c.int_tmpl["n_obj"] == ("eq",)
    assert all(c.bool_tmpl[f"sym{i}"] == ("eq",) for i in range(5))


def test_derive_scale_is_mul():
    c = derive_contract(_samples(lambda g: np.kron(g, np.ones((2, 2), dtype=int))))
    assert c.int_tmpl["h"] == ("mul", 2) and c.int_tmpl["w"] == ("mul", 2)
    assert c.int_tmpl["n_obj"] == ("eq",)          # kron preserves component count
    assert c.int_tmpl["cnt0"] == ("mul", 4)        # every color quadruples


def test_derive_recolor_sound_and_dims_eq():
    # a color-swap preserves dims exactly; whatever else is derived must be SOUND
    # on a fresh batch (the real invariant — exact templates are distribution-dep).
    def recolor(g):
        out = g.copy(); out[g == 1] = 2; return out
    deriv, fresh = _samples(recolor), _samples(recolor)
    c = derive_contract_verified(deriv, fresh)
    assert c.int_tmpl["h"] == ("eq",) and c.int_tmpl["w"] == ("eq",)
    assert verify_contract(c, _samples(recolor))   # sound on a third batch


def test_verify_rejects_overstrong_contract():
    # claim n_obj is preserved by a transform that clearly changes it
    bad = TemplateContract(int_tmpl={"n_obj": ("eq",)}, bool_tmpl={})
    # samples from "keep only largest object" — n_obj changes
    def keep_largest(g):
        from scipy import ndimage
        from clarc.contracts import bg as bgf
        m = g != bgf(g)
        lab, n = ndimage.label(m, structure=ndimage.generate_binary_structure(2, 1))
        if n == 0:
            return g.copy()
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        out = np.full_like(g, bgf(g)); keep = lab == sizes.argmax(); out[keep] = g[keep]
        return out
    samples = _samples(keep_largest)
    # at least one sample must contradict "n_obj eq" → verify returns False
    assert verify_contract(bad, samples) is False


def test_make_encode_refutes_via_derived_contract():
    # derive a scale(2,2) contract, then it must refute a same-dims candidate task
    c = derive_contract(_samples(lambda g: np.kron(g, np.ones((2, 2), dtype=int))))
    enc = make_encode(c)
    # build a tiny task whose dims DON'T double; the contract should be UNSAT
    s = z3.Solver()
    si, so = ZState("i"), ZState("o")
    s.add(*si.wf(), *so.wf())
    s.add(si.h == 4, si.w == 4, so.h == 4, so.w == 4)   # dims unchanged
    s.add(*enc(si, so, {}))
    assert s.check() == z3.unsat        # mul-2 dims contradict unchanged dims


@pytest.mark.asyncio
async def test_induce_end_to_end_with_stub():
    """A stub 'LLM' proposes a doubling transform; induction derives a sound
    contract and the result refutes a non-doubling candidate."""
    code = ('import numpy as np\nDESCRIPTION = "double via kron"\n'
            'def transform(grid):\n    return np.kron(grid, np.ones((2,2), dtype=int))\n')
    gen = StubGenerator([f"```python\n{code}\n```"])
    g = np.array([[1, 0], [0, 2]])
    pairs = [(g, np.kron(g, np.ones((2, 2), dtype=int)))]
    report: dict = {}
    ind = await induce_primitive(gen, pairs, name="induced_0", seed=1, report=report)
    assert ind is not None, report
    assert report["stage"] == "admitted"
    assert ind.contract.int_tmpl["h"] == ("mul", 2)
    # the induced primitive's contract is sound and usable in a Primitive
    prim = ind.to_primitive()
    assert prim.encode(ZState("a"), ZState("b"), {})        # builds constraints
    # and its apply actually doubles
    assert prim.apply(np.array([[5]]), {}).shape == (2, 2)
