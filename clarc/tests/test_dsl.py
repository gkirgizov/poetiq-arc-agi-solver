"""Soundness fuzz suite — the load-bearing gate of the DSL⇄SMT dual.

For every primitive f and random in-domain (grid, params):
    R_f(σ(g), σ(f(g)), params)  must be SAT with BOTH endpoint states pinned
    to their concrete values (plus WF axioms). UNSAT = the abstract contract
    CONTRADICTS the concrete semantics = an unsound claim that would later
    produce false refutations. Any failure here is a contract bug.

Also: interpreter/parser/compile round-trips.
"""

from __future__ import annotations

import numpy as np
import pytest
import z3

from clarc.absdomain import ZState, sigma_of
from clarc.dsl import (
    REGISTRY,
    DslRuntimeError,
    Pipeline,
    Step,
    alloc_params,
    compile_pipeline,
    param_domain,
    param_index,
    run_pipeline,
)
from clarc.dslparse import DslError, extract_dsl_block, parse_pipeline

rng = np.random.default_rng(7)

N_GRIDS = 60  # per primitive; x param samples


def _random_grid() -> np.ndarray:
    h, w = rng.integers(1, 13), rng.integers(1, 13)
    n_colors = int(rng.integers(1, 5))
    colors = rng.choice(10, size=n_colors, replace=False)
    g = rng.choice(colors, size=(h, w))
    if rng.random() < 0.3:  # bias toward sparse "objecty" grids
        m = rng.random((h, w)) < 0.7
        g = np.where(m, colors[0], g)
    if rng.random() < 0.15 and h == w:  # occasionally symmetric squares
        g = np.maximum(g, g.T)
    return g.astype(int)


def _random_params(prim) -> dict:
    return {p.name: p.values[rng.integers(0, len(p.values))] for p in prim.params}


def _check_sound(prim, g: np.ndarray, params: dict) -> None:
    try:
        out = run_pipeline(Pipeline((Step(prim.name, params),)), g)
    except DslRuntimeError:
        return  # precondition failed concretely — nothing to verify
    si_c, so_c = sigma_of(g), sigma_of(out)

    s = z3.Solver()
    si, so = ZState("i"), ZState("o")
    s.add(*si.wf(), *so.wf())
    for grp in si.eq_concrete(si_c).values():
        s.add(grp)
    for grp in so.eq_concrete(so_c).values():
        s.add(grp)
    P = alloc_params(prim, "p")
    s.add(*param_domain(prim, P))
    for name, val in params.items():
        s.add(P[name] == param_index(prim, name, val))
    s.add(*prim.encode(si, so, P))

    if s.check() != z3.sat:
        pytest.fail(
            f"UNSOUND contract: {prim.name}({params}) on grid\n{g}\n-> \n{out}\n"
            f"sigma_in={si_c}\nsigma_out={so_c}"
        )


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_primitive_contract_sound(name):
    prim = REGISTRY[name]
    for _ in range(N_GRIDS):
        _check_sound(prim, _random_grid(), _random_params(prim))


def test_pipeline_contract_sound_composed():
    """Two-step pipelines: composition must also be satisfiable end-to-end
    (intermediate state free, only endpoints pinned)."""
    names = sorted(REGISTRY)
    for _ in range(40):
        a, b = rng.choice(names, 2)
        pa, pb = REGISTRY[a], REGISTRY[b]
        params_a, params_b = _random_params(pa), _random_params(pb)
        g = _random_grid()
        pipe = Pipeline((Step(a, params_a), Step(b, params_b)))
        try:
            out = run_pipeline(pipe, g)
        except DslRuntimeError:
            continue
        s = z3.Solver()
        s0, s1, s2 = ZState("s0"), ZState("s1"), ZState("s2")
        s.add(*s0.wf(), *s1.wf(), *s2.wf())
        for grp in s0.eq_concrete(sigma_of(g)).values():
            s.add(grp)
        for grp in s2.eq_concrete(sigma_of(out)).values():
            s.add(grp)
        Pa, Pb = alloc_params(pa, "pa"), alloc_params(pb, "pb")
        s.add(*param_domain(pa, Pa), *param_domain(pb, Pb))
        for n, v in params_a.items():
            s.add(Pa[n] == param_index(pa, n, v))
        for n, v in params_b.items():
            s.add(Pb[n] == param_index(pb, n, v))
        s.add(*pa.encode(s0, s1, Pa), *pb.encode(s1, s2, Pb))
        assert s.check() == z3.sat, f"unsound composition {a};{b} on\n{g}"


# ---------------------------------------------------------------------------
# parser / interpreter / compile
# ---------------------------------------------------------------------------

def test_parse_roundtrip():
    p = parse_pipeline("crop_bbox(); recolor(1->2, 2->1); scale(3, 3)")
    assert [s.name for s in p.steps] == ["crop_bbox", "recolor", "scale"]
    assert p.steps[1].params["pi1"] == 2 and p.steps[1].params["pi2"] == 1
    assert p.steps[1].params["pi5"] == 5  # identity default
    assert p.steps[2].params == {"kh": 3, "kw": 3}
    p2 = parse_pipeline(p.pretty())
    assert p2 == p


def test_parse_errors():
    for bad in ("", "nope()", "tile(5,1)", "half(up)", "tile(2)", "recolor(11->2)"):
        with pytest.raises(DslError):
            parse_pipeline(bad)


def test_extract_dsl_block():
    text = "thinking...\n```dsl\nrot90(); flip_h()\n```\ndone"
    assert extract_dsl_block(text) == "rot90(); flip_h()"
    assert extract_dsl_block("no block here") is None


def test_interpreter_basics():
    g = np.array([[1, 2], [3, 4]])
    assert np.array_equal(run_pipeline(parse_pipeline("rot180()"), g),
                          np.array([[4, 3], [2, 1]]))
    assert np.array_equal(run_pipeline(parse_pipeline("scale(2,2)"), g)[0:2, 0:2],
                          np.array([[1, 1], [1, 1]]))
    doubled = run_pipeline(parse_pipeline("concat_flip(v)"), g)
    assert doubled.shape == (4, 2)
    assert np.array_equal(doubled[2:], g[::-1, :])


@pytest.mark.asyncio
async def test_compile_pipeline_runs_in_sandbox():
    from arc_agi.sandbox import run as sandbox_run

    code = compile_pipeline(parse_pipeline("rot180()"))
    ok, result = await sandbox_run(code, [[1, 2], [3, 4]], timeout_s=10.0)
    assert ok, result
    import json
    assert json.loads(result) == [[4, 3], [2, 1]]
