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
from clarc.dslobj import grid_to_selection, to_grid
from clarc.dslparse import DslError, extract_dsl_block, parse_pipeline
from clarc.dsltypes import Ty

rng = np.random.default_rng(7)

N_GRIDS = 60  # per primitive; x param samples


def _random_grid(r=None) -> np.ndarray:
    r = r or rng
    h, w = r.integers(1, 13), r.integers(1, 13)
    n_colors = int(r.integers(1, 5))
    colors = r.choice(10, size=n_colors, replace=False)
    g = r.choice(colors, size=(h, w))
    if r.random() < 0.3:  # bias toward sparse "objecty" grids
        m = r.random((h, w)) < 0.7
        g = np.where(m, colors[0], g)
    if r.random() < 0.15 and h == w:  # occasionally symmetric squares
        g = np.maximum(g, g.T)
    return g.astype(int)


def _random_multiobj_grid(r=None) -> np.ndarray:
    """Sparse grid with varied shapes — stresses osz/ocol/oshape/holes/border."""
    r = r or rng
    h, w = int(r.integers(7, 15)), int(r.integers(7, 15))
    g = np.zeros((h, w), dtype=int)
    for _ in range(int(r.integers(2, 7))):
        i, j = int(r.integers(0, h - 1)), int(r.integers(0, w - 1))
        col = int(r.integers(1, 10))
        kind = int(r.integers(0, 6))
        if kind == 0:                               # dot
            g[i, j] = col
        elif kind == 1:                             # hline
            g[i, j:j + int(r.integers(2, 4))] = col
        elif kind == 2:                             # vline
            g[i:i + int(r.integers(2, 4)), j] = col
        elif kind == 3:                             # solid rect
            g[i:i + 2, j:j + 2] = col
        elif kind == 4 and i + 2 < h and j + 2 < w:  # hollow square (holed)
            g[i:i + 3, j:j + 3] = col
            g[i + 1, j + 1] = 0
        else:                                       # small cross (other)
            if 0 < i < h - 1 and 0 < j < w - 1:
                g[i, j] = g[i - 1, j] = g[i + 1, j] = g[i, j - 1] = g[i, j + 1] = col
    return g


def _random_params(prim) -> dict:
    return {p.name: p.values[rng.integers(0, len(p.values))] for p in prim.params}


def _input_for(prim, g):
    """Provide the value of the primitive's in_type (M5: GRID or SELECTION)."""
    return grid_to_selection(g) if prim.in_type == Ty.SELECTION else g


def _check_sound(prim, g: np.ndarray, params: dict) -> None:
    from clarc.absdomain import MAX_DIM
    try:
        val = _input_for(prim, g)
        out = prim.apply(val, params)
    except DslRuntimeError:
        return  # precondition failed concretely — nothing to verify
    og = to_grid(out)
    if og.shape[0] > MAX_DIM or og.shape[1] > MAX_DIM:
        return  # output out of σ's bounded domain (run_pipeline would reject it)
    # σ describes the rendered grid on both sides of the dual.
    si_c, so_c = sigma_of(to_grid(val)), sigma_of(og)

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
        _check_sound(prim, _random_multiobj_grid(), _random_params(prim))


def test_pipeline_contract_sound_composed():
    """Two-step TYPE-COMPATIBLE pipelines: composition must be satisfiable
    end-to-end (intermediate state free, only endpoints pinned)."""
    from clarc.absdomain import MAX_DIM
    lrng = np.random.default_rng(11)   # local rng: deterministic regardless of test order
    names = sorted(REGISTRY)
    for _ in range(300):
        a, b = lrng.choice(names, 2)
        pa, pb = REGISTRY[a], REGISTRY[b]
        if pa.in_type != Ty.GRID or pa.out_type != pb.in_type:
            continue  # not a well-typed Grid→…→? composition
        params_a = {p.name: p.values[lrng.integers(0, len(p.values))] for p in pa.params}
        params_b = {p.name: p.values[lrng.integers(0, len(p.values))] for p in pb.params}
        g = _random_grid(lrng) if lrng.random() < 0.5 else _random_multiobj_grid(lrng)
        try:
            out_val = pa.apply(g, params_a)
            mid = to_grid(out_val)
            out = to_grid(pb.apply(out_val, params_b))
        except DslRuntimeError:
            continue
        # run_pipeline bounds EVERY step; skip if any grid (incl. the intermediate)
        # exceeds σ's domain — such a pipeline never executes for real.
        if max(mid.shape) > MAX_DIM or max(out.shape) > MAX_DIM:
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


def test_types_thread_correctly():
    # M1 whole-grid prims are Grid->Grid; the bridge crosses to/from Selection.
    assert REGISTRY["rot180"].in_type == Ty.GRID == REGISTRY["rot180"].out_type
    assert REGISTRY["objects"].in_type == Ty.GRID
    assert REGISTRY["objects"].out_type == Ty.SELECTION
    assert REGISTRY["render"].out_type == Ty.GRID
    parse_pipeline("rot90(); flip_h(); crop_bbox(); scale(2,2)")          # all Grid
    parse_pipeline("objects(); select_largest(); recolor_all(3); render()")  # crosses
    # ill-typed: object op on a Grid, and a pipeline not ending in Grid
    from clarc.dslparse import DslTypeError
    with pytest.raises(DslTypeError):
        parse_pipeline("select_largest()")
    with pytest.raises(DslTypeError):
        parse_pipeline("objects(); render(); select_color(2)")
    with pytest.raises(DslTypeError):
        parse_pipeline("objects()")  # ends as Selection


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
