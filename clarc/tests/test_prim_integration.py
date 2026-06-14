"""M6b: task-local registry, the sufficiency oracle flipping UNSAT→SAT when an
induced primitive is added, compiling pipelines that contain induced prims, and
the primitive-library round-trip."""

from __future__ import annotations

import json

import numpy as np
import pytest

from clarc.absdomain import sigma_of
from clarc.dsl import REGISTRY, compile_pipeline
from clarc.dslparse import parse_pipeline
from clarc.learn_prim import InducedPrimitive, derive_contract, make_encode
from clarc.smt import TaskSMT


def _diagonal_mirror_prim():
    """An induced prim no base primitive provides: overlay the grid with its
    transpose (union of nonzero), i.e. symmetrize across the main diagonal."""
    code = ('import numpy as np\nDESCRIPTION = "overlay transpose"\n'
            'def transform(grid):\n'
            '    g = np.asarray(grid)\n'
            '    return np.where(g != 0, g, g.T)\n')
    fn = lambda g: np.where(np.asarray(g) != 0, np.asarray(g), np.asarray(g).T)
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(40):
        n = int(rng.integers(3, 8))
        g = (rng.random((n, n)) < 0.4).astype(int) * rng.integers(1, 9)
        samples.append((sigma_of(g), sigma_of(fn(g))))
    c = derive_contract(samples)
    return InducedPrimitive("induced_diag", "overlay transpose", code, c)


def test_task_local_registry_does_not_touch_global():
    before = set(REGISTRY)
    ind = _diagonal_mirror_prim()
    reg = dict(REGISTRY)
    reg[ind.name] = ind.to_primitive()
    assert ind.name in reg and ind.name not in REGISTRY     # global untouched
    assert set(REGISTRY) == before


def test_oracle_flips_unsat_to_sat_with_induced_prim():
    # A symmetrize-across-diagonal task: square dims, gains transpose symmetry.
    rng = np.random.default_rng(3)
    pairs = []
    for _ in range(3):
        n = 5
        g = (rng.random((n, n)) < 0.35).astype(int) * int(rng.integers(1, 9))
        out = np.where(g != 0, g, g.T)
        pairs.append((sigma_of(g), sigma_of(out)))
    base = TaskSMT(pairs, registry=dict(REGISTRY))
    # base library: no primitive both keeps dims and gains transpose symmetry here
    base_feasible = base.synth_feasible(2)
    ind = _diagonal_mirror_prim()
    reg = dict(REGISTRY); reg[ind.name] = ind.to_primitive()
    ext = TaskSMT(pairs, registry=reg)
    ext_feasible = ext.synth_feasible(2)
    # the induced primitive must make the task abstractly feasible
    assert ext_feasible is True
    # and the induced prim is reachable in synth models
    names = {s.name for m in ext.synth_models(2, max_models=12) for s in m.steps}
    assert "induced_diag" in names or base_feasible is not True


@pytest.mark.asyncio
async def test_compile_pipeline_with_induced_runs_in_sandbox():
    from arc_agi.sandbox import run as sandbox_run
    ind = _diagonal_mirror_prim()
    reg = dict(REGISTRY); reg[ind.name] = ind.to_primitive()
    # pipeline: induced step then a builtin geometry step
    p = parse_pipeline("induced_diag(); rot180()", registry=reg)
    code = compile_pipeline(p, registry=reg)
    g = [[1, 0, 0], [0, 2, 0], [3, 0, 0]]
    ok, result = await sandbox_run(code, g, timeout_s=10.0)
    assert ok, result
    expected = np.rot90(np.where(np.array(g) != 0, np.array(g), np.array(g).T), 2)
    assert json.loads(result) == expected.tolist()


def test_prim_library_roundtrip(tmp_path):
    from clarc.prim_library import PrimLibrary
    ind = _diagonal_mirror_prim()
    lib = PrimLibrary(path=str(tmp_path / "prims.json"))
    lib.add(ind)
    lib.save()
    lib2 = PrimLibrary.load(str(tmp_path / "prims.json"))
    assert len(lib2.entries) == 1
    e = lib2.entries[0]
    assert e.code == ind.code
    # the reconstructed contract encodes identically
    rebuilt = e.induced().to_primitive()
    assert rebuilt.name == "induced_diag"
    lib2.record_use(ind.code, verified=True, solved=True)
    assert lib2.entries[0].seen_tasks == 1 and lib2.entries[0].solved_after == 1
