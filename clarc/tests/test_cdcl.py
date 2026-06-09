"""Phase 2 tests: the CDCL learned-clause loop (offline, StubGenerator).

Covers the soundness gate, conflict detection/classification, the store's earned
escalation, hard pruning, and an end-to-end A1 solve.
"""

import numpy as np

from clarc.analyze import admit_clause, classify_conflict, parse_grid
from clarc.contracts import Contract
from clarc.generator import StubGenerator
from clarc.loop import solve_task
from clarc.spec import extract_spec, loo_trusted
from clarc.store import LearnedStore
from clarc.types import ClarcConfig, RunResult

# A genuine recolor task (1 -> 2), repeated colors so contracts are meaningful.
TRAIN_IN = [[[1, 0], [0, 1]], [[1, 1], [0, 0]]]
TRAIN_OUT = [[[2, 0], [0, 2]], [[2, 2], [0, 0]]]
TEST_IN = [[[0, 1], [1, 0]]]

WRONG_1x1 = "import numpy as np\ndef transform(grid):\n    return np.array([[0]])"
CORRECT = ("import numpy as np\n"
           "def transform(grid):\n"
           "    g = np.array(grid).copy()\n"
           "    g[g == 1] = 2\n"
           "    return g")


def _cfg(**kw):
    base = dict(max_iterations=5, seed=0, shuffle_examples=False, problem_id="recolor")
    base.update(kw)
    return ClarcConfig(**base)


# ---- soundness gate ----

def test_store_add_requires_verification():
    store = LearnedStore()
    c = Contract("dummy", "d", lambda gi, go: True)
    assert store.add(c, verified=False) is False        # unverified -> refused
    assert store.contracts == []
    assert store.add(c, verified=True) is True
    assert store.add(c, verified=True) is False          # dedup


def test_admit_clause_gate():
    pairs = [(np.asarray(gi), np.asarray(go)) for gi, go in zip(TRAIN_IN, TRAIN_OUT)]
    holds = Contract("same_shape", "shape eq", lambda gi, go: gi.shape == go.shape)
    fails = Contract("grew", "out bigger", lambda gi, go: go.size > gi.size)
    assert admit_clause(holds, pairs) is True
    assert admit_clause(fails, pairs) is False


# ---- conflict classification + parse ----

def test_classify_and_parse():
    assert classify_conflict([]) == "semantic"
    assert classify_conflict([Contract("x", "x", lambda a, b: True)]) == "structural"
    rr = RunResult(success=False, output="[[1,2],[3,4]]", soft_score=0.0, error=None, code="")
    g = parse_grid(rr)
    assert g is not None and g.shape == (2, 2)
    assert parse_grid(RunResult(success=False, output="boom", soft_score=0.0, error="e", code="")) is None


# ---- store escalation ----

def test_store_escalation_rendering():
    spec = extract_spec(TRAIN_IN, TRAIN_OUT, "recolor")
    store = LearnedStore(contracts=list(spec.contracts))
    name = spec.contracts[0].name
    store.note([spec.contracts[0]])
    store.note([spec.contracts[0]])  # 2nd violation -> escalate
    rendered = store.render_for_prompt()
    assert "MANDATORY" in rendered
    assert store.violations[name] == 2


# ---- LOO trust ----

def test_loo_trusted_nonempty_for_recolor():
    trusted = loo_trusted(TRAIN_IN, TRAIN_OUT, "recolor")
    assert "pixelwise_recolor" in trusted


# ---- end-to-end A1: structural conflict then solve ----

async def test_a1_learns_then_solves():
    gen = StubGenerator([WRONG_1x1, CORRECT])
    cfg = _cfg(spec_inject=True, clause_learn=True, clause_inject=True)
    result = await solve_task(train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
                              generator=gen, config=cfg, arm="A1")
    log = result["clarc_log"]
    assert log["solved"] and log["iterations_to_solve"] == 2
    first = log["records"][0]
    assert first["conflict_type"] == "structural"
    assert "pixelwise_recolor" in first["violated"]
    assert log["clause_yield"] == 1.0


# ---- A2 hard pruning keeps the doomed candidate out of the best pool ----

async def test_a2_prunes_structural_violator():
    gen = StubGenerator([WRONG_1x1, CORRECT])
    cfg = _cfg(spec_inject=True, clause_learn=True, clause_inject=True, clause_prune=True)
    result = await solve_task(train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
                              generator=gen, config=cfg, arm="A2")
    log = result["clarc_log"]
    assert log["records"][0]["pruned"] is True
    assert log["records"][0]["harmful_prune"] is False   # nothing better existed yet
    assert log["solved"] and log["iterations_to_solve"] == 2


# ---- A0 stays a clean baseline (no contract activity) ----

async def test_a0_no_contract_activity():
    gen = StubGenerator([WRONG_1x1, CORRECT])
    cfg = _cfg()  # all flags off
    result = await solve_task(train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
                              generator=gen, config=cfg, arm="A0")
    log = result["clarc_log"]
    assert log["solved"]
    assert all(r["conflict_type"] is None for r in log["records"])
    assert log["n_pruned"] == 0
