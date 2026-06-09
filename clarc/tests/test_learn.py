"""Phase 4 tests: adaptive contract induction (offline, sandboxed verification).

Covers the predicate sandbox, the verification gate (soundness + relevance),
library round-trip, and an end-to-end A1L solve where a new invariant is induced.
"""

import numpy as np

from clarc.generator import StubGenerator
from clarc.learn import parse_proposal, propose_contract, verify_on_pairs
from clarc.library import ContractLibrary
from clarc.loop import solve_task
from clarc.predicate_sandbox import verify_predicate
from clarc.store import LearnedStore
from clarc.types import ClarcConfig, LearnedContract

# Recolor task (1 -> 2). Identity passes the fixed spec (semantic conflict).
TRAIN_IN = [[[1, 0], [0, 1]], [[1, 1], [0, 0]]]
TRAIN_OUT = [[[2, 0], [0, 2]], [[2, 2], [0, 0]]]
TEST_IN = [[[0, 1], [1, 0]]]

PAIRS = [(np.array(gi), np.array(go)) for gi, go in zip(TRAIN_IN, TRAIN_OUT)]

# A genuine invariant of the correct outputs, violated by identity output.
PRED_GOOD = ('import numpy as np\n'
             'DESCRIPTION = "output never contains color 1"\n'
             'def holds(inp, out):\n'
             '    return not bool((np.array(out) == 1).any())')
PRED_FALSE_ON_TRAIN = ('def holds(inp, out):\n'
                       '    return bool((np.array(out) == 9).any())')   # false on real outputs
PRED_NONDISCRIMINATIVE = 'def holds(inp, out):\n    return True'        # holds on everything

IDENTITY = "def transform(grid):\n    return grid"
CORRECT = ("import numpy as np\n"
           "def transform(grid):\n"
           "    g = np.array(grid).copy(); g[g == 1] = 2; return g")


# ---- predicate sandbox ----

async def test_verify_predicate_runs_and_handles_errors():
    res = await verify_predicate(PRED_GOOD, PAIRS)
    assert res == [True, True]
    assert await verify_predicate("def holds(inp,out):\n  return 1/0", PAIRS) == [None, None]
    assert await verify_predicate("not python (((", PAIRS) is None
    assert await verify_predicate("def other(): pass", PAIRS) is None   # no `holds`


def test_parse_proposal():
    descr, code = parse_proposal(PRED_GOOD)
    assert descr == "output never contains color 1"
    assert "def holds" in code
    assert parse_proposal("def transform(x): return x") == ("", None)


# ---- the verification gate ----

async def test_gate_admits_sound_and_relevant():
    gen = StubGenerator([PRED_GOOD])
    failed = [np.array(gi) for gi in TRAIN_IN]   # identity outputs == inputs (contain 1)
    lc = await propose_contract(gen, PAIRS, failed, idx=0, seed=0)
    assert lc is not None
    assert "color 1" in lc.descr


async def test_gate_rejects_unsound():
    gen = StubGenerator([PRED_FALSE_ON_TRAIN])
    failed = [np.array(gi) for gi in TRAIN_IN]
    assert await propose_contract(gen, PAIRS, failed, idx=0, seed=0) is None


async def test_gate_rejects_nondiscriminative():
    gen = StubGenerator([PRED_NONDISCRIMINATIVE])
    failed = [np.array(gi) for gi in TRAIN_IN]
    assert await propose_contract(gen, PAIRS, failed, idx=0, seed=0) is None


async def test_verify_on_pairs_reuse_gate():
    assert await verify_on_pairs(PRED_GOOD, PAIRS) is True
    assert await verify_on_pairs(PRED_FALSE_ON_TRAIN, PAIRS) is False


# ---- library round-trip ----

def test_library_roundtrip(tmp_path):
    p = str(tmp_path / "lib.json")
    lib = ContractLibrary.load(p)
    lib.add("c0", "no color 1", PRED_GOOD)
    lib.save()
    lib2 = ContractLibrary.load(p)
    assert len(lib2.entries) == 1
    assert lib2.has(PRED_GOOD)
    lib2.record_use(PRED_GOOD, verified=True, solved=True)
    assert lib2.entries[0].seen_tasks == 1 and lib2.entries[0].solved_after == 1


def test_store_learned_dedup_and_render():
    store = LearnedStore()
    lc = LearnedContract(name="x", descr="no color 1", code=PRED_GOOD)
    assert store.add_learned(lc) is True
    assert store.add_learned(lc) is False                 # dedup by code
    assert "DISCOVERED INVARIANTS" in store.render_for_prompt()


# ---- end-to-end A1L: induce a contract, then solve ----

async def test_a1l_induces_then_solves():
    # call order: it0 transform(identity) -> semantic conflict -> propose(predicate)
    #             it1 transform(correct) -> solved
    gen = StubGenerator([IDENTITY, PRED_GOOD, CORRECT])
    cfg = ClarcConfig(max_iterations=4, seed=0, shuffle_examples=False, problem_id="recolor",
                      spec_inject=True, clause_learn=True, clause_inject=True,
                      learn_contracts=True, use_library=False)
    result = await solve_task(train_in=TRAIN_IN, train_out=TRAIN_OUT, test_in=TEST_IN,
                              generator=gen, config=cfg, arm="A1L")
    log = result["clarc_log"]
    assert log["solved"] and log["iterations_to_solve"] == 2
    assert log["n_learned"] == 1
    assert any("color 1" in d for d in log["learned"])
