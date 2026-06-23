# clarc — developer quickstart

`clarc` is the contract-learning research layer over the poetiq harness. This file is
the short dev entry; the full guide (surfaces, arms, the R&D loop, conventions) is in
the top-level **[CLAUDE.md](../CLAUDE.md)**, the public story in
**[README.md](../README.md)**, and the results in **[FINDINGS.md](../FINDINGS.md)**.

## Setup & tests

```bash
uv sync
uv run pytest -q          # offline tests; real-CLI tests are marked `cli` and skipped
```

> One test — `test_synth::test_e2_synth_solves_dsl_task_without_llm` — exercises the full
> synth+sandbox solve path and is **resource-fragile in sandboxed environments** (it can
> be SIGKILLed mid-run). It is not a code regression; run the rest with
> `uv run pytest -q --ignore=clarc/tests/test_synth.py` if your environment kills it.

## Try it without spending anything

```bash
uv run python -m clarc.synth_coverage --depth 2   # DSL ceiling, no LLM (≈2/40)
uv run python -m clarc.audit_refutations          # soundness tripwire (0 false refutations)
uv run python -m clarc.ablate --stub --num 2 --arms A0,A1 --seeds 0   # offline solve-loop smoke
```

## Where things live

- **Solve loops:** `loop.py` (A/D/E arms — contract-learning + DSL⇄SMT CDCL),
  `solver.py` (G arms — guided code-gen), `harness.py` (shared runner core used by the
  three CLIs below).
- **Runners:** `run.py` (single/batch), `experiment.py` (resumable failure-band),
  `ablate.py` (arms × seeds), `devset.py` (dev-set curation). Each takes `--stub`.
- **DSL ⇄ SMT:** `dsl.py`, `dsltypes.py`, `dslparse.py`, `dslobj.py`, `absdomain.py`,
  `smt.py`, `clauses.py`, `learn_prim.py`, `prim_library.py`.
- **Object dual (settled-negative; portfolio floor):** `objects.py`, `objsmt.py`,
  `objsolve.py`, `objconf.py`, `dual/`.
- **Contract vocabulary + learning:** `contracts.py`, `spec.py`, `store.py`, `learn.py`,
  `library.py`, `analyze.py`, `predicate_sandbox.py`.
- **Shared utilities:** `data.py` (dataset loading, no LLM import), `geometry.py`
  (segmentation), `llm_parse.py` (fenced-block extraction).
- **`$0` diagnostics:** `synth_coverage.py`, `probe_dsl.py`, `audit_refutations.py`,
  `cegis_audit.py`, `ce_replay.py`, `dual_oracle.py`, `vselect_eval.py`.

Arm definitions are authoritative in **`run.py` (`ARMS`)** — see the arm table in
[CLAUDE.md](../CLAUDE.md#surfaces--arms).
