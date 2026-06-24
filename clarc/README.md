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
uv run python -m clarc.probes.synth_coverage --depth 2   # DSL ceiling, no LLM (≈2/40)
uv run python -m clarc.probes.audit_refutations          # soundness tripwire (0 false refutations)
uv run python -m clarc.cli.ablate --stub --num 2 --arms A0,A1 --seeds 0   # offline solve-loop smoke
```

## Where things live

The package is split into layered subpackages with a strict acyclic dependency graph
(`common < contracts < objects < dsl < solve < {cli, probes}`):

- **`solve/` — solve loops:** `loop.py` (A/D/E arms — contract-learning + DSL⇄SMT CDCL),
  `solver.py` (G arms — guided code-gen), `harness.py` (shared runner core), `generator.py`,
  `analyze.py`.
- **`cli/` — runners:** `run.py` (single/batch), `experiment.py` (resumable failure-band),
  `ablate.py` (arms × seeds), `devset.py` (dev-set curation). Each takes `--stub`; also
  exposed as `clarc-*` console scripts.
- **`dsl/` — DSL ⇄ SMT:** `core.py`, `types.py`, `parse.py`, `obj.py`, `absdomain.py`,
  `smt.py`, `clauses.py`, `induce.py`, `prim_library.py`.
- **`objects/` — object dual (settled-negative; portfolio floor):** `base.py`, `smt.py`,
  `solve.py`, `conf.py`, `dual/`.
- **`contracts/` — contract vocabulary + learning:** `vocab.py`, `spec.py`, `store.py`,
  `learn.py`, `library.py`, `sandbox.py`.
- **`common/` — shared utilities:** `data.py` (dataset loading, no LLM import),
  `geometry.py` (segmentation), `llm_parse.py`/`codeparse.py` (block extraction),
  `types.py`, `instrument.py`, `paths.py` (filesystem anchors).
- **`probes/` — `$0` diagnostics:** `synth_coverage.py`, `probe_dsl.py`,
  `audit_refutations.py`, `cegis_audit.py`, `ce_replay.py`, `dual_oracle.py`, `vselect_eval.py`.

Arm definitions are authoritative in **`cli/run.py` (`ARMS`)** — see the arm table in
[CLAUDE.md](../CLAUDE.md#surfaces--arms).
