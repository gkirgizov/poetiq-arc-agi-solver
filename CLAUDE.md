# CLAUDE.md — clarc / poetiq-arc-agi-solver

Operator + agent guide. The single source of truth for the surfaces, arms, entry
points, and the R&D loop. Read this first; the public story is in [README.md](README.md),
the findings in [FINDINGS.md](FINDINGS.md), the raw lab notebooks in
[docs/notebook/](docs/notebook/).

## What this is

Two bodies of code:

- **`arc_agi/`** — the upstream **poetiq** harness: a dead-simple LLM code-gen ARC
  solver (generate `transform()` → execute on train → feedback → repeat). This is the
  public baseline (**arm A0**) and the deployed solver (`main.py`).
- **`clarc/`** — the research layer: a *CDCL-in-program-space* contract-learning solver
  that adds a **sound, verifiable** description layer (typed DSL + z3 contracts + object
  duals) on top of poetiq's loop. A lesson enters the system only after Python/z3
  re-verifies it on every train pair — no hallucinated knowledge accumulates.

The LLM is driven through the **Claude Code CLI** (`claude -p`, subscription auth — no
API keys); Python/z3 is the sound verifier.

## Surfaces & arms

| Surface | Arms | Entry | What it adds | Verdict |
|---|---|---|---|---|
| **Baseline** (poetiq code-gen) | `A0` | `main.py`, `clarc.cli.run` | the refinement loop itself | the deployed solver; reused by clarc |
| **Contract-learning CDCL** | `A5 A1 A2 A3 A1L` | `clarc.cli.run` / `cli.experiment` / `cli.ablate` | verified invariants → prompt + pruning + induction | sound; faster convergence at 0 harm; solve-rate lift directional only |
| **DSL ⇄ SMT ⇄ synth** | `D0 D1 D2 · E0 E1 E2 E3 · DS` | `clarc.cli.run` / `cli.experiment` + `$0` probes | typed pipelines, pre-exec refutation, LLM-free synthesis | the **positive**: `symmetry_repair`/`connect_dots` solved LLM-free; faithful CEGIS, 99.4% sound refutation |
| **Object dual + guided code-gen** | `G0 G1 G2 G3 G4 G5` | `clarc.cli.experiment` → `solve.solver.guided_solve` | object-correspondence contracts as a prompt guide + counterexamples | settled **negative** (CE doesn't guide); kept as portfolio floor + verified-selection (G5) |

**Arm definitions are authoritative in [`clarc/cli/run.py`](clarc/cli/run.py) (`ARMS`).** Summary:

- **A-arms** (contract-learning): `A0` baseline · `A5` `spec_inject` · `A1`
  `spec_inject+clause_learn+clause_inject` (soft CDCL) · `A2` `A1+clause_prune` ·
  `A3` `clause_learn+clause_prune` (no prompt inject) · `A1L` `A1+learn_contracts`
  (open-vocabulary induction).
- **D/E-arms** (DSL⇄SMT): `D0` DSL try-and-drop · `D1` `+z3_refute` · `D2` `+z3_learn` ·
  `E0` `+induce_prims` (self-extension) · `E1` `+dynamic_objects` · `E2` `+synth_seed`
  (closes CEGIS criterion 4) · `DS` synth without induction · `E3` `E2+induce_holdout`
  (F4 generalization gate).
- **G-arms** (guided code-gen, via `clarc.solve.solver.guided_solve`): `G0` = A0 control ·
  `G1` object dual · `G2` `+σ dual` · `G3` confidence-gated + strong CE · `G4` `G3+σ` ·
  `G5` verified-selection (submit the train-passer whose TEST output respects
  LOO-trusted invariants).

> **Legacy-label warning:** in the older lab notebooks, "E0" sometimes names *output
> directories* from earlier free-form-induction experiments, and `CEGIS.md`'s headline
> arm is `E2`. The table above (= `cli/run.py:ARMS`) is the current source of truth; ignore
> conflicting arm names in `docs/notebook/`.

## Entry points

**Canonical runners** — invoke either as `uv run python -m clarc.cli.<x>` or via the
`[project.scripts]` console entry points `uv run clarc-<x>` (e.g. `clarc-experiment`):

| Module | What | Example |
|---|---|---|
| `clarc.cli.run` | single/batch solve, score, report (non-resumable) | `… clarc.cli.run --arm A0 --tasks 0d3d703e --data 2024-eval` |
| `clarc.cli.experiment` | failure-band Phase A→B, **resumable** (JSONL checkpoint); dispatches A/D/E→`solve_task`, G→`guided_solve` | `… clarc.cli.experiment --model claude-sonnet-4-5 --max-thinking 4000 --devset --arms A0,A1,A1L --iters 6` |
| `clarc.cli.ablate` | arms × seeds over the devset, pass@2 seed-ensemble, go/no-go report | `… clarc.cli.ablate --model sonnet --arms A0,A5,A1 --seeds 0,1,2` |
| `clarc.cli.devset` | regenerate the stratified dev set (`devset_ids.json`) | `… clarc.cli.devset` |

All three runners share `clarc/solve/harness.py` (`make_generator` + `solve_one`); each
accepts `--stub` for an offline, no-spend smoke run.

**`$0` diagnostics** (no LLM, reliable — the primary instrument here):

| Module | What |
|---|---|
| `clarc.probes.synth_coverage` | direct DSL ceiling via skeleton synth + `param_search` (resumable; reproduces the 2/40 floor) |
| `clarc.probes.probe_dsl` | refutation power / coverage / soundness / latency over the devset |
| `clarc.probes.audit_refutations` | **post-run soundness tripwire** — replay a DSL run's refutations (`--out <dir>`); must show 0 false |
| `clarc.probes.cegis_audit` | post-run faithfulness check (the four CEGIS criteria) over `runs.jsonl` |
| `clarc.probes.ce_replay` | (dual) CE-opportunity diagnostic on logged wrong candidates |
| `clarc.probes.dual_oracle` | (dual) object-contract generalization oracle |
| `clarc.probes.vselect_eval` | (dual) verified-selection (G5) measurement |

## The R&D loop — how to test a new hypothesis

1. **Hypothesis** — a new prim family, a new arm, or a lever.
2. **$0 probe first** — `synth_coverage` / `probe_dsl` / `ce_replay` / `dual_oracle`.
   The $0 z3 probes are the reliable instrument; they carry every durable result.
3. **Implement** — a DSL prim is numpy `apply` ⊕ z3 `encode` ⊕ `_reg` ⊕ a test in
   `clarc/tests/`; or an arm is a flag combination in `clarc/cli/run.py:ARMS`.
4. **Soundness gate** — `clarc.probes.probe_dsl` ($0 refutation power + 0 false refutations) up
   front; after a paid D/E run, `clarc.probes.audit_refutations --out <dir>` replays it as the
   post-run tripwire.
5. **Gated paid eval** — `clarc.cli.experiment` (resumable). Pin
   `--model claude-sonnet-4-5 --max-thinking 4000`; gate spend behind explicit approval.
6. **Faithfulness + report** — `clarc.probes.cegis_audit`, `clarc.cli.experiment --report-only`.
7. **Record the finding** — in `FINDINGS.md` (headline) and `docs/notebook/` (detail).

## Frontiers (live edges — see README)

Generalizable relational/program induction for the structural wall · the
**agent-extends-DSL loop** (LLM composes the enriched DSL + induces task-relevant prims
until train is expressed — the core unfinished goal) · per-object SELECTION sublanguage ·
multi-seed evals to clear the noise floor.

## Conventions & gotchas

- **No bare `except Exception`** — catch the specific types you expect. The only broad
  catches kept are documented isolation boundaries (subprocess sandboxes running
  untrusted model code; per-task batch isolation).
- **Pin `claude-sonnet-4-5`** for bounded thinking (`--max-thinking 4000`); the `sonnet`
  alias resolves to an adaptive model that ignores the budget and hangs on hard tasks.
- **Never broad-kill** (`pkill -f claude/node/python`) — it hits other sessions. Stop a
  tracked task or `kill` a confirmed unique PID.
- **`$0` probes are the reliable instrument.** The live LLM/sandbox `solve_task` path is
  resource-fragile in this environment (it can be SIGKILLed mid-run); the env-killed
  `test_synth::test_e2_synth_solves_dsl_task_without_llm` is pre-existing, not a code bug.
- **`output/` is gitignored** — per-cell results live there; the numbers are committed in
  prose under `docs/notebook/`. The learned libraries (`contract_library.json`,
  `prim_library.json`) are gitignored generated state, re-verified on reuse.
- **Tests:** `uv run pytest -q` (offline; `-m cli` tests skipped by default).

## Layout

`clarc/` is split into layered subpackages — the dependency graph is a strict DAG
(`common < contracts < objects < dsl < solve < {cli, probes}`; lower tiers never
import higher):

- `clarc/common/` — leaf utilities, no clarc deps: `types.py`, `data.py`,
  `geometry.py`, `llm_parse.py`, `codeparse.py`, `instrument.py`, `paths.py`
  (canonical filesystem anchors for data-file loaders).
- `clarc/contracts/` — predicate vocabulary + contract learning: `vocab.py`
  (`Contract`/`Pair`/builders/`bg`), `spec.py`, `store.py`, `library.py`, `learn.py`,
  `sandbox.py` (predicate sandbox).
- `clarc/objects/` — object dual: `base.py` (`Object`/`segment`/`SEGMENTERS`),
  `smt.py`, `solve.py`, `conf.py`, and `dual/` (`base.py`/`object_dual.py`/`sigma_dual.py`).
- `clarc/dsl/` — typed DSL ⇄ SMT ⇄ synth + primitive induction: `types.py`,
  `absdomain.py`, `core.py` (the `REGISTRY`/`Primitive`/`run_pipeline` core), `obj.py`,
  `parse.py`, `smt.py`, `clauses.py`, `induce.py` (induces DSL prims), `prim_library.py`.
- `clarc/solve/` — core solving: `loop.py` (A/D/E CDCL), `solver.py` (G guided),
  `harness.py` (shared runner core), `generator.py`, `analyze.py`.
- `clarc/cli/` — runnable drivers: `run.py`, `experiment.py`, `ablate.py`, `devset.py`
  (also exposed as `clarc-*` console scripts via `[project.scripts]`).
- `clarc/probes/` — `$0` diagnostics: `probe_dsl.py`, `synth_coverage.py`,
  `audit_refutations.py`, `cegis_audit.py`, `ce_replay.py`, `dual_oracle.py`, `vselect_eval.py`.
- `arc_agi/` upstream poetiq harness (reused by import; treated as a stable dependency).
- `docs/notebook/` lab notebooks · `docs/assets/` figures · `knowledge/` the original
  research sketch.
