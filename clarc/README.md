# clarc — Contract-Learning ARC solver

A **CDCL-style loop in program-generation space**, layered on the poetiq harness.
Keeps poetiq's dead-simple *generate → execute → feedback* loop, but runs it over a
**verifiable, refinement-typed description layer** of grid transformations:

- The LLM (driven via the **Claude Code CLI**, `claude -p` — no API tokens) is the
  generator / abductive proposer.
- **Python is the sound verifier.** Invariants ("contracts") are extracted from the
  train pairs and *re-verified by evaluation*; a "lesson" enters the learned store
  only if its predicate provably holds on all train pairs — so no hallucinated
  knowledge can accumulate.
- Failures become **durable, machine-checkable clauses** that (a) are injected into
  the next prompt as an authoritative spec and (b) let Python prune/deprioritize
  candidates — the CDCL "learned clause + backjump", versus poetiq's ephemeral diff.
- **The contract vocabulary grows adaptively.** On a *semantic* conflict (the known
  invariants are silent), the LLM proposes a *new* predicate `holds(inp, out)`;
  Python verifies it in a sandbox (holds on all train pairs **and** is violated by
  the failure) before admitting it, and persists it to a **cross-task library** that
  is soundly re-verified on each new task. The fixed builders are only a bootstrap,
  not a ceiling — so the system is adaptive from the start rather than a brittle
  pre-designed DSL.

The thesis bar is high: poetiq's `_build_feedback` already gives shape/palette/cell
diffs. clarc's value is in **global invariants the cell-diff doesn't state**
(`color_hist_preserved`, `object_count_preserved`, `pixelwise_recolor`,
`shape_eq_content_bbox`, gained symmetry, ...), accumulated and enforced.

## Setup

```bash
uv sync
uv run pytest -q          # 20 offline tests (real-CLI tests are marked `cli`, skipped by default)
```

## Quickstart

```bash
# Solve specific tasks through the real Claude Code CLI (A0 = poetiq baseline arm):
uv run python -m clarc.run --arm A1 --tasks 0d3d703e,3c9b0459 --data 2024-train --iters 5

# Regenerate the curated dev set (pure Python, no spend):
uv run python -m clarc.devset

# Dry-run the whole ablation pipeline offline (StubGenerator, no API spend):
uv run python -m clarc.ablate --stub --num 2 --arms A0,A1 --seeds 0,1
```

## Running the head-to-head (spends API budget)

The runner reports cost live and writes results incrementally (`output/ablation/`),
so it is safe to stop.

```bash
# Recommended pilot (~$10): mid model => conflicts actually arise.
uv run python -m clarc.ablate --model sonnet --arms A0,A1 --seeds 0,1 --num 8

# Full planned matrix (~$50):
uv run python -m clarc.ablate --model sonnet --arms A0,A5,A1 --seeds 0,1,2

# Cheapest / max-conflict stress test (~$3):
uv run python -m clarc.ablate --model haiku --arms A0,A1 --seeds 0,1 --num 8
```

> **Regime note (empirical).** A *strong* generator one-shots most dev tasks → no
> conflicts → the contract machinery never engages. Use a **weaker/faster model**
> (`--model sonnet` / `haiku`) to create the conflict regime where contracts can
> help — this is also far cheaper and is the truer "half-the-cost" test. `--model`
> accepts CLI aliases (`opus`/`sonnet`/`haiku`) or a full model name; omit it for
> the CLI default.

### Ablation arms

| Arm | Ingredients | Isolates |
|-----|-------------|----------|
| **A0** | none (poetiq-style feedback through the CLI) | baseline / confound check |
| **A5** | `spec_inject` | static verified invariants in the prompt |
| **A1** | `spec_inject + clause_learn + clause_inject` | full CDCL (soft) |
| **A2** | A1 `+ clause_prune` | value of hard pruning (LOO-trusted) |
| **A3** | `clause_learn + clause_prune` (no prompt injection) | does telling the model matter |
| **A1L** | A1 `+ learn_contracts` | **adaptive induction** of new invariants + cross-task library |

### Reading the report

Per stratum (`structural` / `logic`) and overall, per arm: `acc` (pass@2 via
seed-ensemble voting), `trainsolve`, `med_iters` (iterations-to-solve), `yield`
(fraction of conflicts that were structural — does the CDCL framing bite),
`prune`/`harm`, `genfail`, and `$cost`.

**GO** (core thesis works): A1 beats A0 by ≥+4 tasks overall *or* ≥+3 on the
structural stratum across seeds; `yield` meaningfully > 0; no harm on the logic
stratum. **NO-GO**: A1≈A0, or A5≈A1 (the apparatus adds nothing over an English
spec), or yield≈0 (ARC failures are semantic, not contractual).

## File map

| File | Responsibility |
|------|----------------|
| `generator.py` | `ClaudeCodeGenerator` (`claude -p`) + `StubGenerator`; `Generator` protocol |
| `loop.py` | the solve loop; ingredient flags select the arm; cheapest-first gates |
| `contracts.py` | contract vocabulary (attribute extractors + relation builders) + subsumption |
| `spec.py` | `extract_spec` (sound by construction) + `loo_trusted` (safe-to-prune) |
| `store.py` | `LearnedStore`: verified-only `add`, VSIDS-lite activity, earned escalation, induced contracts |
| `analyze.py` | conflict classification, grid parse, the verification gate |
| `learn.py` | open-vocabulary induction: propose `holds(inp,out)`, gate (sound + discriminative) |
| `predicate_sandbox.py` | subprocess executor for proposed predicates (no in-process exec of model code) |
| `library.py` | cross-task contract library (persist, sound re-verify, utility ranking) |
| `instrument.py` | per-iteration records + thesis-critical rollups |
| `devset.py` | deterministic stratified dev set (cached `devset_ids.json`) |
| `run.py` | single/batch task runner | 
| `ablate.py` | arms × seeds eval + report |

Reused from poetiq (unchanged, by import): `arc_agi.sandbox.run`,
`arc_agi.io.build_kaggle_two_attempts`, `arc_agi.scoring.score_task`,
`arc_agi.solve_coding` helpers (`_eval_on_train_and_test`, `_build_feedback`,
`format_problem`, ...), `arc_agi.types`.

## Adaptive contract learning (Phase 4)

Built and offline-tested (`clarc/tests/test_learn.py`); **paid real-CLI validation
deferred** (it needs the same weak-model conflict regime as the eval). Run it via
the `A1L` arm:

```bash
uv run python -m clarc.ablate --model sonnet --arms A0,A1,A1L --seeds 0,1 --num 8
```

What to watch: `n_learned` per task (induced or library-reused invariants), whether
A1L's `clause_yield` rises over A1 (semantic conflicts converted into checkable
invariants), and whether `contract_library.json` accumulates reusable contracts
that lift later tasks.

Induced predicates are enforced like fixed contracts: they participate in the
sandboxed `violations()` check, make a conflict *structural*, feed structured
feedback, and are pruned under `clause_prune` (sound — each holds on all train
pairs by the gate, so a violation proves train-wrongness). This makes the loop
converge: keep inducing until the candidate's failures are explained by learned
invariants.

## Deferred (extension points, not built)

Library compression/abstraction (Stitch/DreamCoder-style) · SMT/CrossHair contract
*entailment* (prune without evaluation) · ILP selector-invention / counterfactuals
(Popper) · mandated typed-DSL composition (ablation A4). The interfaces are designed
so these slot in without a rewrite.
