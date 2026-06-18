# Faithful CEGIS test — build log & findings

Goal: test the real hypothesis — does SMT-verified, type-guided synthesis (LLM proposing in a
typed DSL, SMT pruning/learning, synth generating from the pruned space, induction growing
coverage) beat the A0 baseline? The prior G-arm effort satisfied **none** of the four
CEGIS / type-directed-synthesis criteria; the DSL⇄SMT arms do. Plan: `~/.claude/plans/…fox.md`.

## Four-criteria audit (`clarc/cegis_audit.py`)
On 117 existing D/E logs: **C1** typed-DSL candidates 100% · **C2** traceable feedback (unsat core
names `slot:<prim>`+`fact:<group>`) 100% · **C3** monotone clause lattice active · **C4** 562K–1.1M
skeletons blocked **but synth-seeded = 0 (DORMANT)**. So C1–C3 were real; C4 only partial.

## Stages landed (branch dsl)
- **S1 wire SYNTH (arm E2):** each iter, `smt.synth_models(blocked_*=clauses)` enumerates
  spec-feasible, clause-pruned skeletons; a concrete fit is a VERIFIED solve (no LLM). **C4 now
  ACTIVE.** Verified: E2 solves `flip_h` with a verified program, $0 LLM.
- **S2 anti-triviality (F3):** reject an induced prim that never changes the abstract state on any
  sample (a true no-op); sound because a real transform moves σ (recolor→histogram, crop→dims), so
  the σ-blind contract-triviality test that would wrongly reject recolors is avoided.

## Mini-eval (real LLM, haiku, 8 tasks — `output/cegis-mini`) — two decisive findings
1. **The full neurosymbolic loop closes.** `009d5c81` (a *structural* task, "0/20 DSL coverage")
   was solved by **SYNTH composing an INDUCED primitive** `ind_009d5c81_0()` — induction grew the
   DSL, synth found the verified solution in the extended pruned space, **no LLM-DSL emission**.
   The coverage lever (Stage 3) works in principle. `195ba7dc`,`31d5ba1a` solved by LLM-DSL.
2. **Failure mode — synth-seeding lured the LLM into dead ends.** `0bb8deee`: the LLM proposed
   `ind_0bb8deee_0(); crop_bbox()` **5× as `dup`** because synth had pre-tried those
   feasible-but-concretely-wrong pipelines, added them to `seen`, AND seeded them as *"adapt these"*
   — so the LLM re-proposed already-tried dead ends with **no feedback** on why they failed.
   **Fix (loop.py):** a synth pipe that's feasible-but-wrong is **negative evidence** — feed its
   train-diff into the LLM memory ("ALREADY TRIED … do something DIFFERENT"), don't advertise it.

## FAITHFUL TEST COMPLETE (mini-eval, 8 tasks, haiku, `output/cegis-mini`)
**All four criteria simultaneously ACTIVE** with a real LLM — the "really tested" condition is met:
C1 typed-DSL 83% (20/24) · C2 traceable refutation **5/5 (100%)** · C3 clause lattice (≤4/task) ·
C4 196,520 skeletons blocked, 44 synth draws, **2 synth-solves**. Unlike the G-arm strawman, this
is a genuine CEGIS test.

**Result (honest, negative for BROAD lift):** A0 **7/8** · E2 **3/8** · portfolio **7/8**. E2 ties
A0 only on DSL-expressible tasks (195ba7dc, 31d5ba1a) and synth-solved one structural task via an
induced prim (009d5c81); elsewhere the **DSL-emission constraint caps coverage below A0's full
Python**, exactly as predicted. E2 adds no portfolio lift on this band.

**New failure mode — F4 at the induction level (overfit):** `0a2355a6` — E2 train-solved (acc=0 on
test) via a synth-composed induced prim whose rule is baroque and coincidental: *"group two
smallest objects if sizes differ by ≤5 cells, else separate; sequential colors 3,2,4,5…"*. The
contract is fresh-batch-verified (sound) but the prim's BEHAVIOR is overfit to 2–4 examples. The
Stage-2 guard catches no-ops, NOT over-specific coincidental prims. (Not a regression: A0 also
fails 0a2355a6 — but it is a train-verified-but-wrong submission, so it matters for the portfolio.)

## Verdict on the hypothesis
- **Faithfulness: YES** — all four criteria active; the hypothesis is, for the first time, actually
  tested. The machinery (typed DSL, sound 99.4% refutation, monotone clause lattice, synth from the
  pruned space, induction) is real and works (009d5c81 closed the full loop end-to-end).
- **Broad lift: NO (on this band)** — E2 < A0 because forcing DSL emission caps expressiveness; the
  binding constraint is DSL/induction coverage, and induction itself overfits on hard tasks (F4).

## Next experiments (monitored)
- Validate the dead-end fix (`output/cegis-fix`): do `0bb8deee`-style dups convert; any regression?
- Coverage frontier: can induction reliably + GENERALIZABLY extend the DSL (mitigate the 0a2355a6
  F4 overfit — e.g. an LOO/simplicity gate on induced-prim BEHAVIOR, not just its contract)?
- Quantify E2's secondary edge where it ties A0: verified (no overfit) + fewer iters / lower cost.
