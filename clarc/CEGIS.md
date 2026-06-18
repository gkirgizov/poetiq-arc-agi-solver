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

## Faithfulness status (with a real LLM)
- **C1 (typed DSL) + C4 (generate-from-pruned-space): ACTIVE** on the mini-eval (8/8 valid DSL;
  24 synth feasible draws; 1 synth-solve).
- **C2 (traceable refutation) + C3 (clause lattice): not exercised** on these easy tasks — the LLM
  one-shot or duped, never producing a wrong-but-VALID candidate to refute. They ARE active in
  general (the 117 D2 logs: 62 refutations, all traceable). → the faithful test needs a band that
  **exercises refutation** (LLM proposes plausible-but-wrong DSL), not tasks solved on iter 1.

## Next experiments (monitored)
- Re-run E2 with the dead-end fix; confirm `0bb8deee`-style dups convert to progress.
- A band that exercises C2/C3 (DSL-shaped but not one-shot) so all four criteria are simultaneously
  active in one run — the actual "really tested" condition.
- Honest caveat: E2 forces DSL emission, so coverage is capped by DSL+induction vs A0's full
  Python. Broad lift hinges on induction coverage; measure synth_feasible/induction reach first.
