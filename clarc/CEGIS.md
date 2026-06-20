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

## $0 coverage probe (`synth_coverage.py`, z3 + concrete, NO LLM — reliable)
Pure-DSL/synth coverage = **0/40** devset (depth 4, top-16 skeletons; structural 0/20, logic
0/20). Shape: **40/40 feasible-but-unsolved, 0 infeasible** — the σ-abstraction is too coarse to
declare ANY task infeasible, so synth enumerates abstractly-feasible-but-concretely-wrong
skeletons and solves none. (True DSL ceiling is ~2/40 per HANDOFF depth-2; top-16 missed those —
either way negligible.) ⇒ **the synthesizer adds ~0 solving power; every E2 solve is the LLM's**
(DSL emission on the 2 expressible tasks + induced prims). The CEGIS machinery is faithful but
does not lift solving on this DSL; the LLM is the engine, and DSL emission caps it below A0's
Python. Corollary: `synth_feasible` is NOT a usable "DSL-can't-express ⇒ induce" trigger (it's
always feasible) — the abstraction is too weak to prune.

Infra note (reliability): the paid haiku evals stalled at **0% CPU** — transient network
degradation hanging an import/CLI subprocess, NOT the code. The $0 probes (z3 + concrete, no LLM)
are the reliable instrument; prefer them. When a paid eval is needed, actively poll
(pgrep + cell count + %CPU-hang detection) and relaunch as TRACKED tasks — never detached `&`.

## LEVER FOUND: the 0/40 was a synth PARAM-WITNESS limit, not a DSL limit
Critically interrogating the 0/40: E2 solved `195ba7dc`/`31d5ba1a` via **single depth-1 prims**
(`split_binop_h(or,1)`, `split_binop_v(xor,6)`) — the DSL expresses them. synth_models even
proposes the right SKELETON, but z3 returns **one arbitrary param witness** (`split_binop_h(and,1)`)
because the σ-abstraction is too coarse to pin params (the same "40/40 feasible, 0 infeasible"
coarseness). A trivial **concrete param-search** over the skeleton's tiny param space (≤40 combos)
recovers `(or,1)`/`(xor,6)` and solves both — confirmed directly + unit-tested.
→ `dsl.param_search` (skip skeletons whose param space > cap, e.g. recolor's 10^10 — the
abstraction pins those via the histogram). Wired into the live E2/DS synth path
(`loop.py`) AND `synth_coverage.py`: synth now param-searches each skeleton concretely before
trusting z3's witness, so E2 can solve depth-1 tasks via pure synth (no LLM). This is the first
mechanism that adds solving power the LLM-free symbolic loop previously lacked.

**MEASURED LIFT (`synth_coverage.py` with param-search, resumable, depth 2 / models 6, NO LLM):
pure-synth coverage 0/40 → 2/40, BOTH test-generalizing** — `195ba7dc` (`split_binop_h(or,1)`) and
`31d5ba1a` (`split_binop_v(xor,6)`) are now solved AND generalized by the LLM-free symbolic loop,
up from 0. (Structural stratum stays ~0 — those need INDUCTION, not just param-search; param-search
fixes the depth-≤2 small-param tasks the abstraction couldn't pin.) Modest in absolute terms but the
first NON-ZERO pure-symbolic result, and a clean sound win (verified + generalizes). depth-2/models-6
is a lower bound; the structural ceiling is the induction frontier.

(Aside — pre-existing bug surfaced: `absdomain.sigma_of` can IndexError on a degenerate
empty-object grid during induction's random sampling; flaky, orthogonal to param-search; fix later.)

## FRONTIER: generalizable induction — the holdout gate (arm E3)
The structural ceiling needs induction, which OVERFITS (0a2355a6: a baroque coincidental prim).
Lever (commit b34b9da): induce the prim from **n-1** train pairs (hold out 1); the loop's existing
all-train verification then rejects a prim that fits the n-1 but not the held-out. **Measured
(E2 vs E3, haiku, 4 structural tasks):** OVERFIT (solved & test-wrong) **E2=1 → E3=0** — E3 rejected
0a2355a6's overfit prim (E2 submitted it: solved/acc=0; E3 solved=False); TEST-CORRECT unchanged
(0→0). So E3 is a **sound reliability win** (no overfit submissions → cleaner train-verified
portfolio) but does **not lift coverage**: it rejects the bad prim rather than finding a good one,
because these tasks aren't generalizably induction-solvable — the LLM induces coincidental rules
from 3–4 examples (the underdetermination wall, now *sound-gated*). Next idea (TESTED, below): turn the gate from
REJECT into SEARCH — or extend the winning param-search to structural recolor.

### Param-search-structural lever — DECISIVELY NO HEADROOM ($0 end-to-end probe)
The winning logic lever was param-search over a DSL prim's tiny param space. Natural extension:
add typed attribute-recolor prims (`recolor_by_rank`/`by_attribute`) and let param-search find the
map, avoiding overfit-prone induction. **Measured before building it** — for each of the 20
structural tasks, learn `attribute→out_color` from train and APPLY to the test input, checking the
real solution: **0/20 solve+test-correct** under rank_asc, rank_desc, OR raw_size. The trap: 6/20
look "consistent by raw size" on train but that is **MEMORIZATION** — test grids have different
sizes, so **0/20 of those actually generalize** (F4 overfit at the DSL level). Generalizing-attribute
(rank) maps also solve 0/20. ⇒ the structural rules are NOT simple attribute recolors; they are
multi-attribute/conditional/spatial — exactly what LLM induction attempts and **overfits** (0a2355a6:
rank-AND-spread, rank→color inconsistent across examples: ex1 ranks→1,1,3,2 vs ex3 ranks→1,3,2,4).
So `recolor_by_*` prims would solve 0/20 — the lever is **futile on this stratum**, and the probe
SAVED building it. The structural ceiling is the fundamental ARC difficulty (a complex generalizing
rule from 3–4 examples), not a fixable mechanism gap.

## Verdict on the hypothesis
- **Faithfulness: YES** — all four criteria active; the hypothesis is, for the first time, actually
  tested. The machinery (typed DSL, sound 99.4% refutation, monotone clause lattice, synth from the
  pruned space, induction) is real and works (009d5c81 closed the full loop end-to-end).
- **Broad lift: NO (on this band)** — E2 < A0 because forcing DSL emission caps expressiveness; the
  binding constraint is DSL/induction coverage, and induction itself overfits on hard tasks (F4).
- **First sound POSITIVE: the param-search lever** — the pure-synth 0/40 was not a wall but a
  fixable synth limitation (z3's arbitrary param witness); param-search lifts it to 2/40, both
  test-generalizing, entirely LLM-free. A small but clean, sound, verified gain.

## Conclusion (this thread)
What was learned, end to end: the prior G-arm effort never tested CEGIS (fails all 4 criteria); the
DSL/SMT machinery does (and is sound, 99.4% refutation, 0 false); wiring the dormant synth closed
criterion 4; and critically re-examining the "synth adds nothing" 0/40 turned it into the
param-search lever (0→2/40, generalizing). The remaining frontiers are (a) deeper synth — marginal
AND impractical in this env (a depth-3 probe managed only ~4/40 per idle-kill window, ~90s/task;
and the 2 confirmed wins are DEPTH-1 single-prim solutions — param-search's sweet spot — so deeper
multi-step DSL solutions are unlikely to add much, ARC rules rarely reduce to ≤3 generic DSL steps),
so the **depth-2 measure of 2/40 stands as the reliable lift**; and (b) the STRUCTURAL ceiling, which
needs generalizable INDUCTION — the hard open problem, where induction overfits (the 0a2355a6 F4 prim).

Honest infra note: in this environment the LLM/sandbox `solve_task` path HANGS (live E2/DS
verification on 195ba7dc produced no output despite exit 0; paid haiku evals stall at 0% CPU). The
reliable instruments are the **$0 z3 probes** with resumable file checkpoints — they carried every
real result here. Live/paid verification is infra-blocked, so the offline probe (same wired
`param_search`, same tasks) is the proof of record. Per "stop rather than churn", the thread
concludes on the param-search lever; the induction frontier is the next session's work.

### Portfolio-reliability framing — ALSO no headroom (the competence OVERLAPS A0, doesn't complement)
The CEGIS arm's real value proposition isn't raw coverage but VERIFICATION (when it solves, it's
certain). Portfolio test: does the verified (LLM-free) arm RECOVER any task A0 gets wrong? **No.**
A0 solves BOTH param-search verified solves (195ba7dc, 31d5ba1a) at acc=1.0; on cegis-mini E2's
solves ⊆ A0's (E2 3/8 ⊆ A0 7/8). The symbolic arm is strong exactly where A0 is strong (easy/
DSL-expressible/depth-1) and weak exactly where A0 is weak (structural) — its competence OVERLAPS
A0 rather than COMPLEMENTING it, so there is no portfolio recovery. A portfolio win would need a
symbolic solve of a task A0 FAILS, which requires cracking the structural stratum — the same
fundamental wall. ⇒ Every tractable mechanism lever is now measured-exhausted: param-search wins
LOGIC (2/40, but ⊆ A0); E3 holdout gate makes induction SOUND (no overfit) but adds no coverage;
structural attribute-recolor is futile (0/20); portfolio-recovery is empty (symbolic ⊆ A0). The
honest deliverable: the param-search lever is a sound, self-contained positive (LLM-free symbolic
solving where the abstraction previously failed to pin params); broad lift over A0 needs the
structural stratum, which is the fundamental ARC difficulty, not a fixable mechanism gap.

## DSL DEVELOPMENT phase (user directive: grow the DSL; agents work inside it, extend until it expresses train)
Method that WORKS: data-driven — implement a candidate operation as a throwaway fn, test it DIRECTLY
(apply vs train+test) across the full ARC-1 eval, keep families with NET-NEW yield (not already
covered — checked via synth), add winners to dsl.py (apply + SOUND encode + _reg + test).
- **Net-new WINS:** `symmetry_repair(noise)` solves **8** ARC-1 tasks (mirror symmetry consistent on
  non-occluded cells + translational periods), `connect_dots()` solves 070dd51e — ALL test-
  generalizing, ALL found via synth_models+param_search **LLM-FREE** (validated end-to-end). Growing
  the DSL directly lifts the sound symbolic coverage (was 2/40) — the user's "agents inside the DSL".
- **Saturation:** ~10 other single-prim families (fractal, quad-mirror, crop-to-color, outline,
  denoise, keep/remove-object, odd-one-out, object-gravity, panel_overlay) yield 0–1 net-new. Single
  GLOBAL prims saturate after the symmetry jackpot; the remaining tippable tasks need COMPOSITION or
  PER-OBJECT application (move bucket) or complex attribute-recolor (structural — 0/20 simple maps).
- **Perf:** vectorized `translate` apply (numpy slicing) 0.5s→0.002s (150x; it dominated param_search).
- **Architecture insight (important):** the new general prims carry permissive dims-only encodes —
  SOUND (no false-refute) but z3-synth canNOT prune them (the σ-abstraction is too coarse), so
  synth-composition over the rich DSL is unguided/slow. ⇒ the **LLM is the natural composer** for the
  rich DSL (semantic reasoning), with synth+param_search as the LLM-free fast path for single prims /
  small-param logic. NEXT PHASE: (a) per-object SELECTION sublanguage (enable object-level
  composition for move/object tasks); (b) the **agent-extends-DSL loop** — LLM composes the enriched
  DSL and induces task-relevant prims until train is expressed (the user's core goal; gate paid).

## DSL-development saturation + induction-reach findings (honest bounds)
After ~20 data-driven candidate families tested vs the full ARC-1 eval, only TWO were net-new:
symmetry_repair (+8) and connect_dots (+1). The rest yield 0 or are REDUNDANT with existing prims
(panel_overlay≡split_binop on 34b99a2b; scale_to≡scale(2,2) on 60c09cac). ⇒ single GENERAL-prim
mining is SATURATED — the clean-algorithm families ARC rewards (symmetry/periodicity, connect,
logic/overlay, gravity, scale, crop, fill) are largely covered.
- **DSL vs A0 (symmetry family, n=6 before infra-kill):** A0 (free-form sonnet) solves 5/6; the
  enriched DSL solves all LLM-free+verified and recovers A0's one miss (929ab4e9). So the DSL's edge
  over the baseline is HARD INSTANCES of clean-algorithm families (where general algo > ad-hoc LLM
  code), NOT broad coverage — A0 already codes the easy clean-algorithm tasks.
- **Induction REACH:** the per-object induction (E1 recolor/move/drop over whole objects) does NOT
  cover the idiosyncratic tail — 3dc255db's objects SPLIT/GROW/transform (size-13 → size-2+size-11,
  new objects), i.e. RESTRUCTURING outside the per-object basis. And the complex-recolor stratum is
  underdetermined (0/20 simple attribute→colour maps). So neither general-prims nor per-object
  induction cracks the tail; it is the genuine ARC difficulty (idiosyncratic restructuring /
  underdetermined rules).
- **Net:** the WORKING lever is clean-algorithm DSL development (symmetry_repair = the win), now
  mined. The one unexplored lever toward beating A0 on the tail: the LLM COMPOSING the enriched DSL
  (multi-step pipelines + task-specific induced steps) — needs gated paid testing (infra keeps
  killing bg tasks; use small resumable batches).

## Composition headroom — also empty (3rd independent angle)
2-prim composition scan over the enriched DSL (curated 20 grid prims, param-searched) on an 18-task
focused band (tippable + symmetry): **0 net-new** — no 2-prim pipeline solves a band task that no
single prim solves. So the tippable tail is outside the DSL's reach from THREE independent angles:
single-prim (saturated), 2-prim composition (0), per-object induction (can't restructure). The tail
is genuinely idiosyncratic (the ARC difficulty); expressing it needs near-free-form restructuring,
which conflicts with the "inside the DSL, not free-form python" constraint. DURABLE WINS of the
DSL-development phase: symmetry_repair (+8, LLM-free via synth, recovers A0's hard instances like
929ab4e9) + connect_dots (+1), both generalizing + sound. That is the lever; it is now mined.
