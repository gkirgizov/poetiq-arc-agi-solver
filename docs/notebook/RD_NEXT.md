# clarc/dual — Next R&D iteration: make the counterexample channel actually GUIDE

> **Archived lab notebook (provenance).** Point-in-time research log; some referenced
> paths/scripts (e.g. `conditional_probe.py`, `audit_ce_actionability.py`) have since
> been removed. Curated findings: [`/FINDINGS.md`](../../FINDINGS.md); current guide:
> [`/CLAUDE.md`](../../CLAUDE.md).

## Where we are (2026-06-16)

Full-devset paid eval (`output/exp-guided-gate`, haiku, 120 cells), pass@1:
`struct G0=15 G1=17 G3=16 (portfolio 17) · logic 19/19/19 · all G0=34 G1=36 G3=35`.
Null / noise-dominated; the confidence gate (G3) was net-neutral-to-negative and the
strong-CE channel fired **0** times (G3) / **1** (G1) across 40 tasks.

### The decisive diagnosis (why CE doesn't guide) — `clarc/ce_replay.py`, $0
- A0 already solves **34/40** → only **6** failing tasks, and of those only ~3 induce a
  contract at all → essentially **no dual-relevant headroom** on this set.
- Replaying every logged WRONG candidate against the **full verifier vocabulary**
  (object-correspondence **and** grid-level σ): a CE would fire on **1 of 47** wrong
  iterations (**2%**). The wrong candidates **satisfy** `shape/object_count/nonbg_count/
  palette/recolor_only/bg_preserved` — they do a valid recolor and **pick the wrong
  colors**. Our verifiers check the **envelope** of the transformation; the errors are in
  the **content** (which value, conditional on which attribute), which the envelope is
  blind to.

**Conclusion:** for a competent generator, failures are *content* errors. The
object/σ duals are *envelope* verifiers — they prune wildly-wrong candidates (wrong dims,
wrong count) but cannot refute a structurally-valid near-miss. So "broaden σ + harder
tasks" (the naive N1/N2) is the wrong fix: more envelope invariants still fire ~2%, and
"harder" only helps if the failures are *structural*. The real levers are below.

## Lever A — Content-pinning (partial conditional) contracts  [primary; $0 to test first]

Induce the **determined part of the conditional rule** — attribute→value constraints that
hold on every train pair — and refute candidates that violate that subset, even when the
full rule isn't captured (so it guides without solving). Examples the menu can't express
today but are inducible:
- `color = f(size | shape | rank)` lookup that holds on train (e.g. "the largest object is
  always recolored to 4", "color 3 → 4 only when the object touches the border").
- output colors ⊆ a train-determined set keyed by an attribute; per-object color **classes**.

A candidate that recolors the largest object wrong is then **refutable** — the CE the
envelope verifiers can't produce. This is the M7c lever (`objsolve.py:12` names it as the
open gap), used here for *refutation* (partial) not full *solve*.

**$0 test RESULT (`clarc/conditional_probe.py`, 2026-06-16) — Lever A is mechanically
viable but fundamentally untrustworthy:**
- Conditional `color = f(size|rank|shape|border)` contracts refute **14/18 = 77%** of the
  wrong iterations the envelope verifiers missed (vs **2%**). So the CE channel CAN be made
  to fire on content errors. ✓
- **But none generalize** (`test_gen=False` on all of `08573cc6`/`0a2355a6`/`1acc24af`), and
  — the killer — they **all survive leave-one-out** (LOO-safe = 77%, identical to raw). A
  rule like "size→color" fits every train pair, predicts every held-out train pair, and is
  *still wrong on test*. At n=3–4 the attributes are correlated, so a coincidence is
  indistinguishable from the rule, and **LOO (the generalization gate) cannot filter it**.

**Conclusion:** refuting with these conditionals would push the generator toward a
train-overfit — the exact over-constraining we set out to avoid, now at the content level,
and *not* fixable by the confidence gate. Lever A makes the CE *fire* but not *trust*.

## Lever B — A structurally-hard, dual-relevant eval band  [needs a paid sweep]

Find tasks where the generator fails on **structure** (the CE's wheelhouse), not content.
- Sweep A0 over a large pool (ARC-1 eval 400 + ARC-2) with a fails-but-returns model
  (bounded-sonnet, §3a recipe; haiku on ARC-2 is too hard — 0/22). Keep tasks where A0
  fails AND `ce_replay` shows **high CE-opportunity** (wrong candidates DO violate an
  induced invariant). Target ~30–50.
- **Multi-sample 3–5 seeds/cell** to clear the ~2-task noise floor that made this run
  unreadable.
- This is the set on which the *existing* dual + strong CE could show iteration-reduction;
  `audit_ce_actionability.py` then measures repair-rate directly.

## Lever C — De-emphasize the gate

The mode-2 gate (G3) was net-negative and silenced CE. Keep only its cheap, safe parts
(drop vacuous params; soft-language phrasing) and drop the hard→soft demotion. The CE
channel, not the prompt gate, is where the remaining signal is.

## The unified thesis (what all the $0 probes converge on)

There is **no free lunch in the verifier**:
- **Envelope** contracts (shape/count/palette/object-preservation) *generalize* — but a
  competent generator's wrong candidates satisfy them, so the CE is **inert** (2% fire).
- **Content** contracts (conditional `color=f(attr)`) *see* the errors — CE fires (77%) —
  but at n=3–4 they're **coincidental and LOO can't tell**, so refuting **overfits**.

The contracts specific enough to make the CE guide are exactly the ones the examples can't
justify. This is ARC's underdetermination showing up inside the dual: it is the *same* wall
the generator faces, not an independent lever the dual can pull. The neurosymbolic CE loop
helps only where (a) the generator makes *structural* errors AND (b) few-shot *determines*
the structure — and for a 34/40 generator that intersection is nearly empty on this set.

## Where a positive result could still live (ranked, all gated on a $0 probe first)
1. **Lever B — structurally-hard band.** Tasks whose failures are envelope errors (wrong
   dims/count/tiling), where the *generalizing* envelope CE *does* fire. Needs a paid sweep
   + `ce_replay` to find them, multi-sample to beat noise. This is the honest remaining bet.
2. **Agreement-ensembled content CE.** Fire a content CE only where *many* independent keys
   agree AND it forward-solves train — i.e. only when the dual essentially *solved* it
   (route to `objsolve`, no LLM). Narrow, but sound. (`08573cc6` shows 7 keys agreeing still
   didn't generalize, so set the bar at forward-solve, not mere agreement.)
3. **Weaker generator / harder benchmark** (bounded-sonnet on ARC-2) to widen the
   structural-failure band — the only regime with real headroom.

## Lever-B RESULT (2026-06-16, `output/leverb-arc2` + `output/leverb-eval`) — fires, doesn't guide

Ran A0/haiku over 36 grid-capped ARC-2 eval tasks (A0 solved **0/36** — max headroom).
`ce_replay`: a **sound, generalizing** envelope CE (`nonbg_count_preserved`,
`color_hist_preserved`, `bg_preserved`, object-correspondence) would fire on **33/80 = 41%**
of wrong iterations — vs 2% on the easy devset. So the CE-firing problem is a *test-set*
problem: on structurally-hard tasks the generator DOES make refutable envelope errors.

Pinned the 17 high-opportunity tasks and ran **G0 / G1 / G3, iters 6** (haiku):

| arm | test-correct | train-solved | med iters | CE fired | CE actionability | score after CE |
|---|---|---|---|---|---|---|
| G0 (=A0) | **2**/17 | 3 | 2.0 | 0 | — | — |
| G1 (dual, vague CE) | **2**/17 | 3 | 2.0 | **33** | — | +14 / −10 |
| G3 (gated, strong CE) | **1**/17 | 3 | 4.0 | 15 | **1/24 ≈ 4%** | +4 / −7 |

**Verdict: the CE fires abundantly (33×) but does NOT guide.** G1 (33 CEs) solves the *same*
2 tasks as G0 (0 CEs), same median iterations; G3 is strictly worse (1 solve, 4.0 iters, and
its CE worsens the next candidate's score more often than it helps). The flagged train
example is repaired ~4% of the time. Knowing "your non-background count / colour histogram is
wrong" tells the generator it failed on a dimension it already knows it's failing — it does
**not** convey the *content* fix. A sound envelope counterexample is not an actionable one.

## CONCLUSION — both levers exhausted; a clean negative result

The SMT/invariant counterexample channel does not improve a code-gen ARC solver in any tested
regime, and the reasons are now fully characterized:
- **Strong generator / easy tasks:** failures are *content* errors; envelope verifiers are
  blind to them → CE inert (2% fire).
- **Weak generator / hard tasks:** failures *are* envelope errors → CE fires soundly (41%),
  but the envelope fact isn't *actionable* → no solve/iteration lift, ~4% repair, and the
  gated strong-CE arm (G3) is net-harmful.
- **Content CEs** (the only actionable kind) are underdetermined by 3–4 examples — they fire
  (77%) but are coincidental and survive LOO, so refuting overfits.

There is no regime where the dual's counterexamples both *fire* and *guide*: the actionable
contracts aren't justified by the examples, and the justified contracts aren't actionable.
This is ARC's few-shot underdetermination reappearing inside the verifier. The portfolio
(run A0 + dual arms, submit the union of train-verified solves) remains ≥ A0 by construction
— that is the only place the dual is non-negative, and it is a packaging trick, not guidance.

## Honest framing
Delivered: a rigorous, well-instrumented **negative result** about neurosymbolic CE-guidance
for code-gen ARC, reached with two modest paid passes + a suite of $0 probes
(`ce_replay.py`, `conditional_probe.py`, `dual_oracle.py`). The mechanism is sound and
correct; the approach does not lift solving. Recommended stance: keep the dual only as the
**portfolio floor** (no regression vs A0) and the $0 harnesses, and stop investing in the CE
channel for this generator class.
