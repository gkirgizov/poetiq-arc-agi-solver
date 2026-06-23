# clarc — Research findings

The honest, load-bearing results of the contract-learning effort. Headline insights
below; the full lab notebooks (with per-run numbers and dead ends) live in
[`docs/notebook/`](docs/notebook/) — `CEGIS.md` (the DSL⇄SMT thread), `HANDOFF.md`
(the contract-learning A-arm evals), `RD_NEXT.md` (the dual / counterexample thread).

Two things run through all of it: **soundness** (a lesson enters the system only after
Python/z3 re-verifies it on every train pair — no hallucinated knowledge accumulates)
and **honesty** (the $0 z3 probes, not the paid LLM runs, are the reliable instrument
here, and the negatives are reported as plainly as the positives).

---

## 1. Growing the DSL lifts *sound, LLM-free* coverage — the durable positive
`symmetry_repair` (fill noise cells via any mirror symmetry consistent on the
non-occluded cells + translational periods) solves **8** ARC-1 eval tasks, and
`connect_dots` solves 1 more — **all test-generalizing, all found with no LLM**:
`synth_models` proposes the primitive from the abstract spec and `param_search` pins
its parameters. It even recovers a hard instance the free-form LLM baseline misses
(`929ab4e9`). The lever that works is *data-driven DSL development*: implement a
candidate clean-algorithm family, measure net-new yield across the eval, keep the
winners. (Single-prim mining is now saturated — see #6.)

## 2. The param-search lever — a synthesis limit, not a DSL limit
Pure-symbolic coverage sat at **0/40** on the devset. The cause was not the DSL's
expressiveness but z3's: `synth_models` proposes the *right* skeleton (e.g.
`split_binop_h`) yet z3 returns *one arbitrary* parameter witness (`(and,1)`) because
the σ-abstraction is too coarse to pin params. A trivial concrete search over the
skeleton's tiny parameter space recovers the real solver (`(or,1)`, `(xor,6)`),
lifting pure-synth coverage to **2/40 — both test-generalizing, entirely LLM-free**.
The first non-zero pure-symbolic result, and a clean sound win. (`dsl.param_search`.)

## 3. Faithful CEGIS with sound verification
The DSL⇄SMT layer satisfies all four type-directed-synthesis criteria *simultaneously*:
typed-DSL candidates, feedback traceable to the violated contract via the z3 unsat
core, a monotone clause lattice (`exact → at_pos → anywhere`), and synthesis drawn
from the clause-pruned feasible subspace. Measured refutation power is **99.4%**
(13,552 / 13,639 non-fitting candidates rejected before execution) with **0 false
refutations in 13,641 checks** — the soundness tripwire that gates every paid run. The
full loop closes end-to-end: `009d5c81` was solved by synthesis composing an *induced*
primitive, with no LLM-DSL emission at all.

## 4. Verified contracts speed convergence at zero harm (the A-arm mechanism result)
On ARC-AGI-2 with a bounded-thinking generator, injecting verified invariants cut the
median iterations-to-train-solve **8 → 5 → 4 → 3** across `A0 → A5 → A1 → A1L`;
open-vocabulary induction doubled clause yield (0.19 → 0.42) and produced 11 unique
human-readable invariants from 98 inductions; **`harm = 0`** (a verified contract never
pruned a correct candidate). The *solve-rate* lift is directional only (A1L 6.2% vs A0
3.8%, p≈0.63) — at this regime the hard floor dominates and per-task variance rivals the
arm effect. The defensible claim is mechanism-level: contracts converge faster and add
interpretability, for free.

## 5. The counterexample channel does not guide code-gen — a rigorous negative
There is **no regime where an SMT/invariant counterexample both fires *and* is
actionable**. *Envelope* contracts (shape / count / palette / object-preservation)
generalize, but a competent generator's wrong candidates already satisfy them — the CE
is inert (fires on ~2% of wrong iterations on easy tasks). *Content* contracts
(`color = f(size|rank|shape)`) do see the errors (fire ~77%), but at 3–4 examples they
are coincidental and survive leave-one-out, so refuting with them *overfits*. On
structurally-hard ARC-2 the envelope CE fires soundly (~41%) but is not actionable
(~4% repair; the gated strong-CE arm is net-harmful). This is ARC's few-shot
underdetermination reappearing *inside* the verifier. The dual is therefore kept only
as the **portfolio floor** (G0 always runs ⇒ the union of train-verified solves is ≥ A0
by construction) and the verified-selection signal.

## 6. The structural wall is the genuine ARC difficulty — confirmed three ways
Cracking the structural stratum needs generalizable induction, which overfits
coincidental rules from 3–4 examples. The reach is bounded from three independent
angles: single-prim DSL mining is **saturated** (only symmetry + connect were net-new
across ~20 candidate families); 2-prim composition adds **0/18** on the tippable band;
and per-object induction cannot express the restructuring tail (objects that split,
grow, or spawn). The `E3` holdout gate makes induction *sound* (it rejects the overfit
prim) but cannot *conjure* the correct rule — a correctness problem, not a complexity
one. This is the fundamental difficulty, not a fixable mechanism gap; it is the live
frontier (see the README).
