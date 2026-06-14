# clarc — Handoff for the next agent

**Goal:** verify the hypothesis that *contract-learning helps an LLM solve ARC* —
i.e. that arms A5/A1/A1L beat the poetiq-style baseline A0. The **system is built
and unit-verified (32 tests)**.

> **UPDATE 2026-06-11 — BLOCKER RESOLVED.** The CLI accepts hidden argv flags
> (found in the `claude-agent-sdk` transport source, absent from `--help`):
> `--max-thinking-tokens N` (N ≥ 1024; lands in the API request as
> `thinking.budget_tokens` — verified by the API 400 on N=1000) and
> `--thinking disabled|adaptive`. **Crucially the model must be pinned to
> `claude-sonnet-4-5`**: the `sonnet` alias resolves to `claude-sonnet-4-6`,
> an adaptive-thinking model that treats numeric budgets as merely on/off →
> unbounded deliberation → the hangs in §1. With
> `--model claude-sonnet-4-5 --max-thinking-tokens 4000`, the hard ARC-2 task
> `20270e3b` (hung >1200s before) returns in **50s** with a real attempt, on
> CLI subscription auth (no API key, no setup-token needed). Also fixed: user-level
> MCP servers were leaking into `-p` runs (the model called Figma/Excalidraw MCP
> tools mid-puzzle!) — generator now passes `--strict-mcp-config`. The
> `MAX_THINKING_TOKENS` env var and `--effort` remain ignored in `-p`; the argv
> flags are the working knob, now plumbed through `generator.py` /
> `experiment.py --max-thinking/--thinking` / `run.py`.

Read `clarc/README.md` first for the architecture. §1 below is the original
blocker analysis (kept for context); §2 lists the alternative paths (now moot).

---

## 1. The original blocker (resolved above; kept for context)

The hypothesis test needs a generator that BOTH (a) returns fast on hard tasks and
(b) fails enough to leave room for contracts to help. The only LLM access on this
machine is the **Claude Code CLI** (`claude -p`) — there are **no API keys**. With it:

| generator × benchmark | result | usable? |
|---|---|---|
| sonnet, trivial prompt | 8s | — |
| sonnet, **easy** ARC-1 task | 68s, solves | one-shots → no room |
| sonnet, **hard** ARC-2 task | **>1200s, never returns** (hangs) | ✗ no attempt |
| sonnet, hard ARC-2, `--effort low/medium` | >250–600s | ✗ (knob ignored in `-p`) |
| sonnet, hard ARC-2, `MAX_THINKING_TOKENS=2k/8k` | >300s | ✗ (ignored in `-p`) |
| sonnet, hard ARC-2, lean prompt | >300s | ✗ (not the prompt) |
| sonnet, ARC-1 small-grid sweep (43 tasks) | **30 one-shot, 3 two-iter, 10 timeouts, 0 fast-fail** | ✗ no failure band |
| **haiku**, ARC-2 (22 tasks) | **0 solved, 21 real failures** | ✓ rich band — but **user vetoed haiku** |

**Root cause:** the interactive CLI gives sonnet an effectively *unbounded thinking
budget* on hard puzzles and exposes **no working cap** in `-p` mode. sonnet is
therefore **bimodal via CLI**: it one-shots easy tasks (no room for contracts) or
hangs on hard tasks (no attempt to evaluate). There is **no sonnet-via-CLI regime
with a failure band.**

**How real evals run sonnet (the fix):** via the **API** (litellm) with a *bounded*
thinking budget — poetiq sets `thinking.budget_tokens: 32000` (see
`arc_agi/llm.py` `props`). Bounded thinking → returns in seconds, even on hard
tasks. The CLI can't do this; the API can. **So you need an API-style path.**

Also fixed mid-session: the `claude` binary auto-updated (2.1.118 → 2.1.173) and
migrated `/opt/homebrew/bin/claude` → `~/.local/bin/claude` (not on PATH). The
generator now resolves this (`clarc/generator.py:_find_claude`).

---

## 2. Paths forward (alternatives, ranked)

To get a clean A0-vs-treatment result you must do **one** of these:

1. **`ANTHROPIC_API_KEY` in `.env`** → run **sonnet via litellm** with bounded
   thinking. `arc_agi/llm.py` already wraps litellm and sets `budget_tokens=32000`
   for `anthropic/claude-sonnet-4-5`. Write a `LiteLLMGenerator` (≈30 lines, wrap
   `arc_agi.llm.llm`) implementing clarc's `Generator` protocol; point the
   experiment at it. Fast, feasible. Spends API credits (~$10–30/run). *This is how
   evals run sonnet.*

2. **`GEMINI_API_KEY` in `.env`** → poetiq's **literal default model**
   (`gemini/gemini-3-pro-preview`, or `gemini-2.5-pro`). Same `LiteLLMGenerator`.
   Google AI Studio has a free tier. Most faithful to the poetiq baseline.

3. **⭐ Subscription session token (no API credits) — `claude setup-token`.**
   This mints a long-lived OAuth token tied to the **Claude subscription** (billed
   to the plan, *not* separate API credits → respects "no API tokens" in the cost
   sense). The promise: **API-style bounded thinking → fast sonnet on hard ARC-2,
   on subscription billing.** Concretely, for the next agent to verify:
   - Run `claude setup-token` (interactive; user does it once).
   - Try the token via the **Claude Agent SDK** (`CLAUDE_CODE_OAUTH_TOKEN` env var)
     or with **litellm/Anthropic SDK** (some versions accept an OAuth bearer token
     in place of `ANTHROPIC_API_KEY`).
   - **Unknown to test:** (a) whether litellm/the SDK accepts this token, and
     (b) whether you can set `thinking.budget_tokens` through that path so sonnet
     returns fast (<~60s) on a hard ARC-2 task. If yes → this is the ideal: fast
     bounded-thinking sonnet, subscription-billed, honors all constraints. **This is
     the most promising untested lever — prioritize verifying it.**

4. **haiku via CLI** (currently vetoed by the user). Fast + fails on ARC-2 → rich
   band → cleanest *mechanism* test, no key. Weaker model so absolute ARC-2 score
   floors ~0, but the **relative** A0-vs-A1L comparison (does the machinery engage
   and help) is fully testable. Only if the veto is lifted. Re-running it is
   blocked by the permission classifier until the user approves.

5. **Any other fast, output-controllable LLM** behind clarc's `Generator` protocol
   (≈30-line adapter). E.g. a local model.

**Recommendation:** verify **#3** first (subscription token → bounded-thinking
sonnet); if it can't bound thinking, fall back to **#1/#2** (a key) for a clean run,
or **#4** (haiku) for a free mechanism check.

---

## 3a. RESULT — full A0/A5/A1/A1L comparison on ARC-AGI-2 (2026-06-11)

`output/exp-arc2-s45-t4k/` — sonnet-4-5, 4k thinking budget, seed 0, pass@1.
Phase A: all 36 gen-safe tasks (≤20px) → 35 test-fail / 1 solve / 0 gen-fail —
the failure band finally exists. Phase B: **35-task band × 4 arms**, run in two
cohorts (12 tasks @ ≤6 iters, then 23 @ ≤10 — same per-task cap across arms, so
paired comparisons stay valid). 176 cells, 0 voids, ~$203 notional (subscription).

| arm | test acc | full solves | train-solved | med iters | conf | yield | induced |
|---|---|---|---|---|---|---|---|
| A0  | 3.8% | 1 | 3/35 | 8.0 | 0 | 0 | 0 |
| A5  | 3.8% | 1 | 1/35 | 5.0 | 0 | 0 | 0 |
| A1  | 3.8% | 1 | 2/35 | 4.0 | 294 | 0.19 | 0 |
| A1L | **6.2%** | 1 | 1/35 | **3.0** | 297 | 0.42 | 98 (11 uniq) |

**Verdict: NOT verified at this regime/n — directional at best.** 27/35 band tasks
floor at 0 for every arm. Only 8 tasks differentiate:

- `1ae2feb7` [6it]: monotone gradient A0 0.33 (never converges) → A5 train-solve@5
  → A1 @4 → A1L @3 with test 0.67. The cleanest pro-contract cell.
- `142ca369` [6it]: only A1L scores (0.5). `8f3a5a89` [10it]: only A1L full-solves
  (1.0, via best-result selection without train convergence).
- `e376de54` [10it]: **A5 and A1 full-solve, A0 and A1L get 0** — the static-spec
  ingredient wins; A1L's induction didn't help here (single-seed variance).
- `bf45cf4b` [10it]: **A0 full-solves @7 iters, all contract arms 0** — the one
  baseline win; late convergence the 6-iter cohort would have missed.
- `3dc255db`/`d35bdbdc`/`db0c5428`: train-convergence only (test 0 — overfit; two
  are A0-only, enabled by the 10-iter cap).

Paired sign tests vs A0 (acc): A5 +1/−1, A1 +1/−1, **A1L +3/−1** (p≈0.63);
any-treatment-beats-A0 +4/−1 (p≈0.38). Each arm's single full solve landed on a
DIFFERENT task — at seed=0/pass@1, per-task stochasticity rivals the arm effect.
What IS solid: machinery engages exactly as designed (conflicts ~8.4/cell in A1/A1L
vs 0; induction more than doubles clause yield 0.19→0.42; 98 inductions → 11 unique
human-readable contracts; `harm=0` — verified contracts never hurt) at equal cost
(~$49/arm).

**To actually verify (pick per budget):** (a) **multi-seed** the band (seed 1,2 →
pass@k / variance reduction; ~$150/seed for all 35×4) — highest information per
dollar given the variance observed; (b) lift the floor: 8–16k thinking budget
(slower/dearer per call) or `--max-grid 30+` to admit more/easier band tasks;
(c) accept the mechanism-level result and reframe the claim as "contracts speed
train convergence + add interpretability at zero cost/harm", which the data does
support; (d) **opus-4-8** — see §3c.

## 3c. Opus 4.8 trial (2026-06-12)

Probes on the sonnet-floor task `20270e3b` + a 9-task one-shot floor-sample
(`output/opus-floor-sample.log`, ~$10 notional total):

- **No thinking-control knob works on opus-4-8 in `-p`**: `--max-thinking-tokens
  8000` is ignored (adaptive model: 46k output tokens, 532s, $1.22) and `--effort
  medium` does nothing (63k tokens, 755s, $1.64 — worse than default). Unlike
  sonnet-4-6 it always *returns* eventually, but gens run 3–15 min (median ~6),
  ~$1/gen, and 2/10 hard-task gens hit a 900s timeout ($0, the old hang mode).
- **Opus one-shot lifts ~55% of the sonnet floor** (pass@1): full solves on
  20270e3b, 136b0064, 332f06d7, 409aa875, 4c3d4a41, half on 247ef758 = 5.5/10
  tasks where every bounded-sonnet arm scored 0 over 6–10 iters. The remaining
  ~45% (269e22fb, 28a6681f, 2b83f449, 6ffbe589 + timeouts) is a genuine
  mid-difficulty opus band — the differentiable surface contracts need.
- **Full opus experiment RAN 2026-06-12** (`output/exp-arc2-opus48/`, fresh
  contract library — the sonnet one is archived at
  `output/contract_library.sonnet-s45-t4k.json`). Phase A: 28/36 returned
  (8 voids = 900s timeouts), 11/28 one-shot-correct → band 14. Phase B
  (14 × A0,A1L × ≤6 iters) finished in ~1.5h, ~$62:

  | arm | test acc | train-solved | med iters | conf | induced | notional |
  |---|---|---|---|---|---|---|
  | A0  | **63.1%** | 11/14 | 1.0 | 0 | 0 | $29.78 |
  | A1L | 58.3% | 11/14 | 1.0 | 24 | 35 (5 uniq) | $32.27* |

  Paired: A0>A1L only on `e376de54` (1.0 vs 0 — A1L train-solved WITH 3 induced
  contracts but test-failed: a hint of contract-steered overfit, invisible to the
  `harm` metric which only counts pruning); A1L>A0 only on `b6f77b65` (0.33 vs 0).
  **+1/−1/=12 — dead even.** Two structural lessons: (1) at opus strength the
  1-shot band is UNSTABLE — Phase B re-solved 8/14 "failures" at med_iters=1,
  i.e. retry variance, not feedback, drives most recovery; (2) the contract
  effect visible in the weak-model regime washes out when the model is strong
  enough to retry-solve. *A1L cost pre-dates the proposal-cost accounting fix
  (commit dba9641) — true A1L cost is somewhat higher in ALL experiments so far.

## 3e. Phase 5 — the DSL ⇄ SMT dual (in progress, 2026-06-12)

Approved plan: `~/.claude/plans/from-all-the-existing-fluttering-iverson.md`.
Built so far (commits f55f229, ea08843, f409b80; 79 tests green):
- `absdomain.py` σ = (dims, cnt[10], n_obj, bbox, 5 sym bits) concrete + z3 sides;
- `dsl.py` 25 typed primitives (numpy apply ⊕ z3 relational contract, havoc
  documented), pipeline interpreter, `compile_pipeline` → existing sandbox path;
- `smt.py` CHECK-exact/class (params SHARED across pairs → cross-pair class
  refutations w/ labeled cores), SYNTH (shortest-first), verified position-lift;
- `clauses.py` refutation → minimized core → verified generalization ladder →
  clause store (syntactic firing + prompt rendering). LOO = robustness
  ANNOTATION, not a gate (clauses are deduced certificates, unlike induced
  spec predicates) — and OFF live for latency;
- loop integration: D0 (DSL try-and-drop) / D1 (+pre-exec refutation) /
  D2 (+clause learning) arms; stages dsl_invalid/dup/refuted; abs_weak
  component log on concrete-fail-despite-SAT; `--devset` in experiment.py;
- `probe_dsl.py` ($0 validation) + `audit_refutations.py` (soundness tripwire).

**Full $0 probe over the 40-task devset (`output/probe-dsl-devset.log`):
refutation power 99.4%** (13,552/13,639 non-fitting candidates refuted
pre-execution), **0 false refutations in 13,641 checks** (the soundness gate
for paid runs — PASSED), CHECK latency p50 137 ms / p95 442 ms / max 2.2 s,
depth-2 coverage 2/40 (`195ba7dc`, `31d5ba1a`, both logic stratum) — the young
catalog's exact-solve floor, reported separately from mechanism claims. The
~0.6% non-refuted band is where concrete-fail learning + abs_weak logging
operate. Next: gated haiku runs (smoke → mini → full, pinned
claude-haiku-4-5-20251001); pre-registered STRONGER/LEARNS criteria in the
plan file.

## 3f. Phase 5b — M5 object catalogue: built, sound, coverage-flat (2026-06-14)

Plan: `~/.claude/plans/from-all-the-existing-fluttering-iverson.md`. Built
(commits b8a1a32, 046e506, …; 89 tests green): `dsltypes.py` (Grid/Selection
threaded types + Obj/Selection carriers); typechecker in `dslparse` (rejects
ill-typed pipelines — the sketch's T1◁T_I); per-object σ extension `osz`/`ocol`
(6 largest sizes + objects-by-color, all-linear WF); `dslobj.py` 11 object
combinators (`objects → select/recolor → render`) registered into the same
REGISTRY (σ tracks the rendered grid, so the SMT layer needed ZERO change).

Fuzz debugging surfaced **4 real soundness bugs, one root cause**: recoloring /
duplicate-line removal can flip which color is the most-frequent "background",
making n_obj/nonbg/osz unpredictable → those are correctly havoc'd now; and
removing objects GROWS bg count, so the subset bound excludes bg. This is a
genuine finding about σ's fragility under color changes.

**KEY RESULT (decisive):** coverage probe over the 40-task devset
(`output/probe-dsl-m5-devset.log`): **2/40 (0/20 structural, 2/20 logic) —
UNCHANGED from before M5.** Refutation power 96.4% (17,123/17,759), **0 false
refutations in 17,761 checks** (soundness held across the whole object layer).
The 11 hand-designed object prims unlocked ZERO new tasks: the structural
stratum needs PER-OBJECT RULES (recolor each object by size-rank / shape /
position — verified: 18/20 structural tasks are per-object/positional
recolorings, not global maps), which are effectively task-specific. **A fixed
hand-designed catalogue undershoots ARC's structural diversity** — exactly the
"we can't pre-design contracts" point that motivates M6 (agent-inferred
primitives gated by the SMT sufficiency oracle). M5 thus delivers the sound
composable *framework* + object basis; M6 is where coverage actually grows.
Reframes M5's "≥8/20 by hand-design" bar as the wrong target.

## 3g. M5d — per-object shape signatures in σ (2026-06-14)

Per the "sharpen the tool before M6" decision: σ now carries per-object SHAPE
discrimination, not just size/color. Added `oshape[0..4]` (disjoint partition
dot/hline/vline/rect/other, Sum==n_obj), `n_holed`, `n_border` — all linear WF,
one labeling pass. Retrofitted (each fuzz-verified sound): geometry permutes
hline↔vline under h/w-swap and preserves sizes/colors/holes/border; scale scales
`osz` + preserves holes/border; crop_bbox preserves the whole object summary;
selectors bound shape/hole/border by subset. recolor stays havoc (the bg-flip
finding from M5c). New refutation power shown in tests: a line-rotation task
refutes `flip_h` via `oshape` (rot90 stays feasible); `select_color` refutes the
wrong color via exact `ocol`. 91 tests green.

**HONEST RESULT — σ-enrichment did NOT move aggregate refutation power.** Devset
probe (`output/probe-dsl-m5d-devset.log`): 96.4% refutation, 0 false / 17,761 —
**identical to the digit to M5c** (pre-shape). A targeted measurement explains
why: of 10,346 non-fitting depth-1 candidates, **99.0% are refuted by dims+
histogram alone**; the remaining 1% (99) are all caught by M5b's `osz`/`ocol`/
`bbox`; the M5d shape fields are the SOLE discriminator for **0** of them (the
real flip_h-vs-rot90 case just doesn't arise in this enumeration). **Conclusion:
refutation power is saturated (~96–99%) from COARSE features — it is NOT the
binding constraint. The wall is COVERAGE (expressible `apply`s), which only M6
(induction) addresses.** M5d is kept (sound, cheap, 0 harm); its residual value
is that M6's auto-derived contracts *can* express shape-conditional invariants —
but expect no refutation gain. The "sharpen the tool first" hypothesis was
reasonable, and measurement (partially) refuted it: the tool wasn't dull.

## 3d. Predicate-loop history capture (commit dba9641)

Per user request, every FUTURE experiment cell now dumps its complete
predicate-loop history to `<out>/logs/<phase>_<task>_<arm>.json`: per-iteration
injected contract set (`active`), fired violations (`violated`), induction
outcome with gate-rejection stage (`proposed`: no-parse | gate1-soundness |
gate2-relevance | admitted, `prop_admitted`, `prop_cost_usd`), and the admitted
contracts WITH predicate code (`learned_contracts`); the cross-task library is
snapshotted to `<out>/contract_library.final.json` at run end. For
`exp-arc2-opus48` itself this landed too late — only runs.jsonl summaries and
the final library (with code) exist; the per-iteration history of the four
divergent cells (`e376de54`, `b6f77b65`, `2b83f449`, `abc82100`) is the main
loss. A targeted A1L replay of those 4 tasks (~$25) would recover it if wanted.

## 3b. What the earlier data showed

On-disk runs (`output/*/runs.jsonl`; each line `{phase,task,arm,solved,iters,acc,
conf,learned,n_learned,cost,...}`):

- **`output/experiment/`** — the **only Phase B comparison** (ARC-1, 2 band tasks,
  arms A0/A5/A1/A1L, 7 cells). Result hint: **A0 solved in 3 iters; A5 in 2; A1 and
  A1L in 1.** I.e. injecting verified contracts made the model converge in **1
  iteration vs 3**. *Directionally supports the hypothesis (contracts speed
  convergence), but n=2 — anecdotal, not verification.* `conf=0, learned=0` there
  (tasks solved before a conflict, so the CDCL/induction parts didn't fire).
- **`output/exp-arc1-sonnet/`** — sonnet ARC-1 sweep: 30 one-shot, 3 two-iter, 10
  timeouts, **0 failure-band**. Confirms sonnet is too strong / bimodal here.
- **`output/exp-arc2-haiku/`** — haiku ARC-2: **0/22 solved** → rich failure band
  (the regime the hypothesis needs), but haiku is vetoed. Shows ARC-2 is genuinely
  hard (consistent with it being unsolved/SOTA territory — the poetiq benchmark).
- **`output/experiment-arc2/`** — sonnet ARC-2: all 10 timeouts ($0).

**Two viable verification framings (pick per generator):**
- **Solve-rate** (needs a failure band): does A1/A1L solve test tasks A0 can't?
  Needs a model that *fails but returns* (haiku, or bounded-thinking sonnet/gemini
  on hard ARC-2). This is the strong test.
- **Iteration-reduction** (works even on solved tasks): does injecting contracts
  reduce iterations-to-solve? Band = tasks the baseline solves in >1 iter OR fails.
  The `experiment.py` band def was just changed to this (`_room()`); the n=2 hint
  above is exactly this signal. Weaker but constraint-friendly with a strong model.

---

## 4. How to run

Offline (always): `uv run pytest -q` (32 tests; real-CLI tests are `-m cli`, skipped
by default).

**The experiment harness** is `clarc/experiment.py` — resumable, supervised:
- Phase A: sweep `--n` tasks with A0 (`--sweep-iters`) → compute the band.
- Phase B: run `--arms` on the band (`--iters`); pass@1 via `score_task`.
- **Resumable:** every `(phase,task,arm)` cell appends to `<out>/runs.jsonl` and is
  **skipped on re-run** — kills lose ≤1 cell; relaunch to continue. Verified.
- `--report-only` prints the report from a checkpoint without running.
- Band def: `_room()` = real attempt AND (test-wrong OR >1 iter). Edit to taste
  (e.g. failure-only for solve-rate framing).

**The working recipe (bounded-thinking sonnet-4-5 on CLI subscription auth) — launched
2026-06-11 into `output/exp-arc2-s45-t4k/`:**
```bash
uv run python -m clarc.experiment --data 2025-eval --model claude-sonnet-4-5 \
  --max-thinking 4000 --n 120 --max-grid 20 --sweep-iters 2 --iters 6 \
  --max-band 12 --arms A0,A5,A1,A1L --concurrency 4 --timeout 300 \
  --out output/exp-arc2-s45-t4k
uv run python -m clarc.experiment --report-only --out output/exp-arc2-s45-t4k
```
Must-knows: pin `claude-sonnet-4-5` (the `sonnet` alias = 4-6, ignores numeric
budgets); budget ≥1024; ~50s and ~$0.09 notional per generation (subscription-billed);
`clarc/probe_bounded.py` re-checks the regime per task. Phase-B cells with zero real
attempts (throttle/outage) are NOT checkpointed (`_void`) so a relaunch retries them.
`run.py` solves specific tasks; `devset.py` curates a stratified dev set.

**Supervise long runs** with a heartbeat monitor (the experiment also emits a
background-task completion/kill notification). On kill → relaunch (it resumes).

---

## 5. Gotchas / lessons (don't relearn these)

- **NEVER broad-kill** (`pkill -f claude`/`node`/`python`) — it hits other sessions.
  Use `TaskStop <task_id>` or `kill <PID>` on a confirmed unique PID. (Now in the
  user's global `~/.claude/CLAUDE.md`.)
- **Generator must use CLI subscription auth, not tokens** — `clarc/generator.py`
  strips `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` from the subprocess env. If you
  add a key-based `LiteLLMGenerator`, that's a *different* generator; keep them
  separate and let the user choose.
- **`claude -p` flags that DON'T bound thinking:** `--effort`, `MAX_THINKING_TOKENS`
  (both ignored in `-p` here). `--max-budget-usd` cuts mid-thinking → no code.
- **Big grids (≥~20) make sonnet hang** — `experiment.py --max-grid` excludes them.
- **stdout from background python is block-buffered** — prefix `env PYTHONUNBUFFERED=1`
  if you want to read interim output; otherwise rely on the flushed `runs.jsonl`.
- **Background commands sometimes get killed at idle** (cause unconfirmed: possibly
  cross-session kills or a runtime cap) — hence the resumable checkpoint. Don't fight
  it; just relaunch.
- ARC-AGI-1 is largely solved by strong models (sonnet ~83%); **ARC-AGI-2
  (`data/arc-prize-2025`) is the real, hard, poetiq-headline benchmark.**

---

## 6. Uncommitted state to commit

On branch `dsl`. Last commit `5176ae0` (induced-predicate pruning). **Uncommitted**
(commit these — they're correct and load-bearing):
- `clarc/generator.py` — CLI-path migration fix (`_find_claude`), `effort`,
  `max_thinking_tokens`, env scrub + `~/.local/bin` PATH insurance.
- `clarc/loop.py` — `LEAN_SOLVER_PROMPT` + `lean_prompt` branch.
- `clarc/types.py` — `lean_prompt`, `effort`/`max_thinking`-related config.
- `clarc/run.py` — arm `A1L`, `n_learned` reporting.
- `clarc/experiment.py` (untracked) — the resumable failure-band/iteration runner.
- `knowledge/` (untracked) — the research design docs (author's; confirm before committing).

Suggested next steps: (1) verify alternative #3 (subscription token → bounded
sonnet); (2) wire a `LiteLLMGenerator`; (3) run A0/A5/A1/A1L on ARC-AGI-2 with a
fast generator; (4) report solve-rate + iteration deltas + machinery engagement
(`conf`/`n_learned`) with the induced contracts listed.
