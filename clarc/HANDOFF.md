# clarc — Handoff for the next agent

**Goal:** verify the hypothesis that *contract-learning helps an LLM solve ARC* —
i.e. that arms A5/A1/A1L beat the poetiq-style baseline A0. The **system is built
and unit-verified (32 tests)**; the **live hypothesis test is NOT done**, blocked on
LLM access (details below). This doc gives you everything to finish it.

Read `clarc/README.md` first for the architecture. This doc focuses on **what we
learned, why we're blocked, and how to unblock + run.**

---

## 1. The blocker (read this first)

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

## 3. What the existing data already shows

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

Example (once a fast generator exists — adapt `--model`/backend):
```bash
# with a key + a LiteLLMGenerator wired in (alt 1/2/3):
uv run python -m clarc.experiment --data 2025-eval --n 120 --max-grid 20 \
  --sweep-iters 2 --iters 6 --arms A0,A5,A1,A1L --concurrency 8 --out output/exp-arc2
uv run python -m clarc.experiment --report-only --out output/exp-arc2
```
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
