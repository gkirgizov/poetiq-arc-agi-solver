"""Probe: do the hidden CLI flags `--max-thinking-tokens N` / `--thinking disabled`
bound sonnet so it RETURNS on hard ARC-2 tasks (which hang >1200s unbounded)?

Builds the exact A0 iteration-0 prompt and times one generator call per config.

FINDINGS (2026-06-11, CLI 2.1.173):
  - `sonnet` alias = claude-sonnet-4-6 (ADAPTIVE thinking): numeric budgets are
    treated as on/off -> still unbounded -> hangs. Both configs timed out at 280s.
  - PINNED claude-sonnet-4-5 honors the numeric budget (API `thinking.budget_tokens`,
    min 1024 — verified via the API 400 on smaller values): task 20270e3b returned
    in 50s with a real attempt vs >1200s unbounded. THE working recipe.

    uv run python -m clarc.probe_bounded [task_id ...]
"""

from __future__ import annotations

import asyncio
import sys
import time

from arc_agi.prompts import SOLVER_PROMPT_1
from arc_agi.solve_coding import _build_prompt, _make_example, format_problem

from clarc.generator import ClaudeCodeGenerator
from clarc.run import _load

DEFAULT_TASKS = ["20270e3b"]  # hung >1200s via plain `claude -p` (output/experiment-arc2)

# label -> (model, extra argv). Pin 4-5 for real budgets; `sonnet` (=4-6) ignores them.
CONFIGS = {
    "45+4k": ("claude-sonnet-4-5", ("--max-thinking-tokens", "4000")),
    "45+off": ("claude-sonnet-4-5", ("--thinking", "disabled")),
}


def a0_prompt(task: dict) -> str:
    example = _make_example(
        [e["input"] for e in task["train"]],
        [e["output"] for e in task["train"]],
        [e["input"] for e in task["test"]],
    )
    return _build_prompt(SOLVER_PROMPT_1, problem=format_problem(example, True, 0))


async def probe(label: str, model: str, extra: tuple[str, ...], prompt: str, timeout: float):
    gen = ClaudeCodeGenerator(model=model, timeout_s=timeout, extra_args=extra)
    t0 = time.monotonic()
    g = await gen.generate(prompt, seed=0)
    dt = time.monotonic() - t0
    print(f"[{label:9s}] {dt:6.1f}s  code={'YES' if g.code else 'NO '}  "
          f"err={g.error}  ctok={g.completion_tokens}  ${g.cost_usd:.3f}", flush=True)


async def main():
    tasks = sys.argv[1:] or DEFAULT_TASKS
    challenges, _ = _load("2025-eval")
    for tid in tasks:
        prompt = a0_prompt(challenges[tid])
        print(f"--- {tid} (prompt {len(prompt)} chars) ---", flush=True)
        await asyncio.gather(*[
            probe(label, model, extra, prompt, timeout=280.0)
            for label, (model, extra) in CONFIGS.items()
        ])


if __name__ == "__main__":
    asyncio.run(main())
