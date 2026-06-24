"""Generators: turn a prompt into a candidate `transform` program.

ClaudeCodeGenerator drives the headless Claude Code CLI (`claude -p`) as a
one-shot, tool-less text generator. The Python loop owns iteration, so we want a
single deterministic call per generation:

    claude -p --output-format json --tools "" --no-session-persistence [--model M]

- Prompt is fed via STDIN (no argv length limit, no shell quoting).
- `--tools ""` disables every tool => pure text generation, hermetic, one turn.
- `--no-session-persistence` keeps runs from writing session files.
- We deliberately do NOT use `--bare` (it forces ANTHROPIC_API_KEY auth and would
  break subscription/CLI auth) or `--max-turns` (absent in this CLI version).
- Auth is the user's existing CLI login; no API token is read by clarc.

StubGenerator is an offline, deterministic generator for tests/CI.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Optional, Sequence

from clarc.common.codeparse import parse_transform_code
from clarc.common.types import GenOutput

def _find_claude() -> str:
    """Resolve the claude binary, robust to the native-install migration
    (~/.local/bin) when it's not on the inherited PATH."""
    p = shutil.which("claude")
    if p:
        return p
    for c in (os.path.expanduser("~/.local/bin/claude"),
              "/opt/homebrew/bin/claude", "/usr/local/bin/claude"):
        if os.path.exists(c):
            return c
    return "claude"


_CLAUDE_BIN = _find_claude()
_LOCAL_BIN = os.path.expanduser("~/.local/bin")
# Repo root = parent of the clarc/ package dir. Used as a trusted cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ClaudeCodeGenerator:
    """Generator backed by the headless Claude Code CLI."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        timeout_s: float = 180.0,
        cwd: Optional[str] = None,
        max_budget_usd: Optional[float] = None,
        effort: Optional[str] = None,
        max_thinking_tokens: Optional[int] = None,
        thinking: Optional[str] = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.cwd = cwd or _REPO_ROOT
        self.max_budget_usd = max_budget_usd
        self.effort = effort           # low|medium|high|xhigh|max; caps thinking time
        # Hidden CLI flags (verified accepted in -p; the Agent SDK passes the same):
        # --max-thinking-tokens N bounds extended thinking; --thinking disabled|adaptive
        # turns it off entirely / makes it adaptive. The MAX_THINKING_TOKENS env var is
        # IGNORED in -p mode — these argv flags are the working knob.
        self.max_thinking_tokens = max_thinking_tokens
        self.thinking = thinking
        self.extra_args = tuple(extra_args)

    def _env(self) -> dict:
        # Honor "CLI subscription auth only": inherit the environment (so claude can
        # find its OAuth creds via HOME/keychain) but strip API-token vars so `claude
        # -p` can NEVER fall back to API-token billing.
        env = dict(os.environ)
        for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            env.pop(k, None)
        if _LOCAL_BIN not in env.get("PATH", ""):
            env["PATH"] = _LOCAL_BIN + os.pathsep + env.get("PATH", "")
        return env

    def _argv(self) -> list[str]:
        argv = [
            _CLAUDE_BIN,
            "-p",
            "--output-format", "json",
            "--tools", "",                  # disable all BUILT-IN tools
            "--strict-mcp-config",          # ...and block user-config MCP servers (they
                                            # otherwise attach and pollute generations)
            "--no-session-persistence",
        ]
        if self.model:
            argv += ["--model", self.model]
        if self.effort:
            argv += ["--effort", self.effort]
        if self.thinking:
            argv += ["--thinking", self.thinking]
        elif self.max_thinking_tokens is not None:
            argv += ["--max-thinking-tokens", str(self.max_thinking_tokens)]
        if self.max_budget_usd is not None:
            argv += ["--max-budget-usd", str(self.max_budget_usd)]
        argv += list(self.extra_args)
        return argv

    async def generate(self, prompt: str, *, seed: int) -> GenOutput:
        # `seed` is unused by the CLI (no seed flag); kept for interface parity and
        # so callers can vary prompts per-iteration upstream.
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._argv(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=self._env(),
            )
        except FileNotFoundError:
            return GenOutput(code=None, raw="", error="claude-cli-not-found")

        try:
            out, err = await asyncio.wait_for(
                proc.communicate(input=prompt.encode()), timeout=self.timeout_s
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return GenOutput(code=None, raw="", error="generator-timeout")

        if proc.returncode != 0:
            msg = (err.decode(errors="replace") or out.decode(errors="replace")).strip()
            return GenOutput(code=None, raw="", error=f"exit {proc.returncode}: {msg[:500]}")

        try:
            obj = json.loads(out.decode(errors="replace"))
        except json.JSONDecodeError as e:
            return GenOutput(code=None, raw=out.decode(errors="replace")[:500], error=f"bad-json: {e}")

        return _parse_envelope(obj)


def _parse_envelope(obj: dict) -> GenOutput:
    """Map the `claude -p --output-format json` envelope to a GenOutput."""
    usage = obj.get("usage") or {}
    cost = float(obj.get("total_cost_usd") or 0.0)
    ptok = int(usage.get("input_tokens") or 0)
    ctok = int(usage.get("output_tokens") or 0)

    if obj.get("is_error") or obj.get("subtype") != "success":
        return GenOutput(
            code=None,
            raw=str(obj.get("result", "")),
            cost_usd=cost,
            prompt_tokens=ptok,
            completion_tokens=ctok,
            error=f"cli-error: subtype={obj.get('subtype')} api_error={obj.get('api_error_status')}",
        )

    text = obj.get("result", "") or ""
    return GenOutput(
        code=parse_transform_code(text),
        raw=text,
        cost_usd=cost,
        prompt_tokens=ptok,
        completion_tokens=ctok,
        error=None if parse_transform_code(text) else "no-code-block",
    )


class StubGenerator:
    """Deterministic offline generator for tests. Returns scripted code per call.

    Each entry of `scripts` may be either raw `transform` source or a full
    response containing a ```python``` block. The last script repeats if calls
    exceed the list length.
    """

    def __init__(self, scripts: Sequence[str]) -> None:
        self.scripts = list(scripts)
        self.calls = 0

    async def generate(self, prompt: str, *, seed: int) -> GenOutput:
        raw = self.scripts[min(self.calls, len(self.scripts) - 1)] if self.scripts else ""
        self.calls += 1
        code = parse_transform_code(raw)
        if code is None:
            # treat the script as raw source (transform OR predicate)
            code = raw or None
            raw = f"```python\n{raw}\n```"
        return GenOutput(code=code, raw=raw, cost_usd=0.0, error=None if code else "no-code-block")
