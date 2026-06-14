"""Sandbox for LLM-proposed contract predicates.

A proposed contract is arbitrary model-written code defining `holds(inp, out) ->
bool`. We NEVER exec it in-process: it runs in a subprocess (like
arc_agi.sandbox), batched over all pairs in one call, with a timeout. This is the
verification gate's executor — the soundness of "learned" contracts rests on it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import textwrap
from typing import Optional

import numpy as np

Pair = tuple[np.ndarray, np.ndarray]

_SCRIPT = """
{code}

if __name__ == '__main__':
    import json, sys
    import numpy as np
    data = json.load(sys.stdin)
    results = []
    for inp, outp in data['pairs']:
        try:
            r = holds(np.array(inp), np.array(outp))
            results.append(bool(r))
        except Exception:
            results.append(None)
    print(json.dumps({{'results': results}}))
"""


_TRANSFORM_SCRIPT = """
{code}

if __name__ == '__main__':
    import json, sys
    import numpy as np
    import scipy
    data = json.load(sys.stdin)
    results = []
    for grid in data['grids']:
        try:
            r = transform(np.array(grid))
            arr = np.asarray(r, dtype=int)
            if arr.ndim == 2 and arr.size > 0:
                results.append(arr.tolist())
            else:
                results.append(None)
        except Exception:
            results.append(None)   # untrusted code: any failure -> None (isolated)
    print(json.dumps({{'results': results}}))
"""


async def run_transform_batch(
    code: str, grids: list[np.ndarray], *, timeout_s: float = 10.0
) -> Optional[list[Optional[np.ndarray]]]:
    """Run an LLM `transform(grid)->grid` over a batch of grids in ONE subprocess.

    Returns per-grid output arrays (None where the code errored / returned a
    non-grid), or None if the whole run failed. Used by primitive induction to
    sample σ-deltas cheaply without execing untrusted code in-process.
    """
    if "def transform" not in code:
        return None
    script = textwrap.dedent(_TRANSFORM_SCRIPT).format(code=code)
    payload = {"grids": [np.asarray(g, dtype=int).tolist() for g in grids]}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tf.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=td,
                env={"PYTHONHASHSEED": "0"},
            )
            out, _err = await asyncio.wait_for(
                proc.communicate(input=json.dumps(payload).encode()), timeout=timeout_s
            )
        except (asyncio.TimeoutError, OSError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return None
    if proc.returncode != 0:
        return None
    try:
        raw = json.loads(out.decode())["results"]
    except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
        return None
    return [None if r is None else np.array(r, dtype=int) for r in raw]


async def verify_predicate(
    code: str, pairs: list[Pair], *, timeout_s: float = 5.0
) -> Optional[list[Optional[bool]]]:
    """Run `holds` over (input, output) pairs in a subprocess.

    Returns a list of per-pair results (True/False/None-on-error), or None if the
    whole run failed (syntax error, no `holds`, timeout, bad output).
    """
    if "def holds" not in code:
        return None
    script = textwrap.dedent(_SCRIPT).format(code=code)
    payload = {"pairs": [[gi.tolist(), go.tolist()] for gi, go in pairs]}

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "pred.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=td,
                env={"PYTHONHASHSEED": "0"},
            )
            out, _err = await asyncio.wait_for(
                proc.communicate(input=json.dumps(payload).encode()), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return None
        except Exception:
            return None

    if proc.returncode != 0:
        return None
    try:
        return json.loads(out.decode())["results"]
    except Exception:
        return None
