"""Strict parser for DSL pipelines.

Grammar (one statement, steps separated by ';'):
    pipeline := step (";" step)*
    step     := NAME "(" args? ")"
    args     := atom ("," atom)*
    atom     := INT | IDENT | INT "->" INT      (the arrow form only for recolor)

Positional args bind to the primitive's declared params in order; recolor takes
any number of `a->b` pairs (unmapped colors stay identity). Errors carry a
message suitable for grammar-echo feedback to the LLM.
"""

from __future__ import annotations

import re

from clarc.dsl.core import N_COLORS, REGISTRY, Pipeline, Step
from clarc.dsl.types import Ty
from clarc.common.llm_parse import extract_block


class DslError(ValueError):
    pass


class DslTypeError(DslError):
    """Ill-typed composition (a step's input type ≠ the threaded value type)."""


_STEP_RE = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*\(\s*(.*?)\s*\)\s*$", re.S)
_MAP_RE = re.compile(r"^(\d)\s*->\s*(\d)$")


def extract_dsl_block(text: str) -> str | None:
    """Pull the last ```dsl fenced block out of an LLM response."""
    return extract_block(text, "dsl", last=True)


def parse_pipeline(src: str, registry: dict | None = None) -> Pipeline:
    reg = registry if registry is not None else REGISTRY
    src = src.strip().rstrip(";")
    if not src:
        raise DslError("empty pipeline")
    steps = []
    for chunk in src.split(";"):
        if chunk.strip():
            steps.append(_parse_step(chunk, reg))
    if not steps:
        raise DslError("empty pipeline")
    _typecheck(steps, reg)
    return Pipeline(tuple(steps))


def _typecheck(steps: list[Step], registry: dict) -> None:
    """Thread the value type from GRID; reject ill-typed compositions and a
    pipeline that doesn't end on a Grid (the sketch's T1 ◁ T_I, syntactic half)."""
    cur = Ty.GRID
    for step in steps:
        prim = registry[step.name]
        if prim.in_type != cur:
            raise DslTypeError(
                f"{step.name} expects {prim.in_type.value} but the pipeline holds "
                f"{cur.value} here"
                + (f" — call render() to get back a Grid" if cur == Ty.SELECTION
                   and prim.in_type == Ty.GRID else ""))
        cur = prim.out_type
    if cur != Ty.GRID:
        raise DslTypeError(f"pipeline ends as {cur.value}, must end as Grid "
                           f"(add render())")


def _parse_step(chunk: str, registry: dict) -> Step:
    m = _STEP_RE.match(chunk)
    if not m:
        raise DslError(f"cannot parse step {chunk.strip()!r}: expected name(args)")
    name, argstr = m.group(1), m.group(2)
    prim = registry.get(name)
    if prim is None:
        raise DslError(f"unknown primitive {name!r}")
    args = [a.strip() for a in argstr.split(",") if a.strip()] if argstr.strip() else []

    if name == "recolor":
        params: dict = {}
        for a in args:
            mm = _MAP_RE.match(a)
            if not mm:
                raise DslError(f"recolor arg {a!r}: expected digit->digit")
            params[f"pi{int(mm.group(1))}"] = int(mm.group(2))
        full = {f"pi{c}": params.get(f"pi{c}", c) for c in range(N_COLORS)}
        return Step(name, full)

    if len(args) != len(prim.params):
        sig = ", ".join(p.name for p in prim.params)
        raise DslError(f"{name} takes {len(prim.params)} arg(s) ({sig}), got {len(args)}")
    params = {}
    for spec, a in zip(prim.params, args):
        val: object = int(a) if re.fullmatch(r"-?\d+", a) else a
        if val not in spec.values:
            raise DslError(f"{name}: {spec.name}={a!r} not in {spec.values}")
        params[spec.name] = val
    return Step(name, params)
