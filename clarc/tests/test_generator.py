"""Unit tests for the generator envelope parsing and the offline stub.

No network: `_parse_envelope` is tested on synthetic dicts; StubGenerator is pure.
"""

import pytest

from clarc.common.codeparse import parse_transform_code
from clarc.solve.generator import StubGenerator, _parse_envelope


def test_parse_envelope_success():
    obj = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "Here:\n```python\ndef transform(grid):\n    return grid\n```",
        "total_cost_usd": 0.0123,
        "usage": {"input_tokens": 42, "output_tokens": 7},
    }
    out = _parse_envelope(obj)
    assert out.error is None
    assert "def transform" in out.code
    assert out.cost_usd == pytest.approx(0.0123)
    assert out.prompt_tokens == 42 and out.completion_tokens == 7


def test_parse_envelope_cli_error():
    obj = {"subtype": "error_max_turns", "is_error": True, "result": "",
           "total_cost_usd": 0.0, "usage": {}}
    out = _parse_envelope(obj)
    assert out.code is None
    assert out.error and "cli-error" in out.error


def test_parse_envelope_success_but_no_code_block():
    obj = {"subtype": "success", "is_error": False,
           "result": "I cannot help with that.", "usage": {}}
    out = _parse_envelope(obj)
    assert out.code is None
    assert out.error == "no-code-block"


def test_parse_transform_code_variants():
    assert parse_transform_code("```python\nx=1\n```") == "x=1\n"
    assert parse_transform_code("no code here") is None
    assert parse_transform_code("") is None


async def test_stub_generator_cycles_and_wraps_raw():
    gen = StubGenerator(["def transform(g): return g"])
    out = await gen.generate("prompt", seed=0)
    assert out.code == "def transform(g): return g"
    assert "```python" in out.raw
    assert gen.calls == 1
