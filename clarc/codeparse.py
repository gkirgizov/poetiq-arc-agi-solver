"""Parse a Python `transform` function out of an LLM response.

Mirrors poetiq's `_parse_code_from_llm` (arc_agi/solve_coding.py:208) so A0 is a
faithful baseline.
"""

import re
from typing import Optional

_CODE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_transform_code(response: str) -> Optional[str]:
    """Return the first ```python ...``` block body, or None if absent."""
    if not response:
        return None
    m = _CODE_RE.search(response)
    return m.group(1) if m else None
