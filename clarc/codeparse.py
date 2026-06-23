"""Parse a Python `transform` function out of an LLM response.

Mirrors poetiq's `_parse_code_from_llm` (arc_agi/solve_coding.py:208) so A0 is a
faithful baseline.
"""

from typing import Optional

from clarc.llm_parse import extract_block


def parse_transform_code(response: str) -> Optional[str]:
    """Return the first ```python ...``` block body verbatim, or None if absent."""
    return extract_block(response, "python", strip=False)
