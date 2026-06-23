"""Extract fenced code blocks from an LLM response.

One place for the ```python``` (transform code, first block, verbatim) and ```dsl```
(pipeline, last block, stripped) extraction so the regex lives in a single spot.
"""

from __future__ import annotations

import re


def extract_block(text: str, lang: str, *, last: bool = False, strip: bool = True) -> str | None:
    """Return the body of a ```<lang> … ``` fenced block, or ``None`` if absent.

    ``last`` picks the final matching block (the DSL convention) instead of the first
    (the python convention). ``strip`` trims surrounding whitespace; pass ``strip=False``
    to preserve the block verbatim (the faithful-baseline python path does this).
    """
    if not text:
        return None
    blocks = re.findall(rf"```{lang}\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        return None
    body = blocks[-1] if last else blocks[0]
    return body.strip() if strip else body
