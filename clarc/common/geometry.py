"""Shared grid geometry.

The connected-component labeling used by BOTH the abstract domain (``absdomain``,
which summarizes objects into σ) and the concrete object recognizer (``objects``,
which builds rich ``Object`` records). One labeling primitive, two consumers.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def label_components(mask, conn: int = 1):
    """Label connected components of a boolean ``mask``.

    ``conn=1`` → 4-connectivity, ``conn=2`` → 8-connectivity. Returns ``(labels, n)``
    exactly as ``scipy.ndimage.label``.
    """
    return ndimage.label(np.asarray(mask, dtype=bool),
                         structure=ndimage.generate_binary_structure(2, conn))
