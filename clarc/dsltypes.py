"""Types for the composable DSL (M5).

A linear pipeline threads a single VALUE whose type is one of `Ty`. Whole-grid
primitives are `GRID -> GRID` leaves; the object layer bridges `GRID -> SELECTION`
(`objects`), transforms `SELECTION -> SELECTION` (filter/map/recolor), and renders
back `SELECTION -> GRID` (`render`). The typechecker (in dslparse) threads `Ty`
from GRID and rejects ill-typed compositions — the syntactic half of the sketch's
consistency `T1 ◁ T_I`; the abstract-contract half (`C1 ⇒ C_I`) lives in z3.

The runtime carriers (`Obj`, `Selection`) are what object primitives' `apply`
functions operate on. An `Obj` keeps a FULL-GRID boolean mask (not a cropped
sprite) so geometry/recolor compose without re-registration and `render` is just
painting masks onto a background canvas.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Ty(str, Enum):
    GRID = "Grid"
    SELECTION = "Selection"   # an ordered list of objects over a fixed canvas


@dataclass
class Obj:
    """One connected component, carried as a full-canvas mask + summary attrs."""

    mask: np.ndarray          # bool, shape == canvas (h, w)
    color: int                # dominant (most-frequent) non-bg color of the object
    size: int                 # number of cells (== mask.sum())

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        rows = np.flatnonzero(self.mask.any(axis=1))
        cols = np.flatnonzero(self.mask.any(axis=0))
        if len(rows) == 0:
            return (0, 0, 0, 0)
        return (int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1]))


@dataclass
class Selection:
    """Objects over a fixed canvas; `grid` holds the original colors, `bg` the
    background to paint where no object covers a cell when rendering."""

    objs: list[Obj]
    grid: np.ndarray          # original grid (object colors read from here)
    bg: int

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.grid.shape[0]), int(self.grid.shape[1]))
