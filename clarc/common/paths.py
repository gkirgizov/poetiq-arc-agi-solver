"""Canonical filesystem anchors for clarc.

Loaders resolve data files against these constants rather than each module's own
``__file__``, so moving a loader between subpackages never breaks a relative path.
"""
import os

# this file is clarc/common/paths.py → the clarc package root is two levels up.
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the repo root holds the gitignored ``data/`` ARC corpus.
REPO_ROOT = os.path.dirname(PKG_ROOT)
DATA_DIR = os.path.join(REPO_ROOT, "data")
