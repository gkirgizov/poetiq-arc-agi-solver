"""clarc — Contract-Learning ARC solver.

A CDCL-style loop in program-generation space, layered on the poetiq harness.
The LLM (driven via the Claude Code CLI) is the generator/abductive proposer;
Python is the sound verifier of contracts learned from conflicts.
"""

__version__ = "0.1.0"
