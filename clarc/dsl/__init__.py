"""clarc.dsl — the typed transformation DSL, its z3 contracts, and synthesis.

`core` is the primitive REGISTRY + pipeline runner/search; `absdomain`/`smt`/`clauses`
are the SMT side; `parse` reads LLM-emitted pipelines; `induce`/`prim_library` extend
the DSL. Depends on `clarc.objects`, `clarc.contracts`, `clarc.common`.
"""
