"""memsom.kernel.lattice -- the integrity channel lattice (Biba low-water-mark).

RANK / NAME are the channel <-> integrity-rank mapping every module in the
package shares. Pure data, zero imports -- the floor of the dependency graph.

Moved out of memsom/__init__.py in Phase 2, ahead of Phase 3's fuller
kernel/lattice.py (which folds in CONF_RANK/CONF_NAME/parse_conf and the
meet/join helpers): kernel/text.py's fmt_node needs NAME, and fmt_node is
shared by interface/cli.py AND integrity/redact.py, so NAME cannot stay above
kernel without creating an upward import from rank 0.
"""

RANK = {"endorsed": 3, "user": 2, "agent-derived": 1, "external": 0}
NAME = {3: "ENDORSED", 2: "USER", 1: "AGENT-DERIVED", 0: "EXTERNAL"}
