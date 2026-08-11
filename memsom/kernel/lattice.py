"""memsom.kernel.lattice -- the integrity + confidentiality lattices (Phase 3).

RANK / NAME: the channel <-> integrity-rank mapping (Biba low-water-mark).
CONF_RANK / CONF_NAME: the confidentiality-rank mapping (Bell-LaPadula
high-water-mark), moved here from memsom.integrity.confid so
memsom.storage.schema's own duplicate _CONF_RANK dict could be dropped in favour
of importing this one.

parse_rank / parse_conf: the two lattices' shared "name-or-int-or-numeric-string
-> validated int" parse core, extracted out of the four near-identical copies
that had accumulated (memsom.storage.session._parse_floor,
memsom.integrity.policy._parse_floor, memsom.interface.ingest.channel_ceiling,
memsom.interface.mcp._mcp_channel_ceiling) -- storage.session's docstring said
outright it was "kept local so this module does not depend on the gate"; moving
the shared logic to kernel (rank 0, below both) removes the reason for the
duplication. Each call site keeps its own wrapping (default handling, error
message, ALLOW/DENY specials) -- only the core parse decision is shared.

meet / join: lattice meet (min) / join (max), moved here from
memsom.integrity.trust, which had the only implementation but is a poor home
for a primitive every layer above kernel may need without importing the
elevate()/audit-log machinery that also lives in trust.py.

Moved out of memsom/__init__.py in Phase 2, ahead of Phase 3's fuller
kernel/lattice.py: kernel/text.py's fmt_node needs NAME, and fmt_node is shared
by interface/cli.py AND integrity/redact.py, so NAME cannot stay above kernel
without creating an upward import from rank 0.
"""

RANK = {"endorsed": 3, "user": 2, "agent-derived": 1, "external": 0}
NAME = {3: "ENDORSED", 2: "USER", 1: "AGENT-DERIVED", 0: "EXTERNAL"}

CONF_RANK = {"public": 0, "internal": 1, "secret": 2, "topsecret": 3}
CONF_NAME = {0: "PUBLIC", 1: "INTERNAL", 2: "SECRET", 3: "TOPSECRET"}


def parse_rank(value):
    """Parse *value* into an int RANK 0..3, or None if it cannot be parsed.

    Accepts int 0..3, a numeric string '0'..'3', or a case-insensitive RANK
    name.  Never raises -- callers each own their own error message (or a
    default fallback) for the None case, which is why this only decides,
    it does not report.
    """
    if isinstance(value, bool):  # bool is an int subclass -- reject explicitly
        return None
    if isinstance(value, int):
        return value if 0 <= value <= 3 else None
    s = str(value).strip()
    if s.isdigit():
        n = int(s)
        return n if 0 <= n <= 3 else None
    if s.lower() in RANK:
        return RANK[s.lower()]
    return None


def _validate_label_int(v):
    """Raise ValueError if *v* is not an integer in 0..3."""
    if not isinstance(v, int):
        raise ValueError(f"label must be an int 0..3, got {v!r}")
    if v < 0 or v > 3:
        raise ValueError(f"label out of range 0..3: {v}")


def meet(a, b):
    """Lattice meet: min of two integrity labels.

    Both values must be integers in 0..3, otherwise ValueError is raised.
    """
    _validate_label_int(a)
    _validate_label_int(b)
    return min(a, b)


def join(a, b):
    """Lattice join: max of two integrity labels.

    Both values must be integers in 0..3, otherwise ValueError is raised.
    """
    _validate_label_int(a)
    _validate_label_int(b)
    return max(a, b)


def parse_min_integrity(val):
    """Accept an int or a RANK name string; return an integer label floor.

    Raises ValueError for unrecognised strings or out-of-range ints. Moved out
    of distill.py (Phase 7): reflex.py needed the identical logic and could
    not import distill (rank 5, above lifecycle's rank 4).
    """
    if isinstance(val, int):
        if val not in NAME:
            raise ValueError(f"integrity floor {val!r} out of range (0-3)")
        return val
    s = str(val).strip().lower()
    if s in RANK:
        return RANK[s]
    name_map = {v.lower(): k for k, v in NAME.items()}
    if s in name_map:
        return name_map[s]
    try:
        n = int(s)
    except ValueError:
        raise ValueError(f"unrecognised integrity name: {val!r}")
    if n not in NAME:
        raise ValueError(f"integrity floor {n!r} out of range (0-3)")
    return n


def parse_conf(value) -> int:
    """Accept int 0-3 or string name (case-insensitive). Raise ValueError otherwise."""
    if isinstance(value, int):
        if value not in CONF_NAME:
            raise ValueError(f"conf level {value!r} out of range 0-3")
        return value
    if isinstance(value, str):
        key = value.lower()
        if key in CONF_RANK:
            return CONF_RANK[key]
        # try numeric string
        try:
            n = int(key)
        except ValueError:
            raise ValueError(f"unknown conf level {value!r}") from None
        if n not in CONF_NAME:
            raise ValueError(f"conf level {n!r} out of range 0-3")
        return n
    raise ValueError(f"unrecognised conf level type: {type(value)}")
