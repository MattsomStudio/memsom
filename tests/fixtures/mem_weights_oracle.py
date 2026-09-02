#!/usr/bin/env python3
"""mem_weights — shared library for the memory "forgetting" layer.

VENDORED COPY (tests/fixtures/mem_weights_oracle.py). This is a read-only
snapshot of the operator's live ~/.claude/episodic/mem_weights.py, checked
into the repo so tests/test_memsom_forget.py::TestParity has a golden
oracle that does not depend on the operator's machine or home directory.
NEUTRALIZED: the only change from the original is the `MEM_DIR` assignment
below (was `find_mem_dir()`) -- see the comment there. Everything else,
including `compute()` and `DEFAULTS` (the two things TestParity actually
exercises), is byte-for-byte the original algorithm.

Models human forgetting as loss of *accessibility*, not *availability*: every
memory carries a weight that rises when it's referenced in sessions (retrieval
reinforcement) and decays with disuse. When a memory's weight falls below a
threshold it is DEMOTED — its index line is removed from MEMORY.md so it stops
costing always-loaded context — but the .md file stays on disk and stays
searchable via /recall (vault_index folds memory/*.md into the vault index).
Surface it again and the weight rebounds and it PROMOTES back. Nothing is ever
deleted by disuse; deletion is a separate, manual, liability-only action.

Storage model (single-writer everywhere, so Syncthing/rsync never conflicts):
  - canonical.json   SYNCED, PC-authored weekly. The source of truth for weights.
  - usage/<m>.jsonl  SYNCED, one writer per machine (append-only usage deltas).
  - mem_weights.db   LOCAL, excluded by *.db .stignore. Inspection cache only.
  - mem_scan_state   LOCAL, per-machine FTS watermark.

A "demote" is NOT a file move (the brain's rsync has no --delete, so a move
would duplicate the file on the other machine). Cold = "absent from MEMORY.md".

This module is pure-ish library code: path discovery, frontmatter parsing,
canonical load/save, pinning rules, and the weight-update computation. The
scan job (mem_usage_scan.py) and the authority job (mem_reconcile.py) import it.
"""
import json, math, os, re, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

EP = Path.home() / ".claude" / "episodic"

# ── defaults (operator-tuned while watching dry-runs) ─────────────────────────
#
# Two-number model (Bjork's New Theory of Disuse):
#   RS = retrieval strength — accessibility. Decays with disuse (slowed by SS),
#        drives hot/cold. This is the old `weight`, generalized.
#   SS = storage strength — durability / "encoding strength". Does NOT decay;
#        grows with SPACED retrieval; set at birth by salience (affect-driven).
#
# The model REDUCES EXACTLY to the old single-weight model when SS == 0:
#   decay_base == old decay, rs_gain == old gain, rs_seed == old seed. So a
#   memory with no usage history (count 0, no salience) behaves identically; only
#   memories with accumulated use or birth-salience gain extra durability. This
#   is what makes the migration safe — it can only ever PROTECT a memory from
#   demotion, never demote one the old model would have kept.
DEFAULTS = {
    # RS (accessibility) — same scale/thresholds as the old weight
    "rs_cap": 1.0,         # max retrieval strength
    "rs_seed": 1.0,        # new memory is born accessible (hot)  [= old seed]
    "decay_base": 0.5,     # weekly RS decay at SS=0              [= old decay]
    "rs_gain": 0.15,       # base RS added per retrieval at SS=0  [= old gain]
    "demote_below": 0.2,   # hot -> cold when RS drops under this ...
    "grace_days": 21,      # ... AND the memory is at least this old
    "promote_at": 0.5,     # cold -> hot when RS climbs to this (hysteresis)
    # SS (durability) — the new axis
    "ss_floor": 0.0,       # baseline durability (0 => exact old behaviour)
    "ss_cap": 3.0,         # max storage strength (salience/spacing ceiling)
    "ss_gain": 0.1,        # SS added per retrieval, scaled by desirable difficulty
    "ss_decay_k": 1.0,     # how much SS slows RS decay (higher SS => slower decay)
    "ss_mig_k": 0.5,       # legacy SS bootstrap: ss_floor + k*log1p(count) — log,
                           # not linear, so durability has diminishing returns and
                           # a count=220 memory isn't 200x as durable as a count=1.
    "salience_default": 0.0,    # SS₀ for a memory saved without a salience tag
}

# ── path discovery (portable Mac/PC; project dir name differs per machine) ────
def find_mem_dir():
    """Locate the live memory dir (the one holding MEMORY.md).

    Portable Mac/PC (the project-dir name differs per machine), and robust against a
    STRAY store: a Claude session launched with an odd cwd (e.g. a scheduled task at
    C:\\WINDOWS\\system32) creates its own project dir with a near-empty MEMORY.md stub.
    The old code returned the alphabetically-FIRST glob hit, which is correct only by
    luck ("Users" < "Windows") — a stray that sorted first would silently win. That is
    exactly how the 2026-07-12 consolidation audit read a 2-file stub and reported ~165
    false orphans. Pick the store with the MOST memory files instead: the real store has
    ~167, any cwd-derived stub has 1-2. Deterministic and immune to junk dirs.
    """
    env = os.environ.get("EPISODIC_MEM_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    candidates = [p.parent for p in
                  (Path.home() / ".claude" / "projects").glob("*/memory/MEMORY.md")]
    if candidates:
        return max(candidates, key=lambda d: sum(1 for _ in d.glob("*.md")))
    # fall back to the known Mac path so callers get a clear error if absent
    return Path.home() / ".claude" / "projects" / "-Users-operator" / "memory"


# NEUTRALIZED for this vendored test fixture: the original line here was
# `MEM_DIR = find_mem_dir()`, which globs Path.home()/.claude/projects/*/memory
# at IMPORT time. TestParity only exercises the pure functions compute() and
# DEFAULTS (which read no path at all), so this fixture must never perform that
# live-filesystem probe -- not even a read -- regardless of whether the test
# process's HOME/USERPROFILE happen to be redirected when this module is
# imported. Point MEM_DIR at a path that cannot exist instead of scanning the
# operator's real ~/.claude tree; every path below is a lazy, disk-free
# Path join off it, so nothing else in this module changes behaviour unless a
# caller actually dereferences one of them (which TestParity does not).
MEM_DIR = Path("nonexistent-mem-weights-oracle-fixture-dir")
MEMORY_MD = MEM_DIR / "MEMORY.md"
WEIGHTS_DIR = MEM_DIR / ".weights"
CANONICAL = WEIGHTS_DIR / "canonical.json"
USAGE_DIR = WEIGHTS_DIR / "usage"
WEIGHTS_DB = EP / "mem_weights.db"
SCAN_STATE = EP / "mem_scan_state.json"
# Reconsolidation input: the weekly consolidation sweep writes the set of
# contradicted/stalled memories here (PC-only; absent on the Mac, which doesn't
# run the sweep or reconcile). compute() suppresses reinforcement for these.
STALE_FILE = Path.home() / ".claude" / "consolidation" / "stale.json"


def machine_id():
    """'mac' or 'pc' — used to name this machine's usage delta file."""
    m = os.environ.get("EPISODIC_MACHINE")
    if m:
        return m
    return "mac" if sys.platform == "darwin" else "pc"


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s):
    if not s:
        return None
    try:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ── memory files + frontmatter ────────────────────────────────────────────────
FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# MEMORY.md index links: "- [Title](user_adhd.md) — hook"
LINK_RE = re.compile(r"\]\(([^)]+\.md)\)")
PIN_TYPES = {"user", "feedback"}
PIN_PREFIXES = ("user_", "feedback_")


def parse_frontmatter(text):
    """Return {key: value} from a markdown file's YAML-ish frontmatter (flat)."""
    m = FM_RE.match(text)
    out = {}
    if not m:
        return out
    for line in m.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def is_pinned(stem, fm):
    """Identity/feedback memories never demote — losing them would forget who
    the operator is. Pin by frontmatter type, explicit pin:true, or filename class."""
    if str(fm.get("pin", "")).lower() in ("true", "1", "yes"):
        return True
    if fm.get("type", "").lower() in PIN_TYPES:
        return True
    return stem.startswith(PIN_PREFIXES)


PROJECTS_DIR = MEM_DIR / "projects"          # project_*.md live here (memsom split)
PROJECTS_INDEX = PROJECTS_DIR / "INDEX.md"    # generated by memsom bridge-render
INDEX_NAMES = {"MEMORY.md", "INDEX.md"}


def hot_set():
    """Filenames currently linked in MEMORY.md OR projects/INDEX.md = indexed.
    INDEX.md links are relative to projects/ ("x.md", "../x.md", "<slug>/x.md")
    -> basename."""
    out = set()
    if MEMORY_MD.exists():
        out |= set(LINK_RE.findall(MEMORY_MD.read_text(encoding="utf-8", errors="replace")))
    if PROJECTS_INDEX.exists():
        out |= {Path(m).name for m in
                LINK_RE.findall(PROJECTS_INDEX.read_text(encoding="utf-8", errors="replace"))}
    return out


def memory_files():
    """Every memory .md: the flat dir, projects/, and projects/<slug>/ (depth 2,
    no deeper), minus the generated indexes. Same walk as memsom's
    bridge_import.iter_memory_files."""
    files = [p for p in MEM_DIR.glob("*.md") if p.name not in INDEX_NAMES]
    if PROJECTS_DIR.is_dir():
        files += [p for p in PROJECTS_DIR.glob("*.md") if p.name not in INDEX_NAMES]
        for d in sorted(x for x in PROJECTS_DIR.iterdir() if x.is_dir()):
            files += [p for p in d.glob("*.md") if p.name not in INDEX_NAMES]
    return sorted(files, key=lambda p: (p.name, str(p)))


VALID_STEM = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def iter_memories():
    """Yield a dict per real memory .md file (skips MEMORY.md and .weights/).
    Skips non-slug filenames — chiefly Syncthing 'Conflicted copy' dupes, which
    aren't real memories and must not be tracked or promoted."""
    hot = hot_set()
    for p in memory_files():
        if not VALID_STEM.match(p.stem):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        fm = parse_frontmatter(text)
        stem = p.stem
        yield {
            "stem": stem,
            "file": p.name,
            "path": p,
            "name": fm.get("name", stem),
            "type": fm.get("type", ""),
            "pinned": is_pinned(stem, fm),
            "tier": "hot" if p.name in hot else "cold",
            "terms": search_terms(stem, fm),
            "salience": fm.get("salience"),  # 0..1 birth encoding strength (-> SS₀)
        }


# ── usage-signal term extraction (mirrors recall.py's keyword logic) ──────────
# The operator's name is in nearly every transcript — it carries no signal about
# which memory was used, so it's dropped from match terms.
_STOP = set("""the a an and or of to in on for with that this
is are was were be been being it its his her their them they we you your my our""".split())
_WORD = re.compile(r"[a-z0-9]{3,}")


def search_terms(stem, fm):
    """Distinctive terms for this memory, drawn from filename + name + desc."""
    blob = " ".join([stem.replace("_", " "), fm.get("name", ""), fm.get("description", "")]).lower()
    seen, out = set(), []
    for w in _WORD.findall(blob):
        if w not in _STOP and not w.isdigit() and w not in seen:
            seen.add(w)
            out.append(w)
    return out


# ── canonical snapshot I/O ────────────────────────────────────────────────────
def load_canonical():
    if CANONICAL.exists():
        try:
            data = json.loads(CANONICAL.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    params = dict(DEFAULTS)
    params.update(data.get("params", {}))
    return {
        "version": data.get("version", 1),
        "updated": data.get("updated"),
        "params": params,
        "memories": data.get("memories", {}),
    }


def save_canonical(canon):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    canon["updated"] = now_iso()
    tmp = CANONICAL.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(canon, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(CANONICAL)


# ── usage deltas ──────────────────────────────────────────────────────────────
def append_usage(hits, machine=None):
    """Append {ts, stem, hits} lines to this machine's usage delta file."""
    machine = machine or machine_id()
    USAGE_DIR.mkdir(parents=True, exist_ok=True)
    f = USAGE_DIR / f"{machine}.jsonl"
    ts = now_iso()
    with f.open("a", encoding="utf-8") as fh:
        for stem, n in hits.items():
            if n:
                fh.write(json.dumps({"ts": ts, "stem": stem, "hits": int(n)}) + "\n")


def read_all_deltas(after_ts=None):
    """Sum hits per memory across every machine's usage file.
    after_ts (iso) lets reconcile ignore deltas already folded in."""
    totals = {}
    if not USAGE_DIR.is_dir():
        return totals
    cutoff = _parse_iso(after_ts) if after_ts else None
    for f in sorted(USAGE_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if cutoff:
                t = _parse_iso(rec.get("ts"))
                if t and t <= cutoff:
                    continue
            totals[rec["stem"]] = totals.get(rec["stem"], 0) + int(rec.get("hits", 0))
    return totals


def read_all_events(after_ts=None):
    """Like read_all_deltas, but PRESERVES per-event timestamps instead of
    summing. Returns {stem: [(ts_iso, hits), …]} in chronological order.

    Spacing needs the temporal distribution of reinforcement, which read_all_deltas
    threw away. The raw `ts` was always written by append_usage — this just stops
    discarding it. Same timestamp-cutoff dedup as read_all_deltas (events at or
    before the last canonical update were already folded)."""
    out = {}
    if not USAGE_DIR.is_dir():
        return out
    cutoff = _parse_iso(after_ts) if after_ts else None
    for f in sorted(USAGE_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ts = rec.get("ts")
            if cutoff:
                t = _parse_iso(ts)
                if t and t <= cutoff:
                    continue
            n = int(rec.get("hits", 0))
            if n:
                out.setdefault(rec["stem"], []).append((ts, n))
    for stem in out:
        out[stem].sort(key=lambda e: e[0] or "")
    return out


def load_stale(max_age_days=10, path=None):
    """Return the set of memory stems the last consolidation sweep flagged as
    contradicted/stalled (reinforcement should be suppressed for these).

    Freshness-gated and FAIL-SAFE TOWARD RESUMING: a stale.json whose `updated`
    is missing, unparseable, or older than max_age_days returns the EMPTY set —
    never keep suppressing forever on a bad/old timestamp (that's the gate's whole
    purpose; the next sweep is the source of truth). max_age_days=10 = one weekly
    cadence plus slack for a single missed Sunday run. Missing file, bad JSON, or
    no consolidation dir (e.g. the Mac) -> empty set."""
    f = Path(path) if path else STALE_FILE
    if not f.exists():
        return set()
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return set()
    upd = _parse_iso(data.get("updated"))
    if upd is None:                                   # unparseable/missing -> resume
        return set()
    if (datetime.now(timezone.utc) - upd).days > max_age_days:
        return set()
    return set(data.get("stale", {}).keys())


# ── the core weight computation (pure; shared by reconcile + its dry-run) ─────
def _salience_of(m, p):
    """Birth salience in [0,1] from the memory's frontmatter `salience` tag.
    Missing/unparseable -> salience_default (mundane). /saveall writes this as
    the affect>surprise>source blend; compute() only reads the final number."""
    raw = m.get("salience")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return float(p.get("salience_default", 0.0))


def _seed_rs_ss(st, m, p):
    """Resolve (RS, SS) for a memory at the start of a pass.

    Three cases, in order:
      - already two-number: use stored rs/ss.
      - legacy single-weight: RS = old weight; SS bootstrapped from prior `count`
        (often-used memories earn durability) — behaviour-preserving since SS=0
        when count=0.
      - brand-new (unseen by canonical): RS = rs_seed; SS₀ from birth salience.
    """
    cap = p["rs_cap"] or 1.0
    if "rs" in st:
        rs, ss = float(st["rs"]), float(st.get("ss", p["ss_floor"]))
    elif "weight" in st:  # migrate a pre-two-number memory
        rs = float(st["weight"])
        cnt = max(0, int(st.get("count", 0)))     # log1p guard on a bad count
        ss = p["ss_floor"] + p["ss_mig_k"] * math.log1p(cnt)
    else:                 # brand-new: SS₀ from birth salience
        rs = float(p["rs_seed"])
        ss = p["ss_floor"] + (p["ss_cap"] - p["ss_floor"]) * _salience_of(m, p)
    # clamp into range so a migrated weight>1 or a misconfigured seed can't push
    # RS over cap (-> negative-SS term) or SS out of [0, ss_cap].
    return min(rs, cap), max(0.0, min(ss, p["ss_cap"]))


def _decay_rs(rs, ss, t0, t1, p):
    """Decay RS over [t0, t1], with the rate SLOWED by storage strength.
    At SS=0 this is exactly the old weekly decay; high SS lengthens the half-life."""
    if not (t0 and t1) or t1 <= t0:
        return rs
    dt_weeks = (t1 - t0).total_seconds() / (7 * 86400.0)
    return rs * (p["decay_base"] ** (dt_weeks / (1.0 + p["ss_decay_k"] * ss)))


def compute(canon, mems, events, now=None, stale=None):
    """Apply one decay+reinforce pass over the two-number (RS, SS) model and
    decide promote/demote actions.

    *events* is {stem: [(ts_iso, hits), …]} (from read_all_events) so spacing is
    visible: massed hits share a ts and barely grow SS; spaced hits let RS decay
    between them and grow SS hard ("desirable difficulty"). Pure — touches no disk.

    *stale* (set of stems, from the consolidation sweep) is the RECONSOLIDATION
    gate: a memory the sweep flagged as contradicted gets its reinforcement
    SUPPRESSED — its events are ignored, so it decays at the natural rate instead
    of being propped up by recurring-topic hits ("stop strengthening the wrong
    fact"). No penalty, fully reversible: drops off the list once Matt fixes it.

    Returns (new_memories_dict, actions); actions carry `rs` (the tier is decided
    on retrieval strength). Reduces to the old model when every SS is 0 and to
    Phase-1 behaviour when *stale* is empty/None.
    """
    now = now or datetime.now(timezone.utc)
    stale = stale or set()
    p = canon["params"]
    rs_cap = p["rs_cap"] or 1.0      # guard a hand-tuned rs_cap=0 (div-by-zero)
    prev = canon["memories"]
    new = {}
    actions = []
    by_stem = {m["stem"]: m for m in mems}

    for stem, m in by_stem.items():
        st = dict(prev.get(stem, {}))
        rs, ss = _seed_rs_ss(st, m, p)
        count = int(st.get("count", 0))
        first_seen = st.get("first_seen") or now_iso()
        last_used = st.get("last_used")
        last_t = _parse_iso(last_used) or _parse_iso(first_seen) or now

        # Reconsolidation: a contradicted memory is not reinforced this pass —
        # treat it as having no events so it just decays.
        stem_events = [] if stem in stale else events.get(stem, [])
        for ts_iso, hits in sorted(stem_events, key=lambda e: e[0] or ""):
            t = _parse_iso(ts_iso) or now
            if t > now:                            # clock-skew guard: a future ts
                t = now                            # (2-machine mesh) must not freeze
                                                   # decay or persist a future last_used
            rs = _decay_rs(rs, ss, last_t, t, p)   # decay up to this retrieval
            for _ in range(int(hits)):
                # desirable difficulty: retrieving when RS is LOW cements SS hard;
                # massed reps (RS already high) add almost nothing to durability.
                # max(0,..): a migrated RS>rs_cap could push (1-RS/cap) negative;
                # never let SS go negative (it also guards _decay_rs's divisor).
                ss = max(0.0, min(p["ss_cap"], ss + p["ss_gain"] * (1.0 - rs / rs_cap)))
                rs = min(rs_cap, rs + p["rs_gain"] * (1.0 + math.log1p(ss)))
            count += int(hits)
            last_t = t
            last_used = t.strftime("%Y-%m-%dT%H:%M:%SZ")
        rs = _decay_rs(rs, ss, last_t, now, p)     # decay tail up to now
        rs = round(rs, 4)
        ss = round(ss, 4)

        current = m["tier"]
        had_stash = bool(st.get("index_line"))     # was it demoted BY this system?
        age_days = (now - (_parse_iso(first_seen) or now)).days
        if m["pinned"]:
            target = "hot"                         # identity/feedback: never forget
        elif current == "hot" and rs < p["demote_below"] and age_days >= p["grace_days"]:
            target = "cold"
        elif current == "cold" and had_stash and rs >= p["promote_at"]:
            # only re-promote what WE demoted; never auto-index files that were
            # deliberately left out of MEMORY.md (that stays /saveall's job)
            target = "hot"
        else:
            target = current

        if target != current:
            actions.append({
                "stem": stem, "file": m["file"], "from": current, "to": target,
                "rs": rs, "ss": ss, "pinned": m["pinned"],
                "reason": ("RS decayed below %.2f (age %dd)" % (p["demote_below"], age_days)
                           if target == "cold" else
                           "RS recalled back above %.2f" % p["promote_at"]),
            })

        new[stem] = {
            "rs": rs, "ss": ss, "count": count, "first_seen": first_seen,
            "last_used": last_used, "tier": target, "pinned": m["pinned"],
        }
        # carry the stashed index line forward while a memory stays cold, so a
        # later promote can restore it. mem_reconcile clears it on promote.
        if target == "cold" and st.get("index_line"):
            new[stem]["index_line"] = st["index_line"]
            new[stem]["index_section"] = st.get("index_section")
    return new, actions


# ── inspection cache (honors the SQLite storage the operator picked) ─────────
def sync_db(canon):
    """Mirror canonical weights into a local SQLite table for easy querying.
    Pure inspection — not a source of truth, and never synced (*.db ignored)."""
    db = sqlite3.connect(WEIGHTS_DB)
    # drop any old single-weight table so the schema matches the two-number model
    db.execute("DROP TABLE IF EXISTS weights")
    db.execute("""CREATE TABLE weights (
        stem TEXT PRIMARY KEY, rs REAL, ss REAL, count INTEGER,
        last_used TEXT, first_seen TEXT, tier TEXT, pinned INTEGER)""")
    for stem, st in canon["memories"].items():
        db.execute("INSERT INTO weights VALUES (?,?,?,?,?,?,?,?)",
                   (stem, st.get("rs"), st.get("ss"), st.get("count"),
                    st.get("last_used"), st.get("first_seen"), st.get("tier"),
                    int(bool(st.get("pinned")))))
    db.commit()
    db.close()


if __name__ == "__main__":
    # quick self-report: where things resolve + current memory inventory
    mems = list(iter_memories())
    hot = [m for m in mems if m["tier"] == "hot"]
    pinned = [m for m in mems if m["pinned"]]
    print(f"MEM_DIR     {MEM_DIR}")
    print(f"MEMORY.md   {'present' if MEMORY_MD.exists() else 'MISSING'} ({MEMORY_MD.stat().st_size if MEMORY_MD.exists() else 0} bytes)")
    print(f"canonical   {CANONICAL} ({'exists' if CANONICAL.exists() else 'not yet'})")
    print(f"machine     {machine_id()}")
    print(f"memories    {len(mems)} total | {len(hot)} hot (in MEMORY.md) | {len(pinned)} pinned")
