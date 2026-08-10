# The known-good wheel (Matt's Q5, PLAN.md Phase 0)

Phase 1 changes behaviour on the live editable tree, so this escape hatch has
to exist before Phase 1 touches anything, not before Phase 2 (A5.5).

## What was built

A wheel from this phase's own last-green commit, stored **outside** this
repo's tree:

```
C:\Users\usr9f2\memsom-refactor\known-good-wheel\memsom-0.2.0-py3-none-any.whl
```

Built with `python -m pip wheel . --no-deps -w <that dir>` from
`memsom-refactor-work` at commit `7862fa8` (Phase 0's starting HEAD). Rebuild
before each later phase's exit gate closes, from that phase's own last-green
commit, so the escape hatch always points one phase behind rather than at a
fixed, increasingly stale baseline.

## The one-command switch

Freeze onto the known-good wheel:

```
pip install --force-reinstall C:\Users\usr9f2\memsom-refactor\known-good-wheel\memsom-0.2.0-py3-none-any.whl
```

Return to the editable working tree:

```
pip install -e .
```

## The one-line "which is live" probe (A9.7)

```
python scripts/live_probe.py
```

## MEASURED FINDING while building this, worth carrying forward

`scripts/live_probe.py` reads `importlib.metadata`'s `direct_url.json` for
the globally installed `memsom` distribution, not the `memsom` a given
script actually imports. On this machine that global editable pointer
resolves to **`C:\Users\usr9f2\memsom`, the LIVE repo** -- not this copy, and
not any wheel. Running the probe from inside `memsom-refactor-work` still
correctly shows `memsom-refactor-work\memsom\__init__.py` as the module a
plain `import memsom` picks up (Python puts the script's own directory /
cwd ahead of site-packages in `sys.path`, MEASURED), so every test run in
this phase genuinely exercised the copy, not the live repo.

The risk the probe exists to catch is real, though: any invocation of a bare
`memsom` console-script, or an `import memsom` from a process whose cwd is
NOT this copy's root, resolves through the global editable pointer straight
to the live repo's code. That is true today, independent of anything this
refactor does, and worth knowing before Phase 1 starts changing behaviour
under that same global pointer.
