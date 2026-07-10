# Contributing to memsom

Thanks for wanting to help. memsom is a memory-integrity system — auditability is
the whole point — so contributions are held to that bar: deterministic by default,
tested, and provenance-clean.

## Contributor License Agreement (required)

Before your first pull request can merge, you must sign the
[Individual CLA](CLA.md). It's a one-time, comment-based signature:

1. Open your pull request.
2. The CLA bot comments with a link to the agreement and the exact sign-off phrase.
3. Reply on the PR with that phrase. The bot records it and turns the check green.

**Why a CLA and not just a DCO sign-off?** memsom is AGPL-3.0. The CLA grants the
maintainer the right to relicense contributions — including under a future
commercial license — so the project can sustain itself via a dual-license / open-core
model without having to hunt down every past contributor for permission. A DCO
certifies you *had the right* to submit; it does not grant relicensing rights. See
[CLA.md](CLA.md) §8 for the exact terms. You keep full ownership of your work.

Contributing on behalf of a company? The individual CLA covers you personally; if
your employer holds rights to your work, open an issue so we can sort out a
corporate CLA before you contribute.

## Dev setup

memsom is **stdlib-only** — there are no runtime dependencies to install.

```bash
git clone https://github.com/MattsomStudio/memsom.git
cd memsom
git config core.hooksPath .githooks   # enable the pre-push leak scan (once per clone)
python -m unittest discover -s . -p "test_*.py" -v
```

The `pre-push` hook scans the commits you're about to push for author-identifying
tokens and blocks the push if it finds one — the tree-scanning scrub gate can't see
leaks that live only in history, so this closes that gap. It's opt-in per clone (git
won't run a repo's committed hooks until you point `core.hooksPath` at them).

## Before you open a PR

- **Tests pass**, including deprecation-as-error: `python -W error::DeprecationWarning -m unittest discover -s . -p "test_*.py"`
- **The frozen core gate passes**: `python -m unittest test_memsom.py`
- **The broker self-check passes**: `python -m memsom.federation.broker --selfcheck`
- **New behavior ships with a test.** Determinism is a feature — if it isn't tested, it isn't done.
- Keep changes provenance-clean: no vendored code without noting its source and license (CLA §7).

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md). Include the
command you ran, what you expected, and what happened.
