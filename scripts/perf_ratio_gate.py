#!/usr/bin/env python3
"""perf_ratio_gate -- PLAN.md Sec6 Phase 11: "Windows full-suite wall time
<= 3x Linux, or a written residual with a number in it."

A copy-confined single-OS session (this refactor's own execution constraint,
same one named in `_meta/AMENDMENTS.md` A-17/A-18/A-19) cannot itself run the
suite on two different operating systems to compute a ratio -- there is only
one OS available locally. Asserting a cross-OS number from a single-OS box
would be exactly the guessed-not-measured failure this repo's own diagnostic
discipline exists to catch. So this gate does not run the suite itself: it
is invoked TWICE by CI, once per OS, and does the actual comparison only
where both real numbers exist -- the GitHub Actions matrix.

USAGE
  python scripts/perf_ratio_gate.py --write <path> <elapsed_seconds>
      record one OS's wall-clock full-suite time (int/float seconds)

  python scripts/perf_ratio_gate.py --check --linux <path> --windows <path>
      [--max-ratio 3.0]
      compare two recorded times; exit 0 if windows <= max_ratio * linux,
      exit 1 (loud, with the actual ratio) otherwise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def write(path: str, seconds: float) -> None:
    Path(path).write_text(f"{seconds}\n", encoding="utf-8")


def _read_seconds(path: str) -> float:
    text = Path(path).read_text(encoding="utf-8").strip()
    return float(text)


def check(linux_path: str, windows_path: str, max_ratio: float) -> int:
    linux_s = _read_seconds(linux_path)
    windows_s = _read_seconds(windows_path)
    ratio = windows_s / linux_s if linux_s else float("inf")
    print(f"linux full-suite wall time:   {linux_s:.1f}s")
    print(f"windows full-suite wall time: {windows_s:.1f}s")
    print(f"ratio (windows/linux):        {ratio:.2f}x  (max allowed {max_ratio:.2f}x)")
    if ratio > max_ratio:
        print(f"FAIL: windows is {ratio:.2f}x linux, exceeds the {max_ratio:.2f}x ceiling "
              "(PLAN.md Sec6 Phase 11)", file=sys.stderr)
        return 1
    print("PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", metavar="PATH", default=None)
    ap.add_argument("seconds", nargs="?", type=float, default=None,
                    help="elapsed seconds (required with --write)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--linux", metavar="PATH", default=None)
    ap.add_argument("--windows", metavar="PATH", default=None)
    ap.add_argument("--max-ratio", type=float, default=3.0)
    args = ap.parse_args()

    if args.write:
        if args.seconds is None:
            ap.error("--write requires a seconds value")
        write(args.write, args.seconds)
        print(f"wrote {args.write}: {args.seconds}s")
        return 0

    if args.check:
        if not args.linux or not args.windows:
            ap.error("--check requires --linux and --windows")
        return check(args.linux, args.windows, args.max_ratio)

    ap.error("specify --write <path> <seconds> or --check --linux <path> --windows <path>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
