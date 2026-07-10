"""`python -m memsom` runs the core demo CLI (seed/ask/explain/revoke/dump).

Restores the pre-package `python memsom.py <cmd>` entry point now that core
lives in memsom/__init__.py. The full-stack CLI is `memsom` / memsom.interface.cli.
"""
from memsom import main

if __name__ == "__main__":
    main()
