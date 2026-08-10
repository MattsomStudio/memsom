# Makes tests/ a package so unittest discover recurses into it.
#
# ALSO the isolation hook for `python -m unittest discover` (Phase 0, A5.1):
# unittest never reads conftest.py, so tests/conftest.py's pytest-only pin
# does nothing for CI's actual test-suite step. Importing tests._isolation
# here pins MEMDAG_HOME/MEMDAG_DB to a throwaway dir as that module's own
# import-time side effect — before any test_*.py module in this package can
# import memsom. See tests/_isolation.py for why it has to happen at import
# time rather than in a fixture or a setUp().
from tests import _isolation  # noqa: F401
