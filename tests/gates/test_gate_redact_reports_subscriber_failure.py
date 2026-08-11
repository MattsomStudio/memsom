"""GATE for MS-27 -- the F-15 index purge swallowed a subscriber failure.

CONTROL-TESTED reasoning against memsom @ 9d165b1: `redact_node`'s F-15 index
purge was `try: from memsom.retrieval import retrieve ...; except Exception:
pass` -- content destroyed, postings survive, redact_node returns success, the
CLI prints "done". `SELECT term FROM postings WHERE node_id=N` IS the destroyed
document's vocabulary; the severity is the silence.

WRITTEN in Phase 4 (SECURITY-REMEDIATION.md Sec3.2, Sec2.3): no gate existed
for this finding before Phase 4 -- "there is no such helper function today,
this is a test, not a one-liner."

WHAT A FIX LOOKS LIKE
----------------------
`redact_node` emits "node_redacted" (kernel.events, which NEVER swallows)
instead of importing retrieval directly inside a bare `except Exception`. A
subscriber that raises is collected and reported through *purge_stats*
(`index_purge_failed`), visible to the CLI and to any other caller -- the
optional-degrade rule from PLAN.md Sec2.3: "never a third option" (silent
swallow was the third option).
"""

import memsom
from memsom.integrity import redact as memsom_redact
from memsom.kernel import events as memsom_events


def test_redact_reports_a_failing_index_subscriber(conn):
    """A subscriber that raises must be visible in purge_stats, not swallowed."""
    calls = []

    def _boom(conn, node_id):
        calls.append(node_id)
        raise RuntimeError("index backend unavailable")

    memsom_events.subscribe("node_redacted", _boom)
    try:
        nid = memsom.insert_node(conn, "content headed for redaction", "user")
        conn.commit()

        purge_stats = {}
        newly = memsom_redact.redact_node(
            conn, nid, "gate-test", purge_stats=purge_stats)

        assert newly == [nid], "the DB redaction itself must still succeed"
        assert calls == [nid], "precondition: the subscriber actually ran"
        assert purge_stats.get("index_purge_failed", 0) == 1, (
            f"a raising subscriber must be reported in purge_stats, got "
            f"{purge_stats!r} -- MS-27 is exactly this silence")

        row = conn.execute(
            "SELECT content, redacted FROM nodes WHERE id=?", (nid,)).fetchone()
        assert row == ("", 1), "payload must still be destroyed despite the failure"
    finally:
        memsom_events.unsubscribe("node_redacted", _boom)


def test_redact_does_not_report_when_nothing_fails(conn):
    """GREEN control: no subscriber failures -> index_purge_failed reports 0."""
    nid = memsom.insert_node(conn, "clean content, no failing subscriber", "user")
    conn.commit()
    purge_stats = {}
    memsom_redact.redact_node(conn, nid, "gate-test", purge_stats=purge_stats)
    assert purge_stats.get("index_purge_failed", 0) == 0, (
        "a clean redaction must not report a phantom index-purge failure")
