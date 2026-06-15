#!/usr/bin/env python3
"""Tests for memdag_redact — run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memdag_redact.py \
    -t <repo> -v
"""

import contextlib, io, os, tempfile, unittest, warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_redact

HERE = Path(__file__).resolve().parent


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()
        memdag_redact.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memdag.insert_node(self.conn, content, channel, memdag.RANK[channel])

    def run_main(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            memdag_redact.main(list(argv))
        return buf.getvalue()


class TestContentDestroyedShapeIntact(Base):
    """test 1: redact a derived node — content gone, shape + metadata intact."""

    def test_content_destroyed_shape_intact(self):
        a = self.add("endorsed source content text here", "endorsed")
        b, _ = memdag.derive_node(self.conn, "derived content from a", [a])

        # Pre-capture b's metadata and edge count
        node_b_pre = memdag.get_node(self.conn, b)
        pre_channel = node_b_pre["channel"]
        pre_label = node_b_pre["label"]
        pre_created_at = node_b_pre["created_at"]
        pre_edge_count = self.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE child=? OR parent=?", (b, b)).fetchone()[0]

        memdag_redact.redact_node(self.conn, b, "pii")

        node_b = memdag.get_node(self.conn, b)
        # Payload destroyed
        self.assertEqual(node_b["content"], "")
        # Redaction flags set
        row = self.conn.execute(
            "SELECT redacted, redacted_at, redact_reason FROM nodes WHERE id=?",
            (b,)).fetchone()
        self.assertEqual(row[0], 1)                  # redacted=1
        self.assertTrue(row[1])                      # redacted_at is truthy ISO text
        self.assertEqual(row[2], "pii")              # redact_reason
        # Channel / label / created_at unchanged
        self.assertEqual(node_b["channel"], pre_channel)
        self.assertEqual(node_b["label"], pre_label)
        self.assertEqual(node_b["created_at"], pre_created_at)
        # NOT tombstoned
        self.assertEqual(node_b["tombstoned"], 0)
        # Edges unchanged
        post_edge_count = self.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE child=? OR parent=?", (b, b)).fetchone()[0]
        self.assertEqual(post_edge_count, pre_edge_count)


class TestDescribeShowsRedactedAndWalks(Base):
    """test 2: describe shows [REDACTED ...] marker; parents_of still walks."""

    def test_describe_shows_redacted_and_walks(self):
        a = self.add("endorsed source line for walking test here.", "endorsed")
        b, _ = memdag.derive_node(self.conn, "derived from endorsed source", [a])

        memdag_redact.redact_node(self.conn, a, "privacy-violation")

        lines = memdag_redact.describe(self.conn, a)
        combined = "\n".join(lines)
        self.assertIn("[REDACTED", combined)
        self.assertIn("privacy-violation", combined)

        # The child node can still walk to its parent (shape intact)
        parents = memdag.parents_of(self.conn, b)
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0][0], a)


class TestComposeNeverLeaks(Base):
    """test 3: compose over live_sources cannot surface redacted content."""

    def test_compose_never_leaks(self):
        secret = "SECRET-TOKEN-XYZ nebula configuration guidance."
        a = self.add(secret, "endorsed")

        # (a) Before redaction: live_unredacted_sources includes a
        sources_before = memdag_redact.live_unredacted_sources(self.conn)
        ids_before = [r[0] for r in sources_before]
        self.assertIn(a, ids_before)

        memdag_redact.redact_node(self.conn, a, "secret")

        # (a) After redaction: live_unredacted_sources excludes a
        sources_after = memdag_redact.live_unredacted_sources(self.conn)
        ids_after = [r[0] for r in sources_after]
        self.assertNotIn(a, ids_after)

        # (b) compose over memdag.live_sources (which still includes the row)
        # must not surface the secret text because content is now ''
        all_sources = memdag.live_sources(self.conn)
        text, _used = memdag.compose("nebula configuration", all_sources)
        # text may be None if no source contributed anything (acceptable)
        if text is not None:
            self.assertNotIn("SECRET-TOKEN-XYZ", text)


class TestRedactIsNotRevoke(Base):
    """test 4: redact != revoke; both directions verified."""

    def test_redact_is_not_revoke(self):
        x = self.add("content of x to be redacted", "user")
        y = self.add("content of y to be revoked", "user")

        # Redact x: tombstoned must stay 0; x must still appear in live_sources
        memdag_redact.redact_node(self.conn, x, "redact-test")
        node_x = memdag.get_node(self.conn, x)
        self.assertEqual(node_x["tombstoned"], 0)       # NOT tombstoned
        live_ids = [r[0] for r in memdag.live_sources(self.conn)]
        self.assertIn(x, live_ids)                       # still live

        # Revoke y: content must survive (revoke != redact)
        memdag.revoke_cascade(self.conn, y, "revoke-test")
        node_y = memdag.get_node(self.conn, y)
        self.assertEqual(node_y["content"], "content of y to be revoked")  # payload survives
        self.assertEqual(node_y["tombstoned"], 1)                          # but tombstoned


class TestCascadeRedactsDescendants(Base):
    """test 5: diamond a->b, a->c, (b,c)->d; cascade from a redacts all 4."""

    def test_cascade_redacts_descendants(self):
        a = self.add("root content a", "endorsed")
        b, _ = memdag.derive_node(self.conn, "left branch b", [a])
        c, _ = memdag.derive_node(self.conn, "right branch c", [a])
        d, _ = memdag.derive_node(self.conn, "join node d", [b, c])

        result = memdag_redact.redact_node(self.conn, a, "spill", cascade=True)
        self.assertEqual(sorted(result), sorted([a, b, c, d]))

        for nid in [a, b, c, d]:
            row = self.conn.execute(
                "SELECT content, redacted, redact_reason FROM nodes WHERE id=?",
                (nid,)).fetchone()
            self.assertEqual(row[0], "")     # content destroyed
            self.assertEqual(row[1], 1)      # redacted flag

        # d's reason should be cascade from node a
        row_d = self.conn.execute(
            "SELECT redact_reason FROM nodes WHERE id=?", (d,)).fetchone()
        self.assertEqual(row_d[0], f"cascade from node {a}")


class TestReredactNoop(Base):
    """test 6: re-redact with a new reason is a no-op; first redaction wins."""

    def test_reredact_noop(self):
        a = self.add("content to redact once", "endorsed")
        first_result = memdag_redact.redact_node(self.conn, a, "original-reason")
        self.assertEqual(first_result, [a])

        row_after_first = self.conn.execute(
            "SELECT redacted_at, redact_reason FROM nodes WHERE id=?", (a,)).fetchone()
        first_redacted_at = row_after_first[0]
        first_reason = row_after_first[1]

        # Re-redact with a different reason
        second_result = memdag_redact.redact_node(self.conn, a, "new-reason")
        self.assertEqual(second_result, [])   # nothing newly redacted

        row_after_second = self.conn.execute(
            "SELECT redacted_at, redact_reason FROM nodes WHERE id=?", (a,)).fetchone()
        self.assertEqual(row_after_second[0], first_redacted_at)   # timestamp unchanged
        self.assertEqual(row_after_second[1], first_reason)         # reason unchanged


class TestUnknownIdRaises(Base):
    """test 7: unknown id raises ValueError."""

    def test_unknown_id_raises(self):
        with self.assertRaises(ValueError):
            memdag_redact.redact_node(self.conn, 99999, "test")

        with self.assertRaises(ValueError):
            memdag_redact.is_redacted(self.conn, 99999)

        with self.assertRaises(ValueError):
            memdag_redact.describe(self.conn, 99999)


class TestCliDryRunThenYes(Base):
    """test 8: CLI dry-run leaves redacted=0; --yes applies."""

    def test_cli_dry_run_then_yes(self):
        a = self.add("source content for cli test here.", "endorsed")

        # Dry run: --yes absent
        out = self.run_main("redact", str(a), "--reason", "cli-test")
        self.assertIn("dry run", out)
        self.assertIn("will redact 1 node(s):", out)

        # Nothing should be applied yet
        row = self.conn.execute("SELECT redacted FROM nodes WHERE id=?", (a,)).fetchone()
        self.assertEqual(row[0], 0)

        # Apply with --yes
        out2 = self.run_main("redact", str(a), "--reason", "cli-test", "--yes")
        self.assertIn("done - 1 redacted", out2)
        self.assertIn("0 rows deleted", out2)
        self.assertIn("all edges intact", out2)

        # Verify it was applied
        row2 = self.conn.execute("SELECT redacted FROM nodes WHERE id=?", (a,)).fetchone()
        self.assertEqual(row2[0], 1)

    def test_cli_unknown_id_exits_1(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            with self.assertRaises(SystemExit) as cm:
                self.run_main("redact", "99999", "--reason", "nope")
        self.assertEqual(cm.exception.code, 1)

    def test_cli_cascade_dry_run_and_yes(self):
        a = self.add("cascade root content", "endorsed")
        b, _ = memdag.derive_node(self.conn, "cascade child content", [a])

        # Dry run with cascade
        out = self.run_main("redact", str(a), "--reason", "cascade-cli", "--cascade")
        self.assertIn("dry run", out)
        self.assertIn("seed", out)
        self.assertIn("descendant", out)

        # Neither a nor b should be redacted yet
        for nid in [a, b]:
            row = self.conn.execute("SELECT redacted FROM nodes WHERE id=?", (nid,)).fetchone()
            self.assertEqual(row[0], 0)

        # Apply
        out2 = self.run_main("redact", str(a), "--reason", "cascade-cli",
                              "--cascade", "--yes")
        self.assertIn("done - 2 redacted", out2)

        # Both should now be redacted
        for nid in [a, b]:
            row = self.conn.execute("SELECT redacted FROM nodes WHERE id=?", (nid,)).fetchone()
            self.assertEqual(row[0], 1)


class TestDefaultRedactCascadesToDescendants(Base):
    """Regression: mirrors poc1_compose_leak.py — default redact MUST cascade.

    F-05: non-cascade redaction left verbatim secret in derived (compose/ask) nodes.
    With cascade=True as the default, redacting a source must also wipe every
    transitive descendant that baked the secret via compose()+derive_node().
    """

    def test_default_redact_cascades_to_descendants(self):
        SECRET = "MyTopSecretPassword is hunter2 and nobody should know this."

        # Insert source containing the secret
        src_id = self.add(SECRET, "user")

        # Compose an answer that bakes the verbatim sentence into a derived node
        sources = memdag.live_sources(self.conn)
        text, used = memdag.compose("password secret", sources)
        self.assertIsNotNone(text, "compose() should produce output from a live source")
        self.assertIn("hunter2", text, "compose() should bake verbatim sentence")

        derived_id, _ = memdag.derive_node(self.conn, text, used)

        # Confirm derived node contains the secret before redaction
        derived_pre = self.conn.execute(
            "SELECT content FROM nodes WHERE id=?", (derived_id,)
        ).fetchone()[0]
        self.assertIn("hunter2", derived_pre)

        # Redact source with NO cascade kwarg — default must cascade
        newly = memdag_redact.redact_node(self.conn, src_id, "audit-poc1")
        self.assertIn(src_id, newly)
        self.assertIn(derived_id, newly, "derived node must be cascade-redacted by default")

        # Derived node content must now be empty and marked redacted
        row = self.conn.execute(
            "SELECT content, redacted FROM nodes WHERE id=?", (derived_id,)
        ).fetchone()
        self.assertEqual(row[0], "", "derived node content must be wiped")
        self.assertEqual(row[1], 1, "derived node must be marked redacted=1")

        # Secret must not appear in ANY node's content
        all_contents = self.conn.execute(
            "SELECT content FROM nodes"
        ).fetchall()
        for (c,) in all_contents:
            self.assertNotIn(
                "hunter2", c,
                f"secret 'hunter2' leaked in node content: {c!r}"
            )

        # compose() over live_sources must not surface the secret
        sources_after = memdag.live_sources(self.conn)
        text_after, _ = memdag.compose("password secret", sources_after)
        if text_after is not None:
            self.assertNotIn(
                "hunter2", text_after,
                "secret 'hunter2' must not appear in compose() output after redaction"
            )

        # Edges must be intact (derived->source edge still exists)
        parents = memdag.parents_of(self.conn, derived_id)
        parent_ids = [p[0] for p in parents]
        self.assertIn(src_id, parent_ids, "edge from derived to source must survive redaction")


class TestRedactSingleOptOut(Base):
    """Regression: --single / cascade=False escape hatch still redacts only the seed.

    Verifies that explicitly passing cascade=False leaves the child node untouched,
    proving the opt-out path is still functional.
    """

    def test_redact_single_opt_out(self):
        src_id = self.add("source content with a secret value abc123", "user")
        child_id, _ = memdag.derive_node(
            self.conn, "child content derived from source abc123", [src_id]
        )

        # Redact ONLY the source, explicitly opting out of cascade
        newly = memdag_redact.redact_node(self.conn, src_id, "single-redact", cascade=False)
        self.assertEqual(newly, [src_id], "only the source should be newly redacted")

        # Source must be redacted
        src_row = self.conn.execute(
            "SELECT content, redacted FROM nodes WHERE id=?", (src_id,)
        ).fetchone()
        self.assertEqual(src_row[0], "")
        self.assertEqual(src_row[1], 1)

        # Child must NOT be redacted — escape hatch preserved
        child_row = self.conn.execute(
            "SELECT content, redacted FROM nodes WHERE id=?", (child_id,)
        ).fetchone()
        self.assertNotEqual(child_row[0], "", "child content must survive with cascade=False")
        self.assertEqual(child_row[1], 0, "child must remain redacted=0 with cascade=False")


if __name__ == "__main__":
    unittest.main()
