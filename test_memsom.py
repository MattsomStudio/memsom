#!/usr/bin/env python3
"""Tests for memsom — run: python -W error::DeprecationWarning -m unittest -v

Unit tests drive the store primitives against a temp DB; the e2e test runs the
literal demo command sequence as real subprocesses (piped stdout forces the
cp1252 capture path, proving the reconfigure guard). If e2e is green, demo day works.
"""

import contextlib, io, os, sqlite3, subprocess, sys, tempfile, unittest, warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)  # 3.12 sqlite3 adapter regression = hard fail

import memsom

HERE = Path(__file__).resolve().parent


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"  # missing parent: exercises mkdir
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memsom.insert_node(self.conn, content, channel, memsom.RANK[channel])


class TestLabels(Base):
    def test_label_floor_min_of_parents(self):
        e = self.add("endorsed src", "endorsed")
        u = self.add("user src", "user")
        x = self.add("external src", "external")
        _, label = memsom.derive_node(self.conn, "a1", [e, u, x])
        self.assertEqual(label, 0)  # min(3,2,0) = EXTERNAL
        _, label = memsom.derive_node(self.conn, "a2", [e, u])
        self.assertEqual(label, 2)  # min(3,2) = USER

    def test_derive_refuses_tombstoned_or_empty_parents(self):
        e = self.add("endorsed src", "endorsed")
        memsom.revoke_cascade(self.conn, e, "gone")
        with self.assertRaises(ValueError):
            memsom.derive_node(self.conn, "a", [e])
        with self.assertRaises(ValueError):
            memsom.derive_node(self.conn, "a", [])
        with self.assertRaises(ValueError):
            memsom.derive_node(self.conn, "a", [999])


class TestConnectionPragmas(Base):
    def test_busy_timeout_set(self):
        # A concurrent writer holding BEGIN IMMEDIATE (revoke_cascade) must make a
        # parallel writer WAIT, not fail-fast with SQLITE_BUSY. The pragma is the
        # belt to the cascade's covering-index suspenders.
        bt = self.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        self.assertEqual(bt, 5000)

    def test_foreign_keys_on(self):
        fk = self.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        self.assertEqual(fk, 1)


class TestCascade(Base):
    def test_cascade_diamond_each_node_once(self):
        a = self.add("a", "user")
        b, _ = memsom.derive_node(self.conn, "b", [a])
        c, _ = memsom.derive_node(self.conn, "c", [a])
        d, _ = memsom.derive_node(self.conn, "d", [b, c])
        n = memsom.revoke_cascade(self.conn, a, "poison")
        self.assertEqual(n, 4)
        dead = self.conn.execute("SELECT COUNT(*) FROM nodes WHERE tombstoned=1").fetchone()[0]
        self.assertEqual(dead, 4)

    def test_cascade_cycle_terminates(self):
        a = self.add("a", "user")
        b = self.add("b", "user")
        with self.conn:  # manual a<->b cycle; guards UNION against 'optimization' to UNION ALL
            self.conn.execute("INSERT INTO edges(child,parent) VALUES (?,?)", (b, a))
            self.conn.execute("INSERT INTO edges(child,parent) VALUES (?,?)", (a, b))
        n = memsom.revoke_cascade(self.conn, a, "loop")
        self.assertEqual(n, 2)

    def test_revoke_preserves_shape_and_content(self):
        a = self.add("source text", "endorsed")
        b, _ = memsom.derive_node(self.conn, "derived text", [a])
        pre = (self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
               self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        memsom.revoke_cascade(self.conn, a, "retracted")
        post = (self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
                self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        self.assertEqual(pre, post)  # nothing deleted, edges intact
        na, nb = memsom.get_node(self.conn, a), memsom.get_node(self.conn, b)
        self.assertEqual(na["content"], "source text")  # payload survives (tombstone != redact)
        self.assertEqual(na["revoke_reason"], "retracted")
        self.assertEqual(nb["revoke_reason"], f"cascade from node {a}")

    def test_first_death_wins(self):
        a = self.add("a", "user")
        b, _ = memsom.derive_node(self.conn, "b", [a])
        memsom.revoke_cascade(self.conn, b, "direct hit")
        first = memsom.get_node(self.conn, b)
        memsom.revoke_cascade(self.conn, a, "ancestor dies later")
        again = memsom.get_node(self.conn, b)
        self.assertEqual(again["revoke_reason"], "direct hit")  # tombstones are immutable history
        self.assertEqual(again["tombstoned_at"], first["tombstoned_at"])

    def test_fk_enforced(self):
        with self.assertRaises(sqlite3.IntegrityError):
            with self.conn:
                self.conn.execute("INSERT INTO edges(child,parent) VALUES (1,999)")

    def test_atomic_derive_no_orphans(self):
        a = self.add("a", "user")
        conn = self.conn

        class BoomConn:  # sqlite3.Connection attrs are read-only; delegate instead
            def __getattr__(self, name):
                return getattr(conn, name)

            def __enter__(self):
                return conn.__enter__()

            def __exit__(self, *exc):
                return conn.__exit__(*exc)

            def executemany(self, *args, **kw):
                raise sqlite3.OperationalError("simulated edge-insert crash")

        with self.assertRaises(sqlite3.OperationalError):
            memsom.derive_node(BoomConn(), "answer", [a])
        agents = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE channel='agent-derived'").fetchone()[0]
        self.assertEqual(agents, 0)  # rollback: never an answer without its edges


class TestCompose(Base):
    SRC = [(1, "Nebula needs a lighthouse node with a public IP for hole punching.\n"
               "Set static_host_map to point at the lighthouse address always.", "endorsed", 3, None),
           (2, "Use UDP port 4242 for nebula tunnels on every host in the mesh.", "user", 2, None)]

    def test_compose_deterministic(self):
        q = "How should I configure Nebula?"
        self.assertEqual(memsom.compose(q, self.SRC), memsom.compose(q, self.SRC))

    def test_every_source_contributes(self):
        srcs = self.SRC + [(3, "Totally unrelated prose line about gardening tomatoes in July.",
                            "external", 0, None)]
        text, used = memsom.compose("How should I configure Nebula?", srcs)
        self.assertEqual(used, [1, 2, 3])  # zero-hit source rides the fallback
        self.assertIn("[mem:3|external]", text)

    def test_markdown_noise_filtered(self):
        srcs = [(1, "# Heading That Is Long Enough To Pass Length Checks\n"
                    "**Tags:** #nebula #configuration #mesh #overlay #network\n"
                    "> blockquoted nebula configuration advice should not appear\n"
                    "| nebula | configuration | table | row | here |\n"
                    "Real nebula configuration advice lives on this prose line.", "user", 2, None)]
        text, _ = memsom.compose("nebula configuration", srcs)
        self.assertIn("Real nebula configuration advice", text)
        for junk in ("Heading", "Tags", "blockquoted", "table"):
            self.assertNotIn(junk, text)

    def test_frontmatter_and_fence_interiors_excluded(self):
        content = ("---\n"
                   "tags: [nebula, mesh, configuration-of-the-overlay-network]\n"
                   "summary: how the nebula mesh overlay network is configured at home\n"
                   "---\n"
                   "# Title\n"
                   "Real prose claim about the nebula mesh configuration lives here fine.\n"
                   "```yaml\n"
                   "static_host_map: nebula configuration interior must not leak out\n"
                   "```\n"
                   "    indented code block about nebula configuration stays hidden too\n"
                   "- [ ] checkbox task about nebula configuration is residue, skip it\n")
        text, _ = memsom.compose("nebula configuration", [(1, content, "endorsed", 3, None)])
        self.assertIn("Real prose claim", text)
        for junk in ("tags:", "summary:", "static_host_map", "indented code", "checkbox"):
            self.assertNotIn(junk, text)

    def test_inline_furniture_stripped(self):
        content = ("My **bold** `nebula` claim with a [config guide](https://example.com/x)"
                   " link inside it stays readable.")
        text, _ = memsom.compose("nebula config", [(1, content, "user", 2, None)])
        self.assertIn("My bold nebula claim with a config guide link", text)
        for junk in ("**", "`", "https://example.com", "]("):
            self.assertNotIn(junk, text)

    def test_ask_excludes_tombstoned_and_relabels(self):
        ids = {}
        for content, channel in [("Nebula lighthouse config guidance prose line here.", "endorsed"),
                                 ("Nebula firewall rule guidance from the user goes here.", "user"),
                                 ("Nebula crypto details from an external article here.", "external")]:
            ids[channel] = self.add(content, channel)
        q = "How should I configure Nebula?"
        text1, used1 = memsom.compose(q, memsom.live_sources(self.conn))
        a1, label1 = memsom.derive_node(self.conn, text1, used1)
        self.assertEqual(label1, 0)
        memsom.revoke_cascade(self.conn, ids["external"], "retracted")
        live = memsom.live_sources(self.conn)
        self.assertEqual(len(live), 2)
        text2, used2 = memsom.compose(q, live)
        a2, label2 = memsom.derive_node(self.conn, text2, used2)
        self.assertNotEqual(a1, a2)            # re-ask = NEW node (immutability)
        self.assertNotEqual(text1, text2)      # answer visibly changes
        self.assertNotIn(f"mem:{ids['external']}", text2)
        self.assertGreater(label2, label1)     # the money shot: label RISES to USER
        old = memsom.get_node(self.conn, a1)
        self.assertEqual(old["content"], text1)  # old answer intact, just tombstoned
        self.assertEqual(old["tombstoned"], 1)


class TestDigestRedaction(Base):
    """A redacted node keeps tombstoned=0, so without an explicit guard the digest
    falls back to the node's stem and leaks the sensitive filename into the
    always-loaded MEMORY.md. render_digest must drop redacted nodes entirely."""

    SECTIONED = ("---\nname: user_secret_thing\nsection: About the User\n"
                 "description: a private fact\n---\nbody")

    def _add_file_node(self):
        with self.conn:
            return memsom.insert_node(self.conn, self.SECTIONED, "user",
                                      source_ref="memory:user_secret_thing")

    def test_stem_renders_then_drops_on_redact(self):
        import memsom_digest, memsom_redact
        nid = self._add_file_node()
        self.assertIn("user_secret_thing", memsom_digest.render_digest(self.conn))
        memsom_redact.redact_node(self.conn, nid, "test erasure", cascade=False)
        self.conn.commit()
        after = memsom_digest.render_digest(self.conn)
        self.assertNotIn("user_secret_thing", after)
        # redaction contract preserved: liveness untouched, payload destroyed
        row = self.conn.execute(
            "SELECT content, redacted, tombstoned FROM nodes WHERE id=?", (nid,)
        ).fetchone()
        self.assertEqual(row, ("", 1, 0))


class TestRedactReachesDisk(Base):
    """Fix 1: redaction must reach DISK (unlink the flat memory file + vault note)
    and the importer must REFUSE to resurrect a redacted node's resurfaced file."""

    def _seed_file_node(self, mem_dir, stem, body="the secret plaintext"):
        import memsom_bridge_import as bi
        bi.migrate(self.conn)                       # ensure bridge_path/content_hash cols
        p = Path(mem_dir) / f"{stem}.md"
        p.write_text(f"---\nname: {stem}\nsection: About the User\n---\n{body}",
                     encoding="utf-8")
        bi.import_memory_dir(self.conn, mem_dir, dry_run=False)
        row = self.conn.execute(
            "SELECT id FROM nodes WHERE bridge_path=? AND tombstoned=0",
            (f"{stem}.md",)).fetchone()
        self.assertIsNotNone(row, "seed did not create a bridge-backed node")
        return row[0], p

    def test_redact_unlinks_flat_file(self):
        import memsom_redact
        md = tempfile.mkdtemp()
        nid, p = self._seed_file_node(md, "user_secret")
        self.assertTrue(p.exists())
        ps = {}
        memsom_redact.redact_node(self.conn, nid, "erase", cascade=False,
                                  memory_dir=md, purge_stats=ps)
        self.conn.commit()
        self.assertFalse(p.exists())                # reached disk
        self.assertEqual(ps["purged"], 1)
        self.assertEqual(
            self.conn.execute("SELECT content, redacted FROM nodes WHERE id=?",
                              (nid,)).fetchone(), ("", 1))

    def test_redact_no_resurrection_on_reimport(self):
        import memsom_redact, memsom_bridge_import as bi
        md = tempfile.mkdtemp()
        nid, p = self._seed_file_node(md, "user_secret")
        memsom_redact.redact_node(self.conn, nid, "erase", cascade=False, memory_dir=md)
        self.conn.commit()
        self.assertFalse(p.exists())                # redact already unlinked it
        # Simulate the surviving-file race: the file resurfaces (edited) from another
        # machine before the redaction record arrived.
        p.write_text("---\nname: user_secret\nsection: About the User\n---\nEDITED",
                     encoding="utf-8")
        stats = bi.import_memory_dir(self.conn, md, dry_run=False)
        self.assertEqual(stats["refused_resurrect"], 1)
        live = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE bridge_path=? AND tombstoned=0 AND redacted=0",
            ("user_secret.md",)).fetchone()[0]
        self.assertEqual(live, 0)                   # no fresh plaintext node
        self.assertFalse(p.exists())                # resurfaced file unlinked

    def test_redact_unlinks_vault_note(self):
        import memsom_redact, memsom_obsidian
        memsom_obsidian.migrate(self.conn)          # ensure obsidian_path column
        vault = tempfile.mkdtemp()
        note = Path(vault) / "note.md"
        note.write_text("secret vault content", encoding="utf-8")
        with self.conn:
            nid = memsom.insert_node(self.conn, "secret vault content", "user",
                                     source_ref="obsidian:note")
            self.conn.execute("UPDATE nodes SET obsidian_path=? WHERE id=?",
                              ("note.md", nid))
        ps = {}
        memsom_redact.redact_node(self.conn, nid, "erase", cascade=False,
                                  vault=vault, purge_stats=ps)
        self.conn.commit()
        self.assertFalse(note.exists())
        self.assertEqual(ps["purged"], 1)

    def test_redact_traversal_guard_refuses_escape(self):
        import memsom_redact
        md = tempfile.mkdtemp()
        nid, _ = self._seed_file_node(md, "user_secret")
        outside = Path(md).parent / "escape_target.md"
        outside.write_text("must survive", encoding="utf-8")
        # a crafted bridge_path that climbs out of the memory dir must be refused
        with self.conn:
            self.conn.execute("UPDATE nodes SET bridge_path=? WHERE id=?",
                              ("../escape_target.md", nid))
        ps = {}
        memsom_redact.redact_node(self.conn, nid, "erase", cascade=False,
                                  memory_dir=md, purge_stats=ps)
        self.conn.commit()
        self.assertTrue(outside.exists())           # traversal blocked, file untouched
        self.assertEqual(ps["failed"], 1)
        self.assertEqual(ps["purged"], 0)


class TestUtf8(Base):
    def test_utf8_roundtrip_synthetic(self):
        text = "em-dash — arrow → check ✓ done"
        nid = self.add(text, "endorsed")
        self.assertEqual(memsom.get_node(self.conn, nid)["content"], text)


class TestCli(Base):
    def run_main(self, *argv):
        buf = io.StringIO()  # no .reconfigure -> proves the AttributeError guard
        with contextlib.redirect_stdout(buf):
            memsom.main(list(argv))
        return buf.getvalue()

    def test_explain_walk_and_revoked_marker(self):
        ids = [self.add("Nebula configuration guidance number one prose line.", "endorsed"),
               self.add("Nebula configuration guidance number two prose line.", "user")]
        text, used = memsom.compose("nebula configuration", memsom.live_sources(self.conn))
        aid, _ = memsom.derive_node(self.conn, text, used)
        out = self.run_main("explain", str(aid))
        for nid in ids:
            self.assertIn(f"[{nid}]", out)
        self.assertNotIn("[REVOKED", out)
        memsom.revoke_cascade(self.conn, ids[0], "bad note")
        out = self.run_main("explain", str(aid))
        self.assertIn("[REVOKED", out)
        self.assertIn("bad note", out)

    def test_revoke_dry_run_default_then_yes(self):
        a = self.add("source", "user")
        b, _ = memsom.derive_node(self.conn, "answer", [a])
        out = self.run_main("revoke", str(a))
        self.assertIn("dry run", out)
        self.assertIn("will tombstone 2 node(s):", out)
        self.assertEqual(memsom.get_node(self.conn, a)["tombstoned"], 0)  # nothing applied
        out = self.run_main("revoke", str(a), "--yes", "--reason", "kill it")
        self.assertIn("done - 2 tombstoned, 0 rows deleted", out)
        out = self.run_main("revoke", str(a), "--yes")
        self.assertIn("already tombstoned", out)  # idempotent, exit 0

    def test_ask_refuses_with_no_live_sources(self):
        a = self.add("only source", "user")
        memsom.revoke_cascade(self.conn, a, "gone")
        with self.assertRaises(SystemExit) as cm:
            self.run_main("ask", "anything at all?")
        self.assertEqual(cm.exception.code, 1)
        agents = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE channel='agent-derived'").fetchone()[0]
        self.assertEqual(agents, 0)  # refusal stores nothing

    def test_explain_deep_chain_no_recursion_error(self):
        with self.conn:
            self.conn.execute("INSERT INTO nodes(content, channel, label, created_at)"
                              " VALUES ('root', 'user', 2, '2026-06-10T00:00:00+00:00')")
            for i in range(2, 1502):  # chain far past the interpreter recursion limit
                self.conn.execute(
                    "INSERT INTO nodes(content, channel, label, created_at)"
                    " VALUES ('n', 'agent-derived', 2, '2026-06-10T00:00:00+00:00')")
                self.conn.execute("INSERT INTO edges(child, parent) VALUES (?,?)", (i, i - 1))
        out = self.run_main("explain", "1501")
        self.assertIn("[1] user", out)  # walked all the way to the root, no crash

    def test_explain_diamond_marks_elided_ancestry(self):
        a = self.add("Shared ancestor prose line long enough to show.", "endorsed")
        b, _ = memsom.derive_node(self.conn, "left branch", [a])
        c, _ = memsom.derive_node(self.conn, "right branch", [a])
        d, _ = memsom.derive_node(self.conn, "join node", [b, c])
        out = self.run_main("explain", str(d))
        self.assertEqual(out.count("(ancestry shown above)"), 1)  # second occurrence only

    @unittest.skipUnless(sys.platform == "win32",
                         "only Windows blocks unlink of an open DB file; POSIX allows it")
    def test_seed_reset_locked_db_clean_error(self):
        holder = memsom.get_connection()  # open handle: Windows blocks the unlink
        try:
            with self.assertRaises(SystemExit) as cm:
                self.run_main("seed", "--reset", "--offline")
            self.assertEqual(cm.exception.code, 1)  # clean message, not a traceback
        finally:
            holder.close()


@unittest.skipUnless(memsom.FALLBACK.exists(), "demo fallback not present")
class TestEndToEndDemo(unittest.TestCase):
    """The literal demo sequence as real subprocesses. Green here = demo day works.

    Seed content is now neutral (SQLite-themed); the question shares the word
    'sqlite' with all three seeded nodes so every source is used (used: 3)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {**os.environ, "MEMDAG_DB": str(Path(self.tmp.name) / "demo.db")}

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *args, expect=0):
        r = subprocess.run([sys.executable, str(HERE / "memsom.py"), *args],
                           capture_output=True, text=True, encoding="utf-8",
                           env=self.env, cwd=HERE, timeout=60)
        self.assertEqual(r.returncode, expect,
                         f"memsom {' '.join(args)}\nstdout:{r.stdout}\nstderr:{r.stderr}")
        return r.stdout

    def test_demo_sequence(self):
        q = "What is SQLite?"
        out = self.cli("seed", "--offline")
        self.assertIn("[1] user", out)
        self.assertIn("[2] endorsed", out)
        self.assertIn("[3] external", out)

        dump = self.cli("dump")                        # runs on camera before the first ask
        self.assertIn("edges (0):", dump)
        for nid in ("[1]", "[2]", "[3]"):
            self.assertIn(nid, dump)
        self.assertNotIn("last-verified", dump)        # endorsed snippet skips frontmatter

        ask1 = self.cli("ask", q)
        self.assertIn("[mem:2|endorsed]", ask1)
        self.assertIn("[mem:3|external]", ask1)
        self.assertIn("stored as node [4]", ask1)
        self.assertIn("integrity: EXTERNAL", ask1)
        self.assertIn("considered: 3, used: 3, excluded: 0", ask1)

        explain = self.cli("explain", "4")
        for nid in ("[1]", "[2]", "[3]"):
            self.assertIn(nid, explain)
        self.assertIn("floor set by [3]", explain)

        dry = self.cli("revoke", "3", "--reason", "untrusted source retracted")
        self.assertIn("will tombstone 2 node(s):", dry)
        self.assertIn("dry run", dry)
        done = self.cli("revoke", "3", "--reason", "untrusted source retracted", "--yes")
        self.assertIn("done - 2 tombstoned, 0 rows deleted, all edges intact.", done)

        ask2 = self.cli("ask", q)
        self.assertIn("stored as node [5]", ask2)
        self.assertNotIn("[mem:3|", ask2)              # external claims gone
        self.assertIn("integrity: USER", ask2)         # label ROSE: min(3,2) not min(3,2,0)
        self.assertIn("considered: 3, used: 2, excluded: 1 (tombstoned)", ask2)
        self.assertNotEqual(ask1.splitlines()[2:], ask2.splitlines()[2:])

        explain5 = self.cli("explain", "5")            # fresh derivation, honest floor
        self.assertIn("floor set by [1]", explain5)
        self.assertNotIn("[REVOKED", explain5)

        history = self.cli("explain", "4")             # history intact after the kill
        self.assertIn("[REVOKED", history)
        self.assertIn("untrusted source retracted", history)

        final_dump = self.cli("dump")                  # tombstone flags + every edge visible
        self.assertIn("[3] T", final_dump)
        self.assertIn("[4] T", final_dump)
        self.assertIn("[5] .", final_dump)
        self.assertIn("edges (5):", final_dump)        # 3 from node 4 + 2 from node 5

        reseed = subprocess.run([sys.executable, str(HERE / "memsom.py"), "seed", "--offline"],
                                capture_output=True, text=True, encoding="utf-8",
                                env=self.env, cwd=HERE, timeout=60)
        self.assertEqual(reseed.returncode, 1)         # refuses to double-seed
        self.assertIn("already seeded", reseed.stderr)

        reset = self.cli("seed", "--reset", "--offline")  # between-rehearsal reset path
        self.assertIn("[1] user", reset)
        ask3 = self.cli("ask", q)
        self.assertIn("stored as node [4]", ask3)      # ids stable again after reset


if __name__ == "__main__":
    unittest.main()
