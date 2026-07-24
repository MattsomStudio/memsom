"""Regression tests for SessionRunner.read_since cursor semantics.

The live inference/voice frontend cursor-polls read_since while a generation
thread streams one token record per line. Two bugs this locks down:

1. Every token must surface exactly once across successive polls — the cursor
   must not overshoot the record count (a split("\\n") trailing-"" bug once
   buried every post-first record below the cursor, freezing the stream).
2. A token whose text carries U+2028/U+2029/U+0085 must not be dropped —
   str.splitlines() treats those as line breaks and would fragment the record.
"""
import tempfile
import unittest
from pathlib import Path

from memsom.providers.session import FileSink, SessionRunner, new_session_id


class ReadSinceMultiPollTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.runner = SessionRunner(Path(self._tmp.name))
        self.sid = new_session_id()
        self.sink = FileSink(self.runner._path(self.sid))

    def tearDown(self):
        # FileSink holds the file open until done()/error(); close it so the
        # Windows TemporaryDirectory cleanup can unlink it.
        try:
            self.sink._fh.close()
        except Exception:
            pass
        self._tmp.cleanup()

    def _drain(self, cursor):
        r = self.runner.read_since(self.sid, cursor)
        toks = [e["text"] for e in r["events"] if e.get("t") == "tok"]
        return toks, r["cursor"], r["status"]

    def test_streamed_tokens_each_emit_exactly_once(self):
        cursor = 0
        seen = []
        for text in ("A", "B", "C", "D"):
            self.sink.token(text)
            toks, cursor, _ = self._drain(cursor)
            seen.extend(toks)
        self.sink.done({"ok": True})
        toks, cursor, status = self._drain(cursor)
        seen.extend(toks)
        self.assertEqual(seen, ["A", "B", "C", "D"])
        self.assertEqual(status, "done")

    def test_unicode_line_separators_are_not_dropped(self):
        # json.dumps(ensure_ascii=False) writes these literally and splitlines()
        # breaks on them, so a record carrying one must still round-trip whole.
        payloads = ["ab\u2028cd", "ef\u2029gh", "ij\u0085kl"]
        cursor = 0
        seen = []
        for text in payloads:
            self.sink.token(text)
            toks, cursor, _ = self._drain(cursor)
            seen.extend(toks)
        self.assertEqual(seen, payloads)

    def test_burst_of_multiple_records_per_poll(self):
        self.sink.token("A")
        self.sink.token("B")
        self.sink.token("C")
        toks, cursor, _ = self._drain(0)
        self.assertEqual(toks, ["A", "B", "C"])
        self.sink.token("D")
        toks, cursor, _ = self._drain(cursor)
        self.assertEqual(toks, ["D"])


if __name__ == "__main__":
    unittest.main()
