"""Tests for the panel's knob providers, validation, and two-phase audit log.

Everything here works against temp files/dirs only — never live memsom state.

Run:  python -m pytest tests/test_panel_knobs.py -q
"""

import json
import tempfile
import unittest
from pathlib import Path

from memsom.interface import panel


def _write_profile(dirpath, *, knobs=None, tasks=None, contracts=None, audit_log=None):
    profile = {
        "knobs": knobs or [],
        "tasks": tasks or [],
        "contracts": contracts or [],
        "telemetry": {},
        "audit_log": audit_log or str(Path(dirpath) / "audit.jsonl"),
    }
    path = Path(dirpath) / "panel_profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


class ValidateBounds(unittest.TestCase):
    def test_int_type_ok(self):
        knob = {"type": "int", "bounds": {"min": 1, "max": 10}}
        self.assertIsNone(panel.validate_bounds(knob, 5))

    def test_int_rejects_bool(self):
        knob = {"type": "int", "bounds": {}}
        reason = panel.validate_bounds(knob, True)
        self.assertIsNotNone(reason)
        self.assertIn("int", reason)

    def test_int_rejects_string(self):
        knob = {"type": "int", "bounds": {}}
        reason = panel.validate_bounds(knob, "5")
        self.assertIsNotNone(reason)

    def test_float_accepts_int(self):
        knob = {"type": "float", "bounds": {"min": 0, "max": 1}}
        self.assertIsNone(panel.validate_bounds(knob, 1))

    def test_bool_rejects_int(self):
        knob = {"type": "bool", "bounds": {}}
        reason = panel.validate_bounds(knob, 1)
        self.assertIsNotNone(reason)

    def test_list_of_str_ok(self):
        knob = {"type": "list-of-str", "bounds": {}}
        self.assertIsNone(panel.validate_bounds(knob, ["a", "b"]))

    def test_list_of_str_rejects_mixed(self):
        knob = {"type": "list-of-str", "bounds": {}}
        reason = panel.validate_bounds(knob, ["a", 1])
        self.assertIsNotNone(reason)

    def test_below_minimum_rejected_with_reason(self):
        knob = {"type": "int", "bounds": {"min": 10, "max": 20}}
        reason = panel.validate_bounds(knob, 5)
        self.assertIsNotNone(reason)
        self.assertIn("minimum", reason)

    def test_above_maximum_rejected_with_reason(self):
        knob = {"type": "int", "bounds": {"min": 10, "max": 20}}
        reason = panel.validate_bounds(knob, 99)
        self.assertIsNotNone(reason)
        self.assertIn("maximum", reason)

    def test_never_clamps_value_unchanged(self):
        # validate_bounds only reports; it must never mutate/return a clamped value.
        knob = {"type": "int", "bounds": {"min": 10, "max": 20}}
        value = 999
        panel.validate_bounds(knob, value)
        self.assertEqual(value, 999)

    def test_unknown_type_is_refused_fail_closed(self):
        # A type we can't validate is a write we don't make: silently skipping
        # let garbage through to consumers with no guards (recall.route_weights
        # indexed the value by key and crashed on the next query).
        knob = {"type": "list", "bounds": {}}
        self.assertIn("fail closed", panel.validate_bounds(knob, {"anything": "goes"}))
        self.assertIn("fail closed", panel.validate_bounds({"bounds": {}}, 5))  # absent type too

    def test_weights_type_structural_check(self):
        default = {"identifier": [1.0, 0.0], "mixed": [0.55, 0.45], "prose": [0.25, 0.75]}
        knob = {"type": "weights", "default": default}
        ok = {"identifier": [0.9, 0.1], "mixed": [0.5, 0.5], "prose": [0.2, 0.8]}
        self.assertIsNone(panel.validate_bounds(knob, ok))
        self.assertIsNotNone(panel.validate_bounds(knob, "garbage"))          # not a dict
        self.assertIsNotNone(panel.validate_bounds(knob, {"identifier": [1, 0]}))  # missing keys
        bad_len = dict(ok, mixed=[0.5])                                        # wrong arity
        self.assertIsNotNone(panel.validate_bounds(knob, bad_len))
        bad_range = dict(ok, prose=[0.2, 1.8])                                 # out of [0,1]
        self.assertIsNotNone(panel.validate_bounds(knob, bad_range))
        bad_bool = dict(ok, prose=[True, 0.5])                                 # bool is not a number
        self.assertIsNotNone(panel.validate_bounds(knob, bad_bool))


class CanonicalParamsProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.target = self.dir / "canonical.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _knob(self, key="demote_below", default=0.2):
        return {"id": "canonical.demote_below", "provider": "canonical-params",
                "target": str(self.target), "key": key, "type": "float",
                "bounds": {"min": 0, "max": 1}, "default": default}

    def test_read_absent_file_returns_default_param(self):
        provider = panel.CanonicalParamsProvider()
        knob = self._knob()
        # forget.load_params merges DEFAULTS when the file is absent.
        from memsom.lifecycle import forget
        self.assertEqual(provider.read(knob), forget.DEFAULTS["demote_below"])

    def test_write_preserves_memories_and_unknown_keys_strips_cruft_bumps_updated(self):
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.write_text(json.dumps({
            "version": 3,
            "updated": "2020-01-01T00:00:00Z",
            "params": {"demote_below": 0.2, "cap": 9.0, "decay": 9.0, "gain": 9.0, "seed": 9.0},
            "memories": {"user_foo": {"rs": 0.9}},
            "some_unknown_top_level_key": "keep-me",
        }), encoding="utf-8")

        provider = panel.CanonicalParamsProvider()
        knob = self._knob()
        result = provider.write(knob, 0.33)
        self.assertEqual(result, 0.33)

        out = json.loads(self.target.read_text(encoding="utf-8"))
        self.assertEqual(out["params"]["demote_below"], 0.33)
        for cruft in ("cap", "decay", "gain", "seed"):
            self.assertNotIn(cruft, out["params"])
        self.assertEqual(out["memories"], {"user_foo": {"rs": 0.9}})
        self.assertEqual(out["version"], 3)
        self.assertEqual(out["some_unknown_top_level_key"], "keep-me")
        self.assertNotEqual(out["updated"], "2020-01-01T00:00:00Z")

    def test_write_reread_merge_survives_concurrent_mutation(self):
        self.target.write_text(json.dumps({
            "version": 1, "updated": "x",
            "params": {"demote_below": 0.2},
            "memories": {"stale": {"rs": 0.1}},
        }), encoding="utf-8")

        def mutate_between_read_and_write(target_path):
            # Simulates a concurrent writer (e.g. the forgetting-layer
            # reconcile) landing fresh `memories` between our baseline read
            # and the final atomic write.
            data = json.loads(Path(target_path).read_text(encoding="utf-8"))
            data["memories"] = {"fresh": {"rs": 0.9}}
            Path(target_path).write_text(json.dumps(data), encoding="utf-8")

        provider = panel.CanonicalParamsProvider(_pre_reread_hook=mutate_between_read_and_write)
        knob = self._knob()
        provider.write(knob, 0.5)

        out = json.loads(self.target.read_text(encoding="utf-8"))
        self.assertEqual(out["memories"], {"fresh": {"rs": 0.9}})
        self.assertEqual(out["params"]["demote_below"], 0.5)


class JsonFileProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.target = self.dir / "tunables.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_absent_file_returns_default(self):
        provider = panel.JsonFileProvider()
        knob = {"target": str(self.target), "key": "recall.rrf_c", "default": 60}
        self.assertEqual(provider.read(knob), 60)

    def test_write_creates_nested_path(self):
        provider = panel.JsonFileProvider()
        knob = {"target": str(self.target), "key": "recall.rrf_c", "type": "int",
                "bounds": {"min": 1, "max": 1000}, "default": 60}
        provider.write(knob, 88)
        data = json.loads(self.target.read_text(encoding="utf-8"))
        self.assertEqual(data["recall"]["rrf_c"], 88)
        self.assertEqual(provider.read(knob), 88)

    def test_write_preserves_unrelated_keys(self):
        self.target.write_text(json.dumps({
            "recall": {"rrf_c": 60, "fuse_depth": 30},
            "hook": {"k": 4},
        }), encoding="utf-8")
        provider = panel.JsonFileProvider()
        knob = {"target": str(self.target), "key": "recall.rrf_c", "type": "int",
                "bounds": {"min": 1, "max": 1000}, "default": 60}
        provider.write(knob, 77)
        data = json.loads(self.target.read_text(encoding="utf-8"))
        self.assertEqual(data["recall"]["rrf_c"], 77)
        self.assertEqual(data["recall"]["fuse_depth"], 30)
        self.assertEqual(data["hook"]["k"], 4)

    def test_write_rejects_type_mismatch_no_clamp(self):
        provider = panel.JsonFileProvider()
        knob = {"target": str(self.target), "key": "recall.rrf_c", "type": "int",
                "bounds": {"min": 1, "max": 1000}, "default": 60}
        with self.assertRaises(panel.KnobValidationError):
            provider.write(knob, "not an int")
        self.assertFalse(self.target.exists())

    def test_write_rejects_out_of_bounds(self):
        provider = panel.JsonFileProvider()
        knob = {"target": str(self.target), "key": "recall.rrf_c", "type": "int",
                "bounds": {"min": 1, "max": 1000}, "default": 60}
        with self.assertRaises(panel.KnobValidationError):
            provider.write(knob, 5000)


class SetLineProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.target = self.dir / "bge_env.cmd"

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_absent_file_returns_default(self):
        provider = panel.SetLineProvider()
        knob = {"target": str(self.target), "key": "BGE_IDLE_SEC", "type": "int", "default": 120}
        self.assertEqual(provider.read(knob), 120)

    def test_read_parses_existing_line(self):
        self.target.write_bytes(b"@rem header\r\nset BGE_IDLE_SEC=120\r\nset OTHER=1\r\n")
        provider = panel.SetLineProvider()
        knob = {"target": str(self.target), "key": "BGE_IDLE_SEC", "type": "int", "default": 999}
        self.assertEqual(provider.read(knob), 120)

    def test_write_rewrites_exact_line_preserves_rest_byte_for_byte(self):
        original = b"@rem a comment\r\nset BGE_IDLE_SEC=120\r\nset BGE_PROC_IDLE_SEC=600\r\n"
        self.target.write_bytes(original)
        provider = panel.SetLineProvider()
        knob = {"target": str(self.target), "key": "BGE_IDLE_SEC", "type": "int",
                "bounds": {"min": 10, "max": 3600}, "default": 120}
        provider.write(knob, 240)

        out = self.target.read_bytes()
        self.assertEqual(out, b"@rem a comment\r\nset BGE_IDLE_SEC=240\r\nset BGE_PROC_IDLE_SEC=600\r\n")
        # CRLF preserved throughout
        self.assertNotIn(b"\n\n", out)
        for line in out.split(b"\r\n")[:-1]:
            self.assertNotIn(b"\n", line)

    def test_write_appends_if_missing(self):
        self.target.write_bytes(b"@rem header\r\nset OTHER=1\r\n")
        provider = panel.SetLineProvider()
        knob = {"target": str(self.target), "key": "BGE_POLL_SEC", "type": "int",
                "bounds": {"min": 1, "max": 60}, "default": 5}
        provider.write(knob, 5)

        out = self.target.read_bytes()
        self.assertEqual(out, b"@rem header\r\nset OTHER=1\r\nset BGE_POLL_SEC=5\r\n")

    def test_write_appends_to_file_missing_trailing_newline(self):
        self.target.write_bytes(b"set OTHER=1")  # no trailing newline at all
        provider = panel.SetLineProvider()
        knob = {"target": str(self.target), "key": "BGE_POLL_SEC", "type": "int",
                "bounds": {}, "default": 5}
        provider.write(knob, 7)
        out = self.target.read_bytes()
        self.assertEqual(out, b"set OTHER=1\r\nset BGE_POLL_SEC=7\r\n")

    def test_write_rejects_out_of_bounds_no_clamp(self):
        self.target.write_bytes(b"set BGE_IDLE_SEC=120\r\n")
        provider = panel.SetLineProvider()
        knob = {"target": str(self.target), "key": "BGE_IDLE_SEC", "type": "int",
                "bounds": {"min": 10, "max": 3600}, "default": 120}
        with self.assertRaises(panel.KnobValidationError):
            provider.write(knob, 99999)
        # file untouched
        self.assertEqual(self.target.read_bytes(), b"set BGE_IDLE_SEC=120\r\n")


class SchtasksProviderTests(unittest.TestCase):
    def test_read_happy_path(self):
        import subprocess

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 0, stdout='{"trigger":"Daily","enabled":true,"lastRunTime":"x","nextRunTime":"y"}',
                stderr="")

        provider = panel.SchtasksProvider(run=fake_run)
        result = provider.read({"key": "SomeTask"})
        self.assertEqual(result, {"trigger": "Daily", "enabled": True, "last_run": "x", "next_run": "y"})

    def test_read_failure_reports_error(self):
        import subprocess

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="access denied")

        provider = panel.SchtasksProvider(run=fake_run)
        result = provider.read({"key": "SomeTask"})
        self.assertIn("error", result)

    def test_write_degrades_on_nonzero_exit(self):
        import subprocess

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="access is denied")

        provider = panel.SchtasksProvider(run=fake_run)
        result = provider.write({"key": "SomeTask"}, 5)
        self.assertFalse(result["ok"])
        self.assertTrue(result["degraded"])
        self.assertIn("elevated_command", result)
        self.assertIn("SomeTask", result["elevated_command"])

    def test_write_degrades_on_missing_binary(self):
        def fake_run(cmd, **kwargs):
            raise FileNotFoundError("no powershell")

        provider = panel.SchtasksProvider(run=fake_run)
        result = provider.write({"key": "SomeTask"}, 5)
        self.assertFalse(result["ok"])
        self.assertTrue(result["degraded"])

    def test_write_succeeds_on_zero_exit(self):
        import subprocess

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        provider = panel.SchtasksProvider(run=fake_run)
        result = provider.write({"key": "SomeTask"}, 5)
        self.assertTrue(result["ok"])
        self.assertEqual(result["current"], 5)


class TwoPhaseAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _config(self, knobs, tasks=None, contracts=None):
        profile_path = _write_profile(self.dir, knobs=knobs, tasks=tasks, contracts=contracts)
        return panel.build_config(profile_path, host="127.0.0.1", port=0)

    def test_successful_write_logs_intent_then_result_ok(self):
        target = self.dir / "tunables.json"
        knobs = [{"id": "k1", "tier": 1, "provider": "json-file", "target": str(target),
                  "key": "a.b", "type": "int", "bounds": {"min": 0, "max": 100}, "default": 1}]
        config = self._config(knobs)
        status, body = panel.handle_knob_write(config, {"id": "k1", "value": 5})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

        lines = [json.loads(l) for l in config.audit_log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["result"], "pending")
        self.assertEqual(lines[0]["knob"], "k1")
        self.assertEqual(lines[0]["new"], 5)
        self.assertEqual(lines[1]["result"], "ok")
        self.assertEqual(lines[1]["knob"], "k1")

    def test_simulated_provider_failure_logs_failed_line(self):
        target = self.dir / "tunables.json"
        knobs = [{"id": "k1", "tier": 1, "provider": "json-file", "target": str(target),
                  "key": "a.b", "type": "int", "bounds": {"min": 0, "max": 100}, "default": 1}]
        config = self._config(knobs)

        def angry_write(knob, value):
            raise OSError("disk exploded")

        config.providers["json-file"].write = angry_write
        status, body = panel.handle_knob_write(config, {"id": "k1", "value": 5})
        self.assertEqual(status, 500)
        self.assertFalse(body["ok"])

        lines = [json.loads(l) for l in config.audit_log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(lines[-1]["result"], "failed: disk exploded")

    def test_audit_append_failure_refuses_the_write(self):
        target = self.dir / "tunables.json"
        knobs = [{"id": "k1", "tier": 1, "provider": "json-file", "target": str(target),
                  "key": "a.b", "type": "int", "bounds": {"min": 0, "max": 100}, "default": 1}]
        config = self._config(knobs)

        orig_append = panel._audit_append

        def failing_append(path, obj):
            raise OSError("audit disk full")

        panel._audit_append = failing_append
        try:
            status, body = panel.handle_knob_write(config, {"id": "k1", "value": 5})
        finally:
            panel._audit_append = orig_append

        self.assertEqual(status, 503)
        self.assertFalse(body["ok"])
        # the write itself must never have happened
        self.assertFalse(target.exists())

    def test_result_line_append_failure_does_not_change_the_reported_outcome(self):
        target = self.dir / "tunables.json"
        knobs = [{"id": "k1", "tier": 1, "provider": "json-file", "target": str(target),
                  "key": "a.b", "type": "int", "bounds": {"min": 0, "max": 100}, "default": 1}]
        config = self._config(knobs)

        orig_append = panel._audit_append
        calls = {"n": 0}

        def flaky_append(path, obj):
            calls["n"] += 1
            if calls["n"] == 1:
                return orig_append(path, obj)  # intent line succeeds
            raise OSError("result append failed")  # result line fails

        panel._audit_append = flaky_append
        try:
            status, body = panel.handle_knob_write(config, {"id": "k1", "value": 5})
        finally:
            panel._audit_append = orig_append

        # the write succeeded even though logging the RESULT line failed
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["a"]["b"], 5)

    def test_crash_marker_detected_for_unresolved_pending(self):
        target = self.dir / "tunables.json"
        knobs = [{"id": "k1", "tier": 1, "provider": "json-file", "target": str(target),
                  "key": "a.b", "type": "int", "bounds": {}, "default": 1}]
        config = self._config(knobs)
        panel._audit_append(config.audit_log_path, {"ts": "t0", "knob": "k1", "old": 1, "new": 2,
                                                     "result": "pending"})
        markers = panel.scan_crash_markers(config.audit_log_path)
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0]["knob"], "k1")

    def test_no_crash_marker_when_pending_is_resolved(self):
        target = self.dir / "tunables.json"
        knobs = [{"id": "k1", "tier": 1, "provider": "json-file", "target": str(target),
                  "key": "a.b", "type": "int", "bounds": {}, "default": 1}]
        config = self._config(knobs)
        panel._audit_append(config.audit_log_path, {"ts": "t0", "knob": "k1", "old": 1, "new": 2,
                                                     "result": "pending"})
        panel._audit_append(config.audit_log_path, {"ts": "t1", "knob": "k1", "result": "ok"})
        self.assertEqual(panel.scan_crash_markers(config.audit_log_path), [])

    def test_tier3_contract_refuses_write_and_audits(self):
        contracts = [{"id": "dense_dim", "label": "dim", "value": 1024, "note": "fixed"}]
        config = self._config(knobs=[], contracts=contracts)
        status, body = panel.handle_knob_write(config, {"id": "dense_dim", "value": 2048})
        self.assertEqual(status, 403)
        self.assertFalse(body["ok"])

        lines = [json.loads(l) for l in config.audit_log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(lines[-1]["result"], "refused-tier3")

    def test_bounds_rejection_still_two_phase_audits(self):
        target = self.dir / "tunables.json"
        knobs = [{"id": "k1", "tier": 1, "provider": "json-file", "target": str(target),
                  "key": "a.b", "type": "int", "bounds": {"min": 0, "max": 10}, "default": 1}]
        config = self._config(knobs)
        status, body = panel.handle_knob_write(config, {"id": "k1", "value": 999})
        self.assertEqual(status, 400)
        self.assertFalse(body["ok"])
        self.assertFalse(target.exists())  # never clamped/written

        lines = [json.loads(l) for l in config.audit_log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[-1]["result"].startswith("refused-invalid"))

    def test_schtasks_readonly_refusal(self):
        knobs = [{"id": "t1", "tier": 2, "provider": "schtasks", "key": "LockedTask",
                  "type": "int", "bounds": {}, "default": 1}]
        tasks = [{"name": "LockedTask", "writable": False}]
        config = self._config(knobs, tasks=tasks)
        status, body = panel.handle_knob_write(config, {"id": "t1", "value": 2})
        self.assertEqual(status, 403)
        self.assertIn("read-only", body["error"])

        lines = [json.loads(l) for l in config.audit_log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(lines[-1]["result"], "refused-readonly")

    def test_reset_writes_the_default(self):
        target = self.dir / "tunables.json"
        target.write_text(json.dumps({"a": {"b": 999}}), encoding="utf-8")
        knobs = [{"id": "k1", "tier": 1, "provider": "json-file", "target": str(target),
                  "key": "a.b", "type": "int", "bounds": {"min": 0, "max": 1000}, "default": 42}]
        config = self._config(knobs)
        status, body = panel.handle_knob_write(config, {"id": "k1", "reset": True})
        self.assertEqual(status, 200)
        self.assertEqual(body["current"], 42)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["a"]["b"], 42)

    def test_unknown_knob_id_is_404_with_no_audit_line(self):
        config = self._config(knobs=[])
        status, body = panel.handle_knob_write(config, {"id": "nope", "value": 1})
        self.assertEqual(status, 404)
        self.assertFalse(config.audit_log_path.exists())


class FindKnobTests(unittest.TestCase):
    def test_finds_regular_knob(self):
        profile = {"knobs": [{"id": "a", "provider": "json-file"}], "contracts": []}
        knob = panel.find_knob(profile, "a")
        self.assertEqual(knob["provider"], "json-file")
        self.assertEqual(knob["tier"], 1)  # default applied

    def test_finds_contract_as_tier3(self):
        profile = {"knobs": [], "contracts": [{"id": "c", "label": "L", "value": 7}]}
        knob = panel.find_knob(profile, "c")
        self.assertEqual(knob["tier"], 3)
        self.assertIsNone(knob["provider"])
        self.assertEqual(knob["default"], 7)

    def test_unknown_returns_none(self):
        profile = {"knobs": [], "contracts": []}
        self.assertIsNone(panel.find_knob(profile, "nope"))


class ConcurrentWrites(unittest.TestCase):
    """Two knob writes racing the same canonical.json must both land — the
    lost-update regression: without _KNOB_WRITE_LOCK both threads read the
    same snapshot and the second os.replace discarded the first's key."""

    def test_concurrent_writes_all_land_through_real_pipeline(self):
        # Drives handle_knob_write itself (not the provider with a hand-held
        # lock), so this fails if _perform_knob_write ever drops its
        # _KNOB_WRITE_LOCK acquisition: 8 barrier-released threads each write a
        # different canonical param; without serialization the read-modify-write
        # snapshots overlap and os.replace discards earlier keys.
        import tempfile
        import threading as th
        KEYS = ["rs_cap", "rs_seed", "decay_base", "rs_gain",
                "demote_below", "promote_at", "ss_gain", "ss_decay_k"]
        for _round in range(5):
            with tempfile.TemporaryDirectory() as td:
                d = Path(td)
                target = d / "canonical.json"
                target.write_text(
                    json.dumps({"version": 1, "params": {}, "memories": {}}),
                    encoding="utf-8")
                knobs = [{"id": k, "tier": 1, "provider": "canonical-params",
                          "target": str(target), "key": k, "type": "float",
                          "bounds": {"min": 0, "max": 1}, "default": 0.5}
                         for k in KEYS]
                config = panel.build_config(_write_profile(d, knobs=knobs),
                                            host="127.0.0.1", port=0)
                barrier = th.Barrier(len(KEYS))
                results = {}

                def go(key, val):
                    barrier.wait()
                    results[key] = panel.handle_knob_write(config, {"id": key, "value": val})

                threads = [th.Thread(target=go, args=(k, round(0.1 + i * 0.05, 2)))
                           for i, k in enumerate(KEYS)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                params = json.loads(target.read_text(encoding="utf-8"))["params"]
                for i, k in enumerate(KEYS):
                    self.assertEqual(results[k][0], 200, results[k])
                    self.assertEqual(params.get(k), round(0.1 + i * 0.05, 2),
                                     f"lost update on {k} (round {_round}): {params}")

    def test_legit_empty_reread_wins_over_stale_base(self):
        # `fresh is not None` semantics: a valid-but-empty {} re-read must be
        # treated as the truth, not discarded as a failed read.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "canonical.json"
            target.write_text(json.dumps(
                {"version": 1, "params": {"rs_gain": 0.2}, "memories": {"m1": {"rs": 1.0}}}),
                encoding="utf-8")

            def wipe(path):  # concurrent writer emptied the file between reads
                Path(path).write_text("{}", encoding="utf-8")

            provider = panel.CanonicalParamsProvider(_pre_reread_hook=wipe)
            provider.write({"target": str(target), "key": "rs_cap"}, 0.8)
            out = json.loads(target.read_text(encoding="utf-8"))
            # fresh {} is a real read: memories/version come from IT, not the
            # stale baseline. params alone falls back to base when fresh lacks
            # the block entirely (conservative: a params-less canonical.json is
            # pathological, and resurrecting known params beats dropping them).
            self.assertEqual(out["params"], {"rs_gain": 0.2, "rs_cap": 0.8})
            self.assertEqual(out["memories"], {})              # not resurrected from base


if __name__ == "__main__":
    unittest.main()
