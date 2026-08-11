"""memsom.interface.setup -- `memsom setup`, the deployment-mode wizard
(PLAN.md Sec3.3).

Distinct from bootstrap.py on purpose: bootstrap INSTALLS (Python check, pipx/
venv, Ollama, wires MCP client configs); setup CONFIGURES (deployment mode,
the syncguard check, writes ~/.memdag/memsom.json). Bootstrap ends by calling
`memsom setup` as its final step.

Non-interactive only for now (PLAN.md's exit gate; an interactive prompt
wizard is a straightforward addition later, layered on top of `run_setup`
which already takes a plain answers dict).

"Write config, then verify" (Sec3.3 point 8): the config write and the
verification are two separate steps, and **setup's exit code is the
verification's, not the writing's** -- a config that was written but failed
`doctor`'s selfcheck is a FAILED setup, not a successful one with a warning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import memsom
from memsom.kernel import syncguard as memsom_syncguard
from memsom.storage import settings as memsom_settings

VALID_MODES = ("local", "server", "client")


class SetupError(RuntimeError):
    """A configuration-time refusal (bad answers, synced data dir, ...)."""


def load_answers(path) -> dict:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SetupError(f"answers file not found: {p}") from None
    except json.JSONDecodeError as exc:
        raise SetupError(f"answers file is not valid JSON: {p} ({exc})") from None
    if not isinstance(data, dict):
        raise SetupError(f"answers file must contain a JSON object: {p}")
    return data


def _resolve_data_dir(answers: dict) -> Path:
    raw = answers.get("data_dir")
    if raw:
        return Path(raw).expanduser()
    return memsom.DATA_DIR


def run_setup(answers: dict, *, verify=True) -> dict:
    """Pure-ish orchestration: raises SetupError on a configuration-time
    refusal (never reaches the DB/config write); returns a report dict on
    success. *verify* off is a test seam only -- the CLI always verifies.
    """
    mode = answers.get("mode", "local")
    if mode not in VALID_MODES:
        raise SetupError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")

    data_dir = _resolve_data_dir(answers)

    markers = memsom_syncguard.sync_markers(data_dir)
    ack = answers.get("sync_check")
    if markers and ack != "acknowledged-unsafe":
        raise SetupError(
            "refusing to configure a store inside a synced folder: " +
            "; ".join(markers) +
            ". Choose a different --data-dir, or set "
            "\"sync_check\": \"acknowledged-unsafe\" in the answers file if "
            "you have read and accept the corruption / MS-28 risk (PLAN.md Sec3.4)."
        )

    # Pin the process env to the resolved data dir so migrate_all/doctor/mcp
    # --selfcheck (a child process, inherits env) all target the SAME store
    # this setup run is configuring -- not whatever MEMDAG_HOME happened to be.
    os.environ["MEMDAG_HOME"] = str(data_dir)
    os.environ["MEMDAG_DB"] = str(data_dir / "memdag.db")

    settings = memsom_settings.load_settings(data_dir)
    settings["mode"] = mode
    if markers:
        settings["sync_check"] = "acknowledged-unsafe"
    if mode == "server":
        settings["bind"] = answers.get("bind", "")
        settings["port"] = answers.get("port", 8765)
    if mode == "client":
        settings["remote_server_url"] = answers.get("remote_server_url", "")
        settings["remote_device_token"] = answers.get("remote_device_token", "")
    settings_path = memsom_settings.save_settings(data_dir, settings)

    from memsom.interface import cli as memsom_cli
    from memsom.interface import features as memsom_features
    from memsom.lifecycle import doctor as memsom_doctor

    conn = memsom.get_connection(data_dir / "memdag.db")
    try:
        memsom_cli.migrate_all(conn)
        statuses = memsom_features.all_statuses(conn)
    finally:
        conn.close()

    report = {
        "data_dir": str(data_dir),
        "settings_path": str(settings_path),
        "mode": mode,
        "features": statuses,
        "ok": True,
    }

    if verify:
        doctor_report = memsom_doctor.gather(features=statuses)
        report["doctor"] = doctor_report
        selfcheck_rc = doctor_report["selfcheck"]["returncode"]
        report["ok"] = selfcheck_rc == 0

        if mode == "server":
            from memsom.interface import serve as memsom_serve
            ip = settings.get("bind") or memsom_serve.discover_mesh_ip()
            try:
                srv = memsom_serve.build_server(ip, 0)
                srv.server_close()
                report["serve_selfcheck"] = {"ok": True, "bind": ip}
            except SystemExit as exc:
                report["serve_selfcheck"] = {"ok": False, "reason": str(exc)}
                report["ok"] = False

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_setup(args):
    if not args.non_interactive:
        print("[memsom] interactive setup is not implemented yet; use "
              "--non-interactive --answers <file.json>", file=sys.stderr)
        return 1
    if not args.answers:
        print("[memsom] --non-interactive requires --answers <file.json>", file=sys.stderr)
        return 1

    try:
        answers = load_answers(args.answers)
        report = run_setup(answers)
    except SetupError as exc:
        print(f"[memsom] setup refused: {exc}", file=sys.stderr)
        return 1

    print(f"[memsom] config written: {report['settings_path']}")
    print(f"[memsom] mode={report['mode']} data_dir={report['data_dir']}")
    if not report["ok"]:
        print("[memsom] verification FAILED -- config was written but the "
              "store did not pass doctor's selfcheck", file=sys.stderr)
        return 1
    print("[memsom] verification OK")
    return 0


def register(sub) -> None:
    p = sub.add_parser("setup", help="configure a deployment mode (PLAN.md Sec3.3)")
    p.add_argument("--non-interactive", action="store_true",
                   help="required for now; interactive mode is not implemented")
    p.add_argument("--answers", default=None, help="path to a JSON answers file")
    p.set_defaults(func=cmd_setup)
