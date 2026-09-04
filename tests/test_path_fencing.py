"""Inventory lock on the path primitives.

Written 2026-07-30 for seam S3 of the memsom-panel white-box engagement.

`test_paths.py` proves `safe_join` behaves. This file proves the codebase
actually *uses* it — which is the failure that keeps happening. The audit filed
two resolve-before-fence sites; an AST sweep of the live tree found seven,
including one inside the scope checker itself and three in code a review had
written up as "well-defended." Nobody was careless. The sites are individually
reasonable and collectively invisible, because `Path.resolve()` reads like a
predicate and grep cannot tell you which call is a fence and which is a lookup.

So: freeze the set of functions allowed to call `.resolve()` on a path. A new
one fails this test, and the author has to either route through
`memsom.paths.safe_join` or add an entry here with a reason. That turns the
eighth instance into a code-review conversation instead of a silent regression.

This is a source-shape assertion, not a behavioural one. It is deliberately
annoying to add to.
"""

from __future__ import annotations

import ast
import pathlib

try:
    import pytest
except ImportError:  # stdlib-only unittest discover (CI runs these under pytest)
    import unittest
    raise unittest.SkipTest("pytest-style module; run under the CI pytest step")

PKG = pathlib.Path(__file__).resolve().parents[1] / "memsom"

# ---------------------------------------------------------------------------
# Every function permitted to call `.resolve()`, and why.
#
# Key:   "<path/relative/to/memsom>::<enclosing function>"
# Value: why this one is not a fence that needs safe_join.
#
# Three legitimate categories, and nothing else should appear here:
#   SELF     - Path(__file__).resolve(), locating the package's own files.
#   ROOT     - resolving a TRUSTED containing directory (config, a CLI arg).
#              The root is the thing safe_join fences *against*; resolving it
#              is required, not a smell.
#   PRIMITIVE- inside memsom/paths.py, which is the implementation.
# ---------------------------------------------------------------------------
ALLOWED: dict[str, str] = {
    "effects/net.py::<module>":
        "SELF - HOME = Path(__file__).resolve().parent.parent, the packaged-resource "
        "dir. This is where HOME now lives (moved from __init__.py, which re-exports "
        "it via the Phase-2 facade in memsom.effects.net).",
    "bridge/wire_claude.py::default_skills_src":
        "SELF - Path(__file__).resolve().parents[2], locates the bundled claude/ dir.",
    "bridge/wire_claude.py::resolve_exe":
        "SELF - sys.executable's own dir and shutil.which('memsom'), resolved to "
        "write an absolute hook path into settings.json. Interpreter/PATH "
        "locations, never model- or file-supplied strings.",
    "bridge/obsidian.py::_walk_markdown":
        "ROOT + walked path. `vroot` is the trusted vault root. `ap.resolve()` is a "
        "path built by os.walk of that same tree, re-checked with _within to drop "
        "symlinks/junctions escaping a Syncthing-shared vault. The input is not "
        "model-authored, so there is nothing for safe_join to fence.",
    "bridge/obsidian.py::export_note":
        "ROOT - `vault = Path(vault).resolve()`. The model-authored `folder` and "
        "`title` now go through safe_join immediately below it.",
    "integrity/tombstone.py::tombstone_memory":
        "ROOT - `mem_root = Path(mem_dir).resolve()`, and it runs AFTER safe_join "
        "has already decided the model-supplied `stem` is contained.",
    "lifecycle/compact.py::_llm_summarize":
        "NOT A PATH - `memsom_llm.resolve(model, base_url)` picks a model endpoint. "
        "Same attribute name, unrelated call. Listed so the sweep's false positive "
        "is on the record rather than being rediscovered every time.",
    "paths.py::safe_join":
        "PRIMITIVE - the root resolve and the post-containment symlink re-check. "
        "This is the implementation the rest of the codebase defers to.",

    # -- kernel/syncguard.py -- resolving the TRUSTED store/data directory (the
    # DB path from MEMDAG_DB/MEMDAG_HOME, or the `memsom setup` wizard's own
    # answer) and OS-set OneDrive env vars, to walk ancestors looking for
    # file-sync markers. Neither input is model-, request-, or store-supplied;
    # both are exactly the kind of trusted root safe_join fences *against*. --
    "kernel/syncguard.py::_ancestors":
        "ROOT - resolves the store's data directory (path.parent of MEMDAG_DB, or "
        "the `memsom setup` wizard's own answer) to walk its ancestors for "
        "file-sync markers. Trusted config path, not model/request-supplied.",
    "kernel/syncguard.py::_onedrive_roots":
        "ROOT - resolves $OneDrive / $OneDriveCommercial / $OneDriveConsumer, "
        "OS-set env-var directories, to compare against the store path above.",

    # -- memsom.tuning.resolve(key) -- an in-process-override > env > default
    # config-knob lookup (see tuning.py's `resolve()` docstring). Same
    # attribute name as Path.resolve(), unrelated call: no filesystem I/O, no
    # path construction. Listed individually (one per call site, same as the
    # existing lifecycle/compact.py::_llm_summarize entry) so the sweep's
    # false positive is on the record rather than being rediscovered. --
    "bridge/bridge_import.py::index_enabled":
        "NOT A PATH - memsom_tuning.resolve(\"bridge.index_enabled\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "bridge/bridge_render.py::_is_author":
        "NOT A PATH - memsom_tuning.resolve(\"bridge.author\") reads a config knob. "
        "Same attribute name, unrelated call.",
    "bridge/claude.py::default_path":
        "NOT A PATH - memsom_tuning.resolve(\"bridge.claude_md_path\") reads a "
        "config knob (the string is then wrapped in Path(...).expanduser(), a "
        "separate, non-.resolve() call). Same attribute name, unrelated call.",
    "bridge/hook.py::hook_mode":
        "NOT A PATH - memsom_tuning.resolve(\"bridge.hook_mode\") reads a config "
        "knob. Same attribute name, unrelated call.",
    "bridge/hook.py::hook_policy_path":
        "NOT A PATH - memsom_tuning.resolve(\"bridge.hook_policy_path\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "bridge/hook.py::shadow_log_path":
        "NOT A PATH - memsom_tuning.resolve(\"bridge.hook_shadow_log\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "bridge/obsidian.py::_default_vault":
        "NOT A PATH - memsom_tuning.resolve(\"obsidian.vault\") reads a config "
        "knob. Same attribute name, unrelated call.",
    "bridge/wire_claude.py::memory_dir_candidates":
        "NOT A PATH - memsom_tuning.resolve(\"bridge.memory_dir\") reads a config "
        "knob. Same attribute name, unrelated call.",
    "distill/digest.py::_section_order":
        "NOT A PATH - memsom_tuning.resolve(\"distill.digest_sections\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "distill/digest.py::_shrink_floor":
        "NOT A PATH - memsom_tuning.resolve(\"distill.digest_shrink_floor\") reads "
        "a config knob. Same attribute name, unrelated call.",
    "distill/digest.py::render_digest":
        "NOT A PATH - memsom_tuning.resolve(\"distill.digest_title\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "distill/digest.py::render_projects_index":
        "NOT A PATH - memsom_tuning.resolve(\"distill.projects_title\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "distill/distill.py::distill_plan":
        "NOT A PATH - memsom_tuning.resolve(\"llm.model\") reads a config knob "
        "(picks an Ollama model name). Same attribute name, unrelated call.",
    "federation/broker.py::__init__":
        "NOT A PATH - memsom_proc.resolve(spec[\"command\"]) resolves an "
        "executable name to an absolute path via PATH search (effects/proc.py, "
        "deliberately never the CWD); it is not pathlib's Path.resolve() and "
        "makes no UNC/SMB-style syscall. `spec` is the federation config file, a "
        "trusted admin-supplied document, not model/request input. Same "
        "attribute name, unrelated call.",
    "federation/broker.py::default_config_path":
        "NOT A PATH - memsom_tuning.resolve(\"federation.broker_config_path\") "
        "reads a config knob. Same attribute name, unrelated call.",
    "federation/federation.py::default_origin":
        "NOT A PATH - memsom_tuning.resolve(\"federation.origin\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "integrity/ingest.py::channel_ceiling":
        "NOT A PATH - memsom_tuning.resolve(\"integrity.channel_ceiling\") reads "
        "a config knob. Same attribute name, unrelated call.",
    "integrity/redact.py::_resolve_vault":
        "NOT A PATH - memsom_tuning.resolve(\"obsidian.vault\") reads a config "
        "knob. Same attribute name, unrelated call.",
    "interface/features.py::_code_rag":
        "NOT A PATH - memsom_tuning.resolve(\"code_rag.qwen_url\") reads a "
        "config knob, used only in a status string. Same attribute name, "
        "unrelated call.",
    "interface/features.py::_contradict_nli":
        "NOT A PATH - memsom_tuning.resolve(\"contradict.nli_enabled\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "interface/features.py::_obsidian":
        "NOT A PATH - memsom_tuning.resolve(\"obsidian.vault\") reads a config "
        "knob, used only in a status string. Same attribute name, unrelated call.",
    "interface/features.py::_remote_server":
        "NOT A PATH - memsom_tuning.resolve(\"remote.action_gate_mode\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "interface/mcp.py::_checked_vault":
        "NOT A PATH - memsom_tuning.resolve(\"obsidian.vault\") reads a config "
        "knob to get the trusted root; the model-supplied `raw` path is then "
        "fenced with safe_join(root, str(raw), ...) on the very next line. Same "
        "attribute name, unrelated call.",
    "interface/mcp.py::_mcp_channel_ceiling":
        "NOT A PATH - memsom_tuning.resolve(\"mcp.channel_ceiling\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "interface/mcp.py::_mcp_export_dir":
        "NOT A PATH - memsom_tuning.resolve(\"mcp.export_dir\") reads a config "
        "knob. Same attribute name, unrelated call.",
    "interface/remote.py::handle_request":
        "NOT A PATH - memsom_tuning.resolve(\"remote.action_gate_mode\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "interface/saveall.py::start":
        "NOT A PATH - memsom_tuning.resolve(\"saveall.userprofile_fallback\") "
        "reads a config knob used as a subprocess cwd fallback. Same attribute "
        "name, unrelated call.",
    "interface/serve.py::_maybe_wrap_tls":
        "NOT A PATH - memsom_tuning.resolve(\"remote.tls_cert\") and "
        "memsom_tuning.resolve(\"remote.tls_key\") read config knobs. Same "
        "attribute name, unrelated call.",
    "interface/telemetry.py::_consolidation_dir":
        "NOT A PATH - memsom_tuning.resolve(\"telemetry.consolidation_dir\") "
        "reads a config knob. Same attribute name, unrelated call.",
    "interface/telemetry.py::_session_count":
        "NOT A PATH - memsom_tuning.resolve(\"telemetry.episodic_db\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "lifecycle/contradict.py::_anchor":
        "NOT A PATH - memsom_tuning.resolve(\"contradict.anchor\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "lifecycle/contradict.py::_default_nli":
        "NOT A PATH - memsom_tuning.resolve(\"contradict.nli_enabled\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "lifecycle/contradict.py::_enforce_default":
        "NOT A PATH - memsom_tuning.resolve(\"contradict.enforce\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "lifecycle/contradict.py::_nli_model_name":
        "NOT A PATH - memsom_tuning.resolve(\"contradict.nli_model\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "lifecycle/contradict.py::_nli_threshold":
        "NOT A PATH - memsom_tuning.resolve(\"contradict.nli_threshold\") reads "
        "a config knob. Same attribute name, unrelated call.",
    "lifecycle/contradict.py::enabled":
        "NOT A PATH - memsom_tuning.resolve(\"contradict.enabled\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "lifecycle/doctor.py::_ollama_status":
        "NOT A PATH - memsom_tuning.resolve(\"retrieval.embed_url\") and "
        "memsom_tuning.resolve(\"retrieval.embed_model\") read config knobs. "
        "Same attribute name, unrelated call.",
    "lifecycle/verify_stale.py::_threshold_days":
        "NOT A PATH - memsom_tuning.resolve(\"lifecycle.verify_stale_days\") "
        "reads a config knob. Same attribute name, unrelated call.",
    "retrieval/code_index.py::_enabled":
        "NOT A PATH - memsom_tuning.resolve(\"code_rag.enabled\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "retrieval/embed.py::_device":
        "NOT A PATH - memsom_tuning.resolve(\"retrieval.bge_device\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "retrieval/embed.py::_maxlen":
        "NOT A PATH - memsom_tuning.resolve(\"retrieval.colbert_maxlen\") reads "
        "a config knob. Same attribute name, unrelated call.",
    "retrieval/embed.py::backend":
        "NOT A PATH - memsom_tuning.resolve(\"embed.backend\") reads a config "
        "knob. Same attribute name, unrelated call.",
    "retrieval/embed.py::colbert_candidates":
        "NOT A PATH - memsom_tuning.resolve(\"retrieval.colbert_candidates\") "
        "reads a config knob. Same attribute name, unrelated call.",
    "retrieval/embed.py::bge_idle_ttl":
        "NOT A PATH - memsom_tuning.resolve(\"retrieval.bge_idle_ttl\") reads "
        "the in-process idle keep-alive knob. Same attribute name, unrelated call.",
    "retrieval/bge_client.py::_url":
        "NOT A PATH - memsom_tuning.resolve(\"retrieval.bge_url\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "retrieval/embed.py::_mode_on":
        "NOT A PATH - memsom_tuning.resolve(key) reads a boolean config knob by "
        "key. Same attribute name, unrelated call.",
    "retrieval/embed.py::encode_via":
        "NOT A PATH - memsom_tuning.resolve(\"retrieval.bge_encode_via\") reads "
        "a config knob. Same attribute name, unrelated call.",
    "retrieval/retrieve.py::_call_ollama_embed":
        "NOT A PATH - memsom_tuning.resolve(\"retrieval.embed_timeout\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "retrieval/llm.py::_cite_overlap_floor":
        "NOT A PATH - memsom_tuning.resolve(\"llm.cite_overlap\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "retrieval/llm.py::keep_alive":
        "NOT A PATH - memsom_tuning.resolve(\"llm.ollama_keep_alive\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "retrieval/llm.py::resolve":
        "NOT A PATH - this IS memsom.retrieval.llm.resolve(), the model/base_url "
        "resolver itself: memsom_tuning.resolve(\"llm.model\") and "
        "memsom_tuning.resolve(\"llm.url\") read config knobs inside it. Same "
        "attribute name as Path.resolve(), unrelated call.",
    "retrieval/qwen_embed.py::_url":
        "NOT A PATH - memsom_tuning.resolve(\"code_rag.qwen_url\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "retrieval/retrieve.py::_cmd_reindex":
        "NOT A PATH - memsom_tuning.resolve(\"retrieval.bge_unload\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "retrieval/retrieve.py::_embed_model":
        "NOT A PATH - memsom_tuning.resolve(\"retrieval.embed_model\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "retrieval/retrieve.py::_embed_url":
        "NOT A PATH - memsom_tuning.resolve(\"retrieval.embed_url\") reads a "
        "config knob. Same attribute name, unrelated call.",
    "retrieval/warm.py::disabled_by_env":
        "NOT A PATH - memsom_tuning.resolve(\"retrieval.warm_disabled\") reads "
        "a config knob. Same attribute name, unrelated call.",
    "storage/schema.py::clearance_ceiling":
        "NOT A PATH - memsom_tuning.resolve(\"integrity.clearance_ceiling\") "
        "reads a config knob. Same attribute name, unrelated call.",
}


def _resolve_sites() -> dict[str, list[int]]:
    """Every `.resolve()` call in the package, keyed by file::function."""
    found: dict[str, list[int]] = {}
    for f in sorted(PKG.rglob("*.py")):
        rel = f.relative_to(PKG).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue

        scopes = [
            (n.lineno, getattr(n, "end_lineno", n.lineno), n.name)
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        def owner(line: int) -> str:
            best = None
            for start, end, name in scopes:
                if start <= line <= end and (best is None or start > best[0]):
                    best = (start, name)
            return best[1] if best else "<module>"

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "resolve"
            ):
                found.setdefault(f"{rel}::{owner(node.lineno)}", []).append(node.lineno)
    return found


def test_no_unreviewed_resolve_site():
    """A `.resolve()` in a function nobody signed off on is a new fence.

    If this fails on code you just wrote, the question to answer is: does this
    call touch a string that came from a model, a request body, a store column,
    or a synced file? If yes, it belongs behind `memsom.paths.safe_join`. If no,
    add it to ALLOWED with which of SELF / ROOT / PRIMITIVE it is.
    """
    found = _resolve_sites()
    unreviewed = sorted(set(found) - set(ALLOWED))
    assert not unreviewed, (
        "unreviewed .resolve() call site(s) — route untrusted input through "
        "memsom.paths.safe_join, or add an annotated entry to ALLOWED in "
        + __file__
        + ":\n"
        + "\n".join(f"  memsom/{k}  (line{'s' if len(found[k]) > 1 else ''} "
                    f"{', '.join(str(n) for n in found[k])})" for k in unreviewed)
    )


def test_allowlist_has_no_dead_entries():
    """A stale entry silently widens the gate.

    If a function is deleted or renamed and its allowlist entry stays, the next
    function to take that name inherits permission it never earned.
    """
    found = _resolve_sites()
    dead = sorted(set(ALLOWED) - set(found))
    assert not dead, (
        "ALLOWED names function(s) that no longer call .resolve(); remove them:\n"
        + "\n".join(f"  {k}" for k in dead)
    )


@pytest.mark.parametrize("key,reason", sorted(ALLOWED.items()))
def test_every_allowlist_entry_states_its_category(key, reason):
    """An entry without a category is an unexplained suppression."""
    assert any(reason.startswith(c) for c in ("SELF", "ROOT", "PRIMITIVE", "NOT A PATH")), (
        f"{key} has no category — say which of SELF / ROOT / PRIMITIVE / NOT A PATH it is"
    )


def test_the_sweep_actually_finds_things():
    """Premise check: if the AST pass silently stopped matching, both gates above
    would pass vacuously and this file would be decoration."""
    found = _resolve_sites()
    assert len(found) >= 5, f"AST sweep found only {len(found)} sites — it is probably broken"
    assert "paths.py::safe_join" in found, "the sweep cannot even see the primitive itself"


def _join_then_resolve_sites() -> list[str]:
    """`(a / b).resolve()` — resolve applied straight to a joined path.

    This is the resolve-before-fence shape itself, matched structurally rather
    than by text. A textual check here is worse than useless: the first version
    of this test failed on the *comment* in obsidian.py that explains what the
    old form was, which is the exact way a source-shape gate becomes something
    people delete instead of fix.
    """
    out: list[str] = []
    for f in sorted(PKG.rglob("*.py")):
        rel = f.relative_to(PKG).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "resolve"):
                continue
            recv = node.func.value
            if isinstance(recv, ast.BinOp) and isinstance(recv.op, ast.Div):
                out.append(f"memsom/{rel}:{node.lineno}")
    return out


def _resolve_then_contain_sites() -> list[str]:
    """`x.resolve().is_relative_to(...)` — the fence one step too late.

    The predicate is correct; the ordering is not. By the time `is_relative_to`
    runs, `resolve()` has already made the syscall — and for a UNC path that
    syscall is a DNS lookup, a TCP/445 connection and an NTLM exchange offering
    this process's credentials to a host the attacker named.
    """
    out: list[str] = []
    for f in sorted(PKG.rglob("*.py")):
        rel = f.relative_to(PKG).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("is_relative_to", "startswith", "commonpath")):
                continue
            recv = node.func.value
            if (isinstance(recv, ast.Call)
                    and isinstance(recv.func, ast.Attribute)
                    and recv.func.attr == "resolve"):
                out.append(f"memsom/{rel}:{node.lineno}")
    return out


def test_no_join_then_resolve():
    """`(root / untrusted).resolve()` must not exist anywhere in the package.

    Both halves of it are wrong. `/` DISCARDS `root` as soon as `untrusted`
    carries a drive or a UNC prefix, so the join is not containment; and the
    resolve then touches the network before anything has decided it wanted to.
    """
    sites = _join_then_resolve_sites()
    assert not sites, (
        "resolve-before-fence: `(a / b).resolve()` at\n"
        + "\n".join(f"  {s}" for s in sites)
        + "\nUse memsom.paths.safe_join(a, b) — it fences on the string first."
    )


def test_no_resolve_then_containment_check():
    sites = _resolve_then_contain_sites()
    assert not sites, (
        "containment checked AFTER resolve() — the syscall already happened:\n"
        + "\n".join(f"  {s}" for s in sites)
        + "\nUse memsom.paths.safe_join, which decides before touching the disk."
    )


def test_the_shape_detectors_actually_detect():
    """Premise check for the two gates above.

    Both assert an empty list, so a detector that silently stopped matching
    would leave them passing forever. Feed each one the shape it hunts and
    require a hit.
    """
    joined = ast.parse("(root / user_input).resolve()")
    node = joined.body[0].value
    assert (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "resolve"
            and isinstance(node.func.value, ast.BinOp)
            and isinstance(node.func.value.op, ast.Div)), "join-then-resolve matcher is broken"

    late = ast.parse("p.resolve().is_relative_to(root)")
    node = late.body[0].value
    recv = node.func.value
    assert (isinstance(node.func, ast.Attribute)
            and node.func.attr == "is_relative_to"
            and isinstance(recv, ast.Call)
            and recv.func.attr == "resolve"), "resolve-then-contain matcher is broken"


def test_the_fixed_sites_no_longer_resolve_untrusted_input():
    """The three sites this seam repaired must not have regressed."""
    found = _resolve_sites()
    assert "integrity/redact.py::_unlink_within" not in found, (
        "_unlink_within resolves again — it must fence with safe_join first"
    )


def test_fixed_sites_call_the_shared_primitive():
    """Positive half of the check above: they route through safe_join."""
    for src in ("integrity/redact.py", "integrity/tombstone.py", "bridge/obsidian.py"):
        text = (PKG / src).read_text(encoding="utf-8")
        assert "safe_join(" in text, f"{src} does not use the shared primitive"
        assert "from memsom.paths import" in text, f"{src} does not import it"
