"""Contained filesystem paths — the one place untrusted text becomes a path.

WHY THIS MODULE EXISTS, in one paragraph, because the alternative keeps getting
rewritten from scratch and getting it wrong the same way each time:

``Path.resolve()``, ``Path.is_file()`` and ``os.stat()`` read like predicates.
They are not. On Windows ``resolve()`` is ``ntpath.realpath`` -> ``nt._getfinalpathname``
(a ``CreateFileW``) plus an unconditional ``stat()``, and for a path of the form
``\\\\host\\share\\x`` that goes through the MUP: DNS or NetBIOS resolution of
*host*, TCP/445, and an NTLM session setup that offers a challenge-response for
whatever account this process runs as. A fence written as "join it under my root,
resolve it, then check containment" therefore performs an outbound authenticated
network call BEFORE it decides whether it wanted to. It gets worse: both
``pathlib``'s ``/`` and ``os.path.join`` DISCARD the left operand as soon as the
right one carries a drive or a root, so ``root / user_input`` is not contained by
construction — ``PureWindowsPath(r'C:\\a') / '//h/s/x'`` is ``\\\\h\\s\\x``,
measured, no I/O.

So: fence FIRST, with string arithmetic, and only then touch the disk. Every
rule below is enforced on the STRING, before any syscall.

Rules, in order:

1. **Absolute and re-anchoring components are refused.** A drive letter (``C:``),
   a POSIX root (``/``), a UNC prefix (``\\\\host\\share``), a Win32 device
   namespace prefix (``\\\\?\\``, ``\\\\.\\``). With ``allow_absolute=True`` an
   absolute component is permitted iff it is LEXICALLY inside the root — checked
   with ``os.path.normcase``, because Windows paths are case-insensitive and a
   ``startswith`` on raw strings is a fence a different capitalisation walks
   through. UNC and device prefixes are refused even then: there is no root they
   could legitimately be inside.
2. **``..`` is refused outright**, not normalised. ``normpath`` would collapse it
   safely, but a hard refusal is auditable in a way "we normalise and then
   compare" is not, and no legitimate caller in this codebase emits one. If you
   need it, delete the ``..`` branch in ``_check_segment`` and rely on the
   containment check; that is safe, and it is a decision someone should make on
   purpose.
3. **NUL and leading/trailing whitespace are refused.** A trailing ``\\n`` is the
   ``$``-vs-``\\Z`` bug arriving from a different direction; a NUL truncates the
   name at the Win32 layer.
4. **Windows name traps are refused**: a segment containing ``:`` (drive or
   alternate data stream), a segment with a trailing dot or space (Win32 strips
   them, so ``x.`` and ``x`` are the same file but not the same string), and the
   reserved device names (``CON``, ``NUL``, ``COM1``...) with or without an
   extension.
5. **The ROOT is resolved once**; the candidate is normalised with
   ``os.path.normpath``, which is pure string arithmetic and touches nothing.
6. **Containment is checked on the normalised strings**, normcase-folded.
7. **Only then is the filesystem touched**, and only optionally: with
   ``resolve_symlinks=True`` (the default) the already-contained candidate is
   resolved and re-checked, which catches a symlink planted INSIDE the root
   pointing out of it. That resolve is the only syscall in this function and it
   runs on a path already proved contained. Pass ``resolve_symlinks=False`` where
   even that is unwanted (a hot path, or a caller that will open the file with an
   O_NOFOLLOW-equivalent anyway).

WHAT THIS DOES NOT DO. It does not create directories — a fence that mkdirs is a
fence that can be used to create directories. It does not check existence,
permissions, or type; that is the caller's job after the path is proved safe. It
does not defend against a symlink planted inside the root by someone who already
has write access there (that is same-user code execution, which owns the machine
anyway). And it is not a substitute for asking whether the ROOT itself should be
caller-chosen — ``safe_join(attacker_root, "x")`` is perfectly contained and
perfectly useless.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

_NT = os.name == "nt"

#: Win32 reserved device names. Reserved with ANY extension, so `nul.txt` too.
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{d}" for d in "123456789"),
    *(f"LPT{d}" for d in "123456789"),
}

_SEP_SPLIT = re.compile(r"[\\/]+")


class UnsafePath(ValueError):
    """A path component was refused before it reached the filesystem."""


def _reject(msg: str) -> None:
    raise UnsafePath(msg)


def _check_segment(seg: str) -> None:
    if seg in ("", "."):
        return
    if seg == "..":
        _reject("path component contains '..'")
    if not _NT:
        return
    if ":" in seg:
        _reject(f"path segment contains ':' (drive or ADS): {seg!r}")
    if seg.rstrip(" .") != seg:
        _reject(f"path segment has a trailing dot or space: {seg!r}")
    if seg.split(".", 1)[0].upper() in _RESERVED:
        _reject(f"path segment is a reserved device name: {seg!r}")


def _scrub(raw, *, allow_absolute: bool) -> tuple[str, bool]:
    """Validate one component as a STRING. Returns (component, is_absolute)."""
    if not isinstance(raw, str):
        _reject(f"path component must be str, got {type(raw).__name__}")
    if raw == "":
        _reject("empty path component")
    if "\x00" in raw:
        _reject("path component contains NUL")
    if raw != raw.strip():
        _reject("path component has leading/trailing whitespace (incl. newline)")

    w = PureWindowsPath(raw)
    drive = w.drive
    if drive.startswith("\\\\") or drive.startswith("//"):
        # UNC (\\host\share) and the device namespaces (\\?\, \\.\). Never a
        # legitimate component, with or without allow_absolute: this is the form
        # that makes resolve() open an outbound SMB session.
        _reject(f"path component is a UNC or device path: {raw!r}")
    is_abs = bool(drive) or bool(w.root) or PurePosixPath(raw).is_absolute()
    if is_abs and not allow_absolute:
        _reject(f"path component is absolute: {raw!r}")

    # Segment rules run on the part AFTER the drive letter; the drive itself is
    # judged by the containment check, not by the segment rules.
    body = raw[len(drive):] if drive else raw
    for seg in _SEP_SPLIT.split(body):
        _check_segment(seg)
    return raw, is_abs


def safe_join(root, *parts: str, allow_absolute: bool = False,
              resolve_symlinks: bool = True) -> Path:
    """``root`` joined with ``parts``, proved contained BEFORE any syscall.

    :param root: the containing directory. Trusted: it is resolved, and it is
        the caller's responsibility that it is not itself attacker-chosen.
    :param parts: untrusted components. Separators are allowed inside a part
        (``"sub/deep/ok.md"`` is fine); ``..`` is not.
    :param allow_absolute: accept an absolute FIRST component if it is lexically
        inside *root*. Use for tools whose documented contract already accepts
        absolute paths (``file_read``); leave False everywhere else.
    :param resolve_symlinks: after containment is proved lexically, resolve the
        candidate and re-check. The only syscall this function makes.
    :raises UnsafePath: on any rule violation. Subclasses ``ValueError``, so
        existing ``except ValueError`` handlers keep working.
    """
    if not parts:
        _reject("safe_join needs at least one path component")

    root_s = str(Path(root).resolve())
    root_key = os.path.normcase(root_s)

    scrubbed: list[str] = []
    for raw in parts:
        comp, is_abs = _scrub(raw, allow_absolute=allow_absolute)
        if is_abs:
            if scrubbed:
                _reject("an absolute component may only appear first")
            norm = os.path.normpath(comp)
            key = os.path.normcase(norm)
            if not (key == root_key or key.startswith(root_key + os.sep)):
                # A different-but-equivalent SPELLING of an inside-the-root path
                # (Windows 8.3 short names like RUNNER~1, macOS /var ->
                # /private/var) fails the lexical compare because only the root
                # was canonicalized. Canonicalize the candidate and re-check
                # before refusing; still fail-closed if it truly lies outside.
                # comp was scrubbed above (no UNC/device), so realpath cannot
                # touch a network namespace, and the post-join resolve re-check
                # below still guards the symlink case.
                norm = os.path.realpath(norm)
                key = os.path.normcase(norm)
                if not (key == root_key or key.startswith(root_key + os.sep)):
                    _reject(f"absolute path is outside the root: {comp!r}")
            comp = os.path.relpath(norm, root_s)
            if comp == ".":
                comp = ""
        if comp:
            scrubbed.append(comp)

    cand = os.path.normpath(os.path.join(root_s, *scrubbed)) if scrubbed \
        else root_s
    ckey = os.path.normcase(cand)
    if not (ckey == root_key or ckey.startswith(root_key + os.sep)):
        _reject("joined path escapes the root")

    out = Path(cand)
    if not resolve_symlinks:
        return out
    real = out.resolve()
    rkey = os.path.normcase(str(real))
    if not (rkey == root_key or rkey.startswith(root_key + os.sep)):
        _reject("path resolves (via a link) outside the root")
    return real


def is_unc_or_device(raw) -> bool:
    """True if *raw* names a UNC share or a Win32 device path.

    The cheap pre-check for the sites that are NOT joining under a root and so
    cannot use ``safe_join`` — ``os.path.isdir(cwd)`` on a request field,
    ``Path(model).is_file()``, ``_stem_of_path``. Pure string arithmetic.
    """
    if not isinstance(raw, str) or not raw:
        return False
    drive = PureWindowsPath(raw).drive
    return drive.startswith("\\\\") or drive.startswith("//")
