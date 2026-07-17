"""The five builtin tools an agent can be handed.

Each is deliberately thin: stdlib only (urllib, subprocess, re, pathlib — the
repo has no third-party HTTP dependency to lean on), bounded output, and
failure-as-message where the model can usefully react (an HTTP 404 or an empty
search result is INFORMATION, not an error). :class:`ToolError` is reserved for
failures the model cannot act on — network down, path fence violation, CLI
crash, timeout.

Posture note: :class:`Shell` runs whatever it is told with no denylist. This is
the single-user lab stance the panel uses everywhere — the two-phase audit
trail the RUNNER writes around every call is the guardrail, not a capability
gate that would only ever be theater on a machine the user already owns.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from memsom.providers.base import run_no_window
from memsom.providers.tools.base import Tool, ToolContext, ToolError, truncate_output

#: repo root (…/memsom), used as cwd for CLI shell-outs so `-m` imports resolve.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_USER_AGENT = "memsom-panel-agent/1.0"
_TRUNCATION_MARK = "\n…[output truncated]"


def _cap(text: str, max_bytes: int) -> str:
    """truncate_output plus the visible marker every builtin uses."""
    text, truncated = truncate_output(text, max_bytes)
    return text + _TRUNCATION_MARK if truncated else text


# ---------------------------------------------------------------------------
# http_fetch
# ---------------------------------------------------------------------------


class _CappedRedirects(urllib.request.HTTPRedirectHandler):
    """urllib's default follows 10 redirects and any scheme the target names.
    Cap at 3 and refuse a hop off http/https (a redirect to file:// or ftp://
    is how a fetch tool gets turned into a local file reader)."""

    max_redirections = 3

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme not in ("http", "https"):
            raise ToolError(f"redirect to non-http(s) url refused: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HttpFetch(Tool):
    """GET a URL and return status + content-type + body text."""

    type = "http_fetch"
    description = (
        "Fetch a URL over HTTP GET and return the status line, content-type "
        "and body text. Non-2xx responses are returned (not errors) so you "
        "can react to them."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "http(s) URL to fetch"},
        },
        "required": ["url"],
    }
    options_schema = {
        "type": "object",
        "properties": {
            "max_bytes": {
                "type": "integer",
                "description": "max body bytes to read (default 65536)",
            },
        },
    }

    def run(self, arguments: dict, ctx: ToolContext) -> str:
        url = str(arguments.get("url") or "").strip()
        scheme = urllib.parse.urlsplit(url).scheme
        if scheme not in ("http", "https"):
            raise ToolError(f"only http/https urls are allowed, got: {url!r}")
        max_bytes = int(self.options.get("max_bytes", 65536))

        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        opener = urllib.request.build_opener(_CappedRedirects)
        try:
            resp = opener.open(req, timeout=ctx.timeout_s)
        except urllib.error.HTTPError as exc:
            # non-2xx is information for the model, not a tool failure.
            resp = exc
        except ToolError:
            raise
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ToolError(f"fetch failed: {exc}") from exc

        try:
            # belt-and-braces: enforce scheme on the FINAL url too, in case a
            # handler ahead of ours rewrote the target.
            final = resp.geturl() or url
            if urllib.parse.urlsplit(final).scheme not in ("http", "https"):
                raise ToolError(f"landed on non-http(s) url refused: {final}")
            status = getattr(resp, "status", None) or getattr(resp, "code", 0)
            reason = getattr(resp, "reason", "") or ""
            ctype = resp.headers.get("Content-Type", "unknown") if resp.headers else "unknown"
            body = resp.read(max_bytes).decode("utf-8", errors="replace")
        except ToolError:
            raise
        except OSError as exc:
            raise ToolError(f"read failed: {exc}") from exc
        finally:
            resp.close()
        return f"HTTP {status} {reason}\ncontent-type: {ctype}\n\n{body}"


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

_RESULT_LINK_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
_RESULT_SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(fragment: str) -> str:
    """Strip tags and entities from an HTML fragment, collapse whitespace."""
    return " ".join(html.unescape(_TAG_RE.sub("", fragment)).split())


def _unwrap_ddg(href: str) -> str:
    """DDG wraps result links as //duckduckgo.com/l/?uddg=<encoded-url>&…;
    unwrap to the real target, pass anything else through untouched."""
    query = urllib.parse.urlsplit(href).query
    uddg = urllib.parse.parse_qs(query).get("uddg")
    if uddg:
        return urllib.parse.unquote(uddg[0])
    return href


class WebSearch(Tool):
    """Scrape DuckDuckGo's HTML endpoint — no API key, no third-party dep."""

    type = "web_search"
    description = (
        "EXPERIMENTAL: search the web via DuckDuckGo and return the top "
        "results as title, url and snippet. Scraped HTML — may return 'no "
        "results / parse failed' when the page shape changes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "search query"},
        },
        "required": ["query"],
    }
    options_schema = {
        "type": "object",
        "properties": {
            "max_results": {
                "type": "integer",
                "description": "max results to return (default 5)",
            },
        },
    }

    def run(self, arguments: dict, ctx: ToolContext) -> str:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ToolError("web_search requires a non-empty 'query'")
        max_results = int(self.options.get("max_results", 5))

        url = ("https://html.duckduckgo.com/html/?q="
               + urllib.parse.quote_plus(query))
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=ctx.timeout_s) as resp:
                page = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ToolError(f"search fetch failed: {exc}") from exc

        # Parse failures return a string, not an exception — the model should
        # SEE that the scrape came up empty rather than the call exploding.
        try:
            links = _RESULT_LINK_RE.findall(page)
            snippets = [_clean(s) for s in _RESULT_SNIPPET_RE.findall(page)]
        except Exception:
            return "no results / parse failed"
        if not links:
            return "no results / parse failed"

        blocks = []
        for i, (href, title) in enumerate(links[:max_results], start=1):
            snippet = snippets[i - 1] if i - 1 < len(snippets) else ""
            blocks.append(
                f"{i}. {_clean(title)}\n{_unwrap_ddg(html.unescape(href))}\n{snippet}")
        return _cap("\n\n".join(blocks), ctx.max_output_bytes)


# ---------------------------------------------------------------------------
# memory_recall
# ---------------------------------------------------------------------------


class MemoryRecall(Tool):
    """Shell out to the full-stack memsom CLI (memsom.interface.cli).

    Mirrors the MCP server's ``_tool_argv`` shapes exactly: ``retrieve <query>
    --k N`` and ``ask <question>``. A subprocess (not an in-process import)
    keeps the agent runner isolated from the CLI's DB/embedder state.
    """

    type = "memory_recall"
    description = (
        "Search the memsom memory store. 'retrieve' mode returns ranked raw "
        "hits; 'ask' mode composes an answer."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "what to look up"},
        },
        "required": ["query"],
    }
    options_schema = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["retrieve", "ask"],
                "description": "retrieve = ranked hits (default); ask = composed answer",
            },
            "k": {
                "type": "integer",
                "description": "max hits in retrieve mode (default 5)",
            },
        },
    }

    def __init__(self, options: dict) -> None:
        super().__init__(options)
        self.mode = str(self.options.get("mode", "retrieve"))
        if self.mode not in ("retrieve", "ask"):
            raise ToolError(f"memory_recall mode must be retrieve|ask, got {self.mode!r}")

    def run(self, arguments: dict, ctx: ToolContext) -> str:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ToolError("memory_recall requires a non-empty 'query'")
        argv = [sys.executable, "-m", "memsom.interface.cli", self.mode, query]
        if self.mode == "retrieve":
            argv += ["--k", str(int(self.options.get("k", 5)))]
        try:
            proc = run_no_window(argv, capture_output=True, text=True,
                                 errors="replace", timeout=ctx.timeout_s,
                                 cwd=str(_REPO_ROOT))
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"memory_recall timed out after {ctx.timeout_s}s") from exc
        except OSError as exc:
            raise ToolError(f"memory_recall failed to launch: {exc}") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip()[-500:]
            raise ToolError(f"memsom {self.mode} exited {proc.returncode}: {tail}")
        return _cap(proc.stdout or "", ctx.max_output_bytes)


# ---------------------------------------------------------------------------
# file_read
# ---------------------------------------------------------------------------


class FileRead(Tool):
    """Read a text file inside a configured root — and ONLY inside it."""

    type = "file_read"
    description = (
        "Read a text file. Paths are relative to the tool's configured root "
        "directory; anything resolving outside it is refused."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "path relative to the root"},
        },
        "required": ["path"],
    }
    options_schema = {
        "type": "object",
        "properties": {
            "root": {
                "type": "string",
                "description": "directory the tool is fenced to (required)",
            },
        },
        "required": ["root"],
    }

    def __init__(self, options: dict) -> None:
        super().__init__(options)
        root = self.options.get("root")
        if not root:
            raise ToolError("file_read requires a 'root' option")
        self.root = Path(root).resolve()

    def run(self, arguments: dict, ctx: ToolContext) -> str:
        path = str(arguments.get("path") or "")
        if not path:
            raise ToolError("file_read requires a non-empty 'path'")
        # resolve() collapses any ../ the model smuggles in; the fence is on
        # the RESOLVED path, so symlinked escapes are caught too.
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ToolError("path outside root")
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ToolError(f"read failed: {exc}") from exc
        return _cap(text, ctx.max_output_bytes)


# ---------------------------------------------------------------------------
# shell
# ---------------------------------------------------------------------------


class Shell(Tool):
    """Run a command via the platform shell. No denylist — single-user lab
    posture: the runner's audit trail is the guardrail, a keyword filter would
    only be theater on a machine the user already owns."""

    type = "shell"
    description = (
        "Run a shell command (cmd /c on Windows, /bin/sh -c elsewhere) and "
        "return its exit code with merged stdout/stderr."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "command line to run"},
        },
        "required": ["command"],
    }
    options_schema = {
        "type": "object",
        "properties": {
            "cwd": {
                "type": "string",
                "description": "working directory (default: runner's cwd)",
            },
            "timeout_s": {
                "type": "integer",
                "description": "per-command timeout override in seconds",
            },
        },
    }

    def run(self, arguments: dict, ctx: ToolContext) -> str:
        command = str(arguments.get("command") or "").strip()
        if not command:
            raise ToolError("shell requires a non-empty 'command'")
        timeout = int(self.options.get("timeout_s") or ctx.timeout_s)
        # cmd /c over powershell for predictability: no profile load, no
        # execution policy, exit codes pass straight through.
        if sys.platform == "win32":
            argv = ["cmd", "/c", command]
        else:
            argv = ["/bin/sh", "-c", command]
        try:
            proc = run_no_window(argv, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 errors="replace", timeout=timeout,
                                 cwd=self.options.get("cwd") or None)
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"timed out after {timeout}s") from exc
        except OSError as exc:
            raise ToolError(f"failed to launch: {exc}") from exc
        return _cap(f"exit {proc.returncode}\n{proc.stdout or ''}",
                    ctx.max_output_bytes)
