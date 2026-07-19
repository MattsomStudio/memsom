"""memsom.interface.telemetry — request-time system telemetry for the panel server.

Machine-agnostic sampling: every section of the returned dict is best-effort and
independently guarded, so a missing dependency (psutil not installed, nvidia-smi
absent, Syncthing unreachable) degrades that one section to an "unavailable" shape
instead of raising. Nothing here caches to disk or runs a background thread — a
sample is only ever taken when the panel server (WP2, built separately) asks for
one, at request time.

Public API
----------
SystemTelemetry(profile, *, run=None, urlopen=None)
    .sample() -> dict     one point-in-time snapshot, see module-level SAMPLE_SHAPE
                          below for the exact keys.

`profile` dict (all keys optional — an absent key degrades that section, it never
raises):
  probe_targets   [{name, host, port}]   TCP reachability checks (no ICMP, ever)
  probe_timeout_s float, default 1.5
  disk_dirs       [str]                  recursive-size + in-process delta tracking
  syncthing       {base_url, config_xml} polls Syncthing's REST connections endpoint
  top_n           int, default 8         how many processes to report per ranking

Sample shape
------------
{
  "ts": iso8601,
  "ram": {"physical": {"total","available","used_pct"} | {"error"},
          "commit":   {"total","limit","used_pct"} | {"error"}},
  "top_procs": {"by_working_set": [{"pid","name","rss"}, ...],
                "by_commit":      [{"pid","name","vms"}, ...]} | {"error"},
  "cpu": {"used_pct"} | {"error"},
  "gpu": {"available": bool, "name"?, "vram_used_mb"?, "vram_total_mb"?,
          "gpu_util_pct"?, "gpu_temp_c"?, "procs"?: [{"pid","name","used_mb"}]},
  "disks": [{"path","bytes","delta_bytes"}],   # delta_bytes is None on first sample
  "probes": [{"name","host","port","ok","ms"}],
  "syncthing": {"reachable": bool, "connections"?: {...}, "error"?: str},
}

Notes on the two memory numbers (both matter, for different reasons):
  ram.physical  — psutil virtual_memory(): what's actually resident right now.
  ram.commit    — Windows "system commit charge" via the stdlib ctypes call
                  GetPerformanceInfo, the exact number Task Manager's Performance
                  tab shows. This is deliberately NOT derived from psutil (psutil
                  has no direct equivalent) — it is the real overcommit ceiling
                  (CommitLimit = RAM + pagefile), which physical-RAM-only numbers
                  can't tell you when a box is about to hit "out of memory" despite
                  having free physical RAM.
"""
from __future__ import annotations

import ctypes
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

PSUTIL_HINT = "psutil not installed - pip install memsom[panel]"


# ---------------------------------------------------------------------------
# psutil — lazy-optional. Imported inside a helper, never at module scope, so
# the core panel module keeps loading (degraded) when psutil isn't installed.
# ---------------------------------------------------------------------------

def _import_psutil():
    try:
        import psutil
    except ImportError:
        return None
    return psutil


def _physical_ram(psutil_mod):
    if psutil_mod is None:
        return {"error": PSUTIL_HINT}
    vm = psutil_mod.virtual_memory()
    return {
        "total": vm.total,
        "available": vm.available,
        "used_pct": round(float(vm.percent), 2),
    }


def _cpu_percent(psutil_mod):
    """System-wide CPU utilization, mirroring _physical_ram's lazy-psutil shape.

    interval=None is the non-blocking form: it returns the load since the LAST
    call (the first call in a process returns 0.0). The panel samples at request
    time, not on a fixed cadence, so this is "load since the previous sample" —
    good enough for a live gauge and it never blocks the request thread the way
    interval>0 would.
    """
    if psutil_mod is None:
        return {"error": PSUTIL_HINT}
    return {"used_pct": round(float(psutil_mod.cpu_percent(interval=None)), 2)}


def _top_procs(psutil_mod, top_n):
    if psutil_mod is None:
        return {"error": PSUTIL_HINT}
    rows = []
    for proc in psutil_mod.process_iter(["pid", "name", "memory_info"]):
        try:
            info = proc.info
            mem = info.get("memory_info")
            if mem is None:
                continue
            rows.append({
                "pid": info.get("pid"),
                "name": info.get("name") or "",
                "rss": mem.rss,
                "vms": mem.vms,
            })
        except (psutil_mod.NoSuchProcess, psutil_mod.AccessDenied):
            continue  # process exited or is unreadable mid-scan — skip it, not fatal
    by_ws = sorted(rows, key=lambda r: r["rss"], reverse=True)[:top_n]
    by_commit = sorted(rows, key=lambda r: r["vms"], reverse=True)[:top_n]
    return {
        "by_working_set": [{"pid": r["pid"], "name": r["name"], "rss": r["rss"]} for r in by_ws],
        "by_commit": [{"pid": r["pid"], "name": r["name"], "vms": r["vms"]} for r in by_commit],
    }


# ---------------------------------------------------------------------------
# System commit charge — stdlib ctypes GetPerformanceInfo (Windows only).
# Exact Task Manager parity; deliberately NOT psutil arithmetic (see module
# docstring). The struct is defined lazily inside the guarded branch so
# ctypes.windll — which doesn't exist off Windows — is never touched elsewhere.
# ---------------------------------------------------------------------------

def _read_commit():
    if sys.platform != "win32":
        return {"error": "windows-only"}

    from ctypes import wintypes

    class PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("CommitTotal", ctypes.c_size_t),
            ("CommitLimit", ctypes.c_size_t),
            ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t),
            ("PhysicalAvailable", ctypes.c_size_t),
            ("SystemCache", ctypes.c_size_t),
            ("KernelTotal", ctypes.c_size_t),
            ("KernelPaged", ctypes.c_size_t),
            ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t),
            ("HandleCount", wintypes.DWORD),
            ("ProcessCount", wintypes.DWORD),
            ("ThreadCount", wintypes.DWORD),
        ]

    pi = PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(PERFORMANCE_INFORMATION)
    ok = ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), ctypes.sizeof(pi))
    if not ok:
        return {"error": "GetPerformanceInfo failed"}

    page = pi.PageSize
    total = pi.CommitTotal * page
    limit = pi.CommitLimit * page
    used_pct = round(100.0 * total / limit, 2) if limit else 0.0
    return {"total": total, "limit": limit, "used_pct": used_pct}


# ---------------------------------------------------------------------------
# GPU — nvidia-smi via the injected `run`. Absent binary / timeout / nonzero
# exit all collapse to the same "no GPU visible" shape; the per-process query
# is best-effort on top of that and never turns an available GPU unavailable.
# ---------------------------------------------------------------------------

def _to_int(s):
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _read_gpu(run):
    try:
        r = run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"available": False}
    if r.returncode != 0:
        return {"available": False}
    line = (r.stdout or "").strip().splitlines()
    if not line:
        return {"available": False}
    parts = [p.strip() for p in line[0].split(",")]
    # Still gate on the three fields we've always required (name + the two VRAM
    # numbers). util/temp are additive: a driver that omits them (or the test
    # fakes that only emit three fields) must degrade to util/temp=None, never
    # collapse the whole GPU section to unavailable.
    if len(parts) < 3:
        return {"available": False}
    result = {
        "available": True,
        "name": parts[0],
        "vram_used_mb": _to_int(parts[1]),
        "vram_total_mb": _to_int(parts[2]),
        "gpu_util_pct": _to_int(parts[3]) if len(parts) > 3 else None,
        "gpu_temp_c": _to_int(parts[4]) if len(parts) > 4 else None,
    }

    try:
        rp = run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return result  # GPU itself is confirmed present; per-process list is optional
    if rp.returncode == 0:
        procs = []
        for pl in (rp.stdout or "").strip().splitlines():
            pp = [x.strip() for x in pl.split(",")]
            if len(pp) >= 3:
                used = _to_int(pp[2])
                if used is None:
                    # WDDM-mode drivers report "[N/A]" per-process — a row with
                    # no number is noise, not information; drop it.
                    continue
                procs.append({"pid": _to_int(pp[0]),
                              "name": pp[1].replace("\\", "/").rsplit("/", 1)[-1],
                              "used_mb": used})
        result["procs"] = procs
    return result


# ---------------------------------------------------------------------------
# Disk — recursive size via os.walk, delta against the previous in-process
# sample only (never persisted). Unreadable entries are skipped, not fatal.
# ---------------------------------------------------------------------------

def _dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue  # gone / permission-denied mid-walk — skip it
    return total


# ---------------------------------------------------------------------------
# Probes — raw TCP connect only. NEVER ping/ICMP.
# ---------------------------------------------------------------------------

def _probe_one(host, port, timeout):
    start = time.perf_counter()
    ok = False
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        ok = True
    except OSError:
        ok = False
    ms = round((time.perf_counter() - start) * 1000, 2)
    return ok, ms


# ---------------------------------------------------------------------------
# Syncthing — API key comes out of Syncthing's own config.xml
# (configuration/gui/apikey), never out of the profile. The key is used only
# to build the outgoing request header; it must never end up in the returned
# payload, an error message, or a log line.
# ---------------------------------------------------------------------------

def _read_syncthing(cfg, urlopen):
    if not cfg:
        return {"reachable": False, "error": "not configured"}
    base_url = cfg.get("base_url")
    config_xml = cfg.get("config_xml")
    if not base_url or not config_xml:
        return {"reachable": False, "error": "missing base_url or config_xml"}

    try:
        root = ET.parse(config_xml).getroot()
    except (OSError, ET.ParseError):
        return {"reachable": False, "error": "could not read syncthing config"}

    apikey_el = root.find("gui/apikey")
    apikey = (apikey_el.text or "").strip() if apikey_el is not None else ""
    if not apikey:
        return {"reachable": False, "error": "apikey not found in config"}

    url = base_url.rstrip("/") + "/rest/system/connections"
    req = urllib.request.Request(url, headers={"X-API-Key": apikey})
    try:
        with urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {"reachable": False, "error": "request failed"}

    return {"reachable": True, "connections": data}


class SystemTelemetry:
    """Point-in-time system sampler for the panel server. Every public method
    call is side-effect-free except the in-process disk-delta tracker."""

    def __init__(self, profile: dict, *, run=None, urlopen=None):
        self._profile = profile or {}
        self._run = run or subprocess.run
        self._urlopen = urlopen or urllib.request.urlopen
        self._prev_disk_bytes = {}  # path -> bytes from the previous sample() call

    @staticmethod
    def _safe(fn, fallback):
        """Run *fn*; on ANY exception (including bugs we didn't anticipate),
        fall back to *fallback* instead of letting one section's failure take
        down the whole sample()."""
        try:
            return fn()
        except Exception:
            return fallback

    def _ram(self, psutil_mod):
        return {"physical": _physical_ram(psutil_mod), "commit": _read_commit()}

    def _disks(self):
        out = []
        for path in self._profile.get("disk_dirs") or []:
            size = _dir_size(path)
            prev = self._prev_disk_bytes.get(path)
            out.append({
                "path": path,
                "bytes": size,
                "delta_bytes": None if prev is None else size - prev,
            })
            self._prev_disk_bytes[path] = size
        return out

    def _probes(self):
        timeout = self._profile.get("probe_timeout_s", 1.5)
        out = []
        for t in self._profile.get("probe_targets") or []:
            ok, ms = _probe_one(t.get("host"), t.get("port"), timeout)
            out.append({
                "name": t.get("name"),
                "host": t.get("host"),
                "port": t.get("port"),
                "ok": ok,
                "ms": ms,
            })
        return out

    def sample(self) -> dict:
        psutil_mod = _import_psutil()
        top_n = int(self._profile.get("top_n", 8))
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ram": self._safe(
                lambda: self._ram(psutil_mod),
                {"physical": {"error": "unavailable"}, "commit": {"error": "unavailable"}},
            ),
            "top_procs": self._safe(
                lambda: _top_procs(psutil_mod, top_n), {"error": "unavailable"}
            ),
            "cpu": self._safe(
                lambda: _cpu_percent(psutil_mod), {"error": "unavailable"}
            ),
            "gpu": self._safe(lambda: _read_gpu(self._run), {"available": False}),
            "disks": self._safe(self._disks, []),
            "probes": self._safe(self._probes, []),
            "syncthing": self._safe(
                lambda: _read_syncthing(self._profile.get("syncthing"), self._urlopen),
                {"reachable": False, "error": "unavailable"},
            ),
        }
