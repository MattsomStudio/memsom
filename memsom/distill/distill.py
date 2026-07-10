"""memsom_distill — provenance-filtered training-set export and distillation planner.

PURPOSE: the weights/distillation app, ACTIVE up to the GPU boundary.
Low-trust memory must NEVER bake into weights — that is the security thesis.
This module exports a clean, provenance-verified training set and writes a
ready-to-edit runner + config for the local Ollama/unsloth stack.  It NEVER
executes GPU work or `ollama` itself (deterministic, unattended-safe).

Deliberate scope: the 'may attempt ollama create' in the original design note
has been softened here: the module writes the runner and documents the one
manual GPU step, but does NOT spawn any child process.  ollama detection is via
shutil.which only, reflected in the returned plan text.

Public API
----------
migrate(conn)
export_training(conn, min_integrity=1) -> list[dict]
write_jsonl(path, records) -> None
distill_plan(model=None, out_dir=None) -> str

CLI
---
export-training <out.jsonl> [--min-integrity NAME]
distill-plan    [--model NAME] [--out-dir P]

register(subparsers) mounts this into a unified CLI.
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

import memsom
from memsom.integrity import quarantine as memsom_quarantine
from memsom.integrity import redact as memsom_redact
from memsom.storage import schema as memsom_schema

# ---------------------------------------------------------------------------
# Ancestor CTE — walks UP the DAG from a node to find all ancestors.
# We need LIVE (tombstoned=0) ancestors whose channel='external'.
# ---------------------------------------------------------------------------

_ANC_CTE = """
WITH RECURSIVE anc(id) AS (
    SELECT e.parent FROM edges e WHERE e.child = ?
    UNION
    SELECT e.parent FROM edges e JOIN anc a ON e.child = a.id
)
SELECT COUNT(*) FROM nodes n
WHERE n.id IN (SELECT id FROM anc)
  AND n.channel = 'external'
  AND n.tombstoned = 0
"""


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(conn):
    """Ensure redact and quarantine columns are present. No new columns of its own."""
    memsom_redact.migrate(conn)
    memsom_quarantine.migrate(conn)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_min_integrity(val):
    """Accept an int or a RANK name string; return an integer label floor.

    Raises ValueError for unrecognised strings or out-of-range ints.
    """
    if isinstance(val, int):
        if val not in memsom.NAME:
            raise ValueError(f"integrity floor {val!r} out of range (0-3)")
        return val
    s = str(val).strip().lower()
    # Accept full channel names as used in RANK dict
    if s in memsom.RANK:
        return memsom.RANK[s]
    # Also accept the display names from NAME (upper or mixed)
    name_map = {v.lower(): k for k, v in memsom.NAME.items()}
    if s in name_map:
        return name_map[s]
    # Accept plain integers passed as strings
    try:
        n = int(s)
    except ValueError:
        raise ValueError(f"unrecognised integrity name: {val!r}")
    if n not in memsom.NAME:
        raise ValueError(f"integrity floor {n!r} out of range (0-3)")
    return n


def _has_live_external_ancestor(conn, nid):
    """Return True if node *nid* has at least one LIVE ancestor with channel='external'."""
    row = conn.execute(_ANC_CTE, (nid,)).fetchone()
    return bool(row and row[0] > 0)


def _instruction_from_content(content):
    """Extract the instruction from a node's content.

    If content starts with 'Q: ' (the memsom.compose format), strip that prefix
    and return the first line (up to 200 chars).
    Otherwise return the first non-empty line, truncated to 200 chars.
    """
    if not content:
        return ""
    first_line = content.splitlines()[0] if content.splitlines() else content
    if first_line.startswith("Q: "):
        return first_line[3:].strip()[:200]
    # Fall back: first non-empty line
    for line in content.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_training(conn, min_integrity=1):
    """Return a list of training records from clean, provenance-verified nodes.

    Selection criteria (all must hold — belt-and-braces):
      - channel = 'agent-derived'         (answers only, not raw sources)
      - tombstoned = 0                    (alive)
      - redacted = 0                      (content intact)
      - status != 'quarantined'           (not held for review)
      - label >= min_integrity            (label floor)
      - NO live external-channel ancestor (ancestor CTE taint check)

    The last two checks are deliberately redundant: the label floor alone would
    miss an externally-tainted node that had its label manually elevated; the
    ancestor CTE alone would miss a node with label=0 that has no external
    ancestors but was still derived from untrusted material.  Both must pass.

    Each record:
      {
        'instruction': str,   # originating question (first line, 'Q: ' stripped)
        'input':       '',
        'output':      str,   # full node content
        'provenance':  [{'id': int, 'channel': str, 'label': int}, ...]
                              # immediate parents, label DESC order
      }
    """
    migrate(conn)

    floor = _parse_min_integrity(min_integrity)

    # Fetch candidate rows.  The redacted column may not exist in a fresh DB
    # before migrate() is called — migrate() guarantees it exists first.
    # DISTILL-AUDIT-1: also exclude archived (consolidated-away) nodes when the
    # column exists — a superseded node must not bake into the distilled weights,
    # matching the taint dimensions every other read path enforces.
    archived_clause = ""
    if memsom_schema.column_exists(conn, "nodes", "archived"):
        archived_clause = " AND archived=0"
    rows = conn.execute(
        "SELECT id, content FROM nodes"
        " WHERE channel='agent-derived'"
        "   AND tombstoned=0"
        "   AND redacted=0"
        "   AND status != 'quarantined'"
        + archived_clause +
        "   AND label >= ?"
        " ORDER BY id",
        (floor,)
    ).fetchall()

    records = []
    for nid, content in rows:
        # Ancestor taint check
        if _has_live_external_ancestor(conn, nid):
            continue
        # Build provenance from immediate parents
        parent_rows = memsom.parents_of(conn, nid)
        # parents_of returns tuples: (id, content, channel, label, source_ref, ...)
        # label DESC order is already applied by parents_of
        provenance = [
            {"id": r[0], "channel": r[2], "label": r[3]}
            for r in parent_rows
        ]
        records.append({
            "instruction": _instruction_from_content(content),
            "input": "",
            "output": content,
            "provenance": provenance,
        })
    return records


def write_jsonl(path, records):
    """Write *records* as line-delimited JSON to *path*, UTF-8, no BOM."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def distill_plan(model=None, out_dir=None):
    """Write distill_config.json and distill.ps1; return a multi-line plan text.

    Parameters
    ----------
    model   : str or None  — model name; falls back to MEMDAG_LLM_MODEL env var,
                             then 'qwen3-abliterated:30b-a3b'.
    out_dir : str or None  — destination directory; defaults to the memsom package dir.

    Written files
    -------------
    <out_dir>/distill_config.json   — {model, base_model, dataset, min_integrity, output}
    <out_dir>/distill.ps1           — PowerShell runner stub with clear TODO markers

    Returns
    -------
    str — multi-line plan text (model name, paths, ollama detection, GPU boundary note).

    NOTE: ollama is DETECTED (shutil.which) but NEVER executed.  The fine-tune
    itself is the one manual GPU step that memsom deliberately does not automate.
    """
    model = model or os.environ.get("MEMDAG_LLM_MODEL") or "qwen3-abliterated:30b-a3b"
    # DISTILL-1: `model` is interpolated raw into the generated distill.ps1 (FROM
    # line + Modelfile here-string). Constrain it to the Ollama model-name charset
    # so a crafted --model / env value can't inject PowerShell into the emitted
    # artifact (codegen hygiene — memsom never executes the stub, but the operator
    # might).
    if not re.fullmatch(r"[A-Za-z0-9._:/-]+", model):
        raise ValueError(
            f"invalid model name {model!r}: allowed chars are letters, digits, . _ : / -"
        )
    if out_dir is not None:
        out_dir = Path(out_dir)
    else:
        out_dir = Path(memsom.__file__).parent

    dataset_path = out_dir / "training.jsonl"
    config_path = out_dir / "distill_config.json"
    ps1_path = out_dir / "distill.ps1"

    # --- Write distill_config.json ---
    config = {
        "model": model,
        "base_model": model,
        "dataset": str(dataset_path),
        "min_integrity": 1,
        "output": "memsom-distilled",
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Write distill.ps1 ---
    modelfile_example = (
        "FROM " + model + "\n"
        "SYSTEM \"\"\"You are a memory-grounded assistant.  "
        "Answer only from your trained context.\"\"\"\n"
    )
    ps1_lines = [
        "# distill.ps1 — memsom distillation runner stub",
        "# Generated by memsom_distill.distill_plan()",
        "# Edit the TODO block below to match your unsloth/QLoRA setup.",
        "",
        "Set-StrictMode -Version Latest",
        "$ErrorActionPreference = 'Stop'",
        "",
        "# ---- Step 1: export the clean training set ----",
        "# Runs on any machine (no GPU required).",
        "python memsom_cli.py export-training training.jsonl --min-integrity agent-derived",
        "",
        "# ---- Step 2: fine-tune (MANUAL GPU STEP) ----",
        "# TODO (MANUAL GPU STEP): run the unsloth/QLoRA fine-tune on the RTX 5070 box.",
        "# Uncomment and adapt the block below once your unsloth environment is ready.",
        "#",
        "# python -c @'",
        "# from unsloth import FastLanguageModel",
        "# import json",
        f'# model_name = "{model}"',
        "# max_seq_length = 2048",
        "# # load model + LoRA adapter",
        "# model, tokenizer = FastLanguageModel.from_pretrained(",
        "#     model_name=model_name, max_seq_length=max_seq_length,",
        "#     load_in_4bit=True)",
        "# model = FastLanguageModel.get_peft_model(model,",
        "#     r=16, target_modules=['q_proj','v_proj'],",
        "#     lora_alpha=16, lora_dropout=0, bias='none',",
        "#     use_gradient_checkpointing=True)",
        "# # TODO: wire your training loop here using training.jsonl",
        "# model.save_pretrained('memsom-distilled-lora')",
        "# '@",
        "",
        "# ---- Step 3: register with Ollama (after GPU step) ----",
        "# TODO: uncomment after the LoRA is trained and merged.",
        "#",
        "# $modelfile = @'",
        "# " + modelfile_example.replace("\n", "\n# "),
        "# '@",
        "# $modelfile | Out-File -Encoding utf8 Modelfile",
        "# ollama create memsom-distilled -f Modelfile",
        "",
        "Write-Host 'distill.ps1: export step complete.  Complete steps 2 and 3 manually on the GPU box.'",
    ]
    ps1_path.write_text("\n".join(ps1_lines) + "\n", encoding="utf-8")

    # --- Detect ollama ---
    ollama_found = shutil.which("ollama") is not None
    if ollama_found:
        ollama_note = "ollama detected on PATH — step 3 is ready to uncomment once your LoRA is trained."
    else:
        ollama_note = "ollama not found on PATH — install it or run step 3 on the GPU box where ollama is available."

    plan_text = "\n".join([
        f"=== memsom distill plan ===",
        f"model:            {model}",
        f"output dir:       {out_dir}",
        f"config written:   {config_path}",
        f"runner written:   {ps1_path}",
        f"dataset path:     {dataset_path}",
        f"",
        f"ollama:           {ollama_note}",
        f"",
        f"boundary note:    memsom guarantees WHAT goes into the weights — it exports only",
        f"                  agent-derived answers that are alive, unredacted, unquarantined,",
        f"                  and have NO live external-channel ancestor.  The fine-tune itself",
        f"                  is the one manual GPU step that memsom deliberately does not automate.",
        f"",
        f"next steps:",
        f"  1. Review {dataset_path} (export-training writes it)",
        f"  2. Run step 2 (GPU fine-tune) manually — see distill.ps1",
        f"  3. Uncomment step 3 in distill.ps1 and run `ollama create memsom-distilled -f Modelfile`",
    ])
    return plan_text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_export_training(args):
    conn = memsom.get_connection()
    try:
        records = export_training(conn, args.min_integrity)
        out = Path(args.out)
        write_jsonl(out, records)
        print(f"wrote {len(records)} records -> {out}")
    finally:
        conn.close()


def cmd_distill_plan(args):
    plan = distill_plan(
        model=args.model or None,
        out_dir=args.out_dir or None,
    )
    print(plan)


def register(subparsers):
    p_exp = subparsers.add_parser(
        "export-training",
        help="export a provenance-filtered JSONL training set",
    )
    p_exp.add_argument("out", metavar="out.jsonl",
                       help="output path for the JSONL file")
    p_exp.add_argument("--min-integrity", default="agent-derived",
                       dest="min_integrity",
                       help="integrity floor: name (e.g. 'agent-derived', 'user') or int 0-3")
    p_exp.set_defaults(func=cmd_export_training)

    p_plan = subparsers.add_parser(
        "distill-plan",
        help="write distill_config.json + distill.ps1 runner stub and print the plan",
    )
    p_plan.add_argument("--model", default=None,
                        help="base model name (overrides MEMDAG_LLM_MODEL env var)")
    p_plan.add_argument("--out-dir", default=None, dest="out_dir",
                        help="directory for config + runner (default: memsom package dir)")
    p_plan.set_defaults(func=cmd_distill_plan)


def main(argv=None):
    import argparse
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_distill")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
