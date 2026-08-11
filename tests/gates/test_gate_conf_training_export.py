"""GATE for MS-01 -- confidentiality must gate the training export.

CONTROL-TESTED: every test in this file was run WITHOUT its xfail marker
against memsom @ 9d165b1 and FAILED, proving each was a real gap. Phase 1
landed the fix: all three now XPASS(strict), markers removed in that commit,
`test_control_the_gate_is_not_vacuous` stays green throughout.


WHAT A FIX LOOKS LIKE
---------------------
  * `distill.export_training` and `reflex.eligible_consolidated` take an
    explicit clearance ceiling and default it CLOSED (lowest), not open.
  * `reflex._tainted_node_ids` (reflex.py:284) gains `conf_label > <ceiling>`
    and `archived = 1`.
  * `schema.taint_filter_clauses` either makes `clearance` mandatory or
    defaults it to 0; `clearance=None` currently means "emit no conf_label
    clause at all" (schema.py:457), which is fail-open.
"""

import json

import pytest

import memsom
from memsom.distill import distill as memsom_distill
from memsom.integrity import confid as memsom_confid
from memsom.lifecycle import reflex as memsom_reflex
from memsom.storage import schema as memsom_schema

MARKER = "PASSPHRASE-9f3a2b"
SECRET = f"The lighthouse CA passphrase is {MARKER}, topsecret compartment only."


def _topsecret_answer(conn):
    src = memsom.insert_node(conn, SECRET, "user", source_ref="vault:ca-key")
    memsom_confid.classify(conn, src, "topsecret")
    conn.commit()
    nid, _ = memsom.derive_node(
        conn, f"Q: passphrase\nA (composed from 1 live sources):\n- {SECRET} "
              f"[mem:{src}|user]", [src])
    memsom_confid.recompute_conf(conn, nid)
    conn.commit()
    assert conn.execute("SELECT conf_label FROM nodes WHERE id=?",
                        (nid,)).fetchone()[0] == 3
    return src, nid


def test_topsecret_content_is_not_in_the_training_export(conn):
    _topsecret_answer(conn)
    records = memsom_distill.export_training(conn, min_integrity=1)
    assert MARKER not in json.dumps(records), (
        "topsecret payload was exported into training data -- model weights are "
        "the one sink in memsom that tombstone/redact/cascade cannot reach")


def test_reflex_backstop_treats_a_topsecret_node_as_tainted(conn):
    _src, nid = _topsecret_answer(conn)
    assert nid in memsom_reflex._tainted_node_ids(conn), (
        "the REFLEX-1 backstop -- which exists to catch a node slipping past the "
        "selection gate -- does not consider confidentiality at all")


def test_taint_primitive_defaults_closed_on_confidentiality(conn):
    clauses, params = memsom_schema.taint_filter_clauses(conn)
    assert any("conf_label" in c for c in clauses), (
        "the shared taint primitive's DEFAULT drops the entire confidentiality "
        "axis; 6 of its 10 call sites -- including the training export "
        "(reflex.py:131) and the always-loaded MEMORY.md digest (digest.py:120) "
        "-- rely on that default")


def _public_answer(conn):
    """Same shape as _topsecret_answer but stays at the DEFAULT conf_label (0,
    PUBLIC) -- isolates the REVOKE dimension from the (now-enforced)
    confidentiality dimension so this control keeps testing what it always
    tested."""
    src = memsom.insert_node(conn, SECRET, "user", source_ref="vault:ca-key")
    nid, _ = memsom.derive_node(
        conn, f"Q: passphrase\nA (composed from 1 live sources):\n- {SECRET} "
              f"[mem:{src}|user]", [src])
    return src, nid


def test_control_the_gate_is_not_vacuous(conn):
    """GREEN today and must stay green: the export DOES discriminate on the
    dimensions it actually implements -- specifically tombstone, isolated
    from the confidentiality dimension MS-01/MS-02 added above."""
    src, _nid = _public_answer(conn)
    assert memsom_distill.export_training(conn, min_integrity=1), "precondition"
    memsom.revoke_cascade(conn, src, "control")
    conn.commit()
    records = memsom_distill.export_training(conn, min_integrity=1)
    assert MARKER not in json.dumps(records)
