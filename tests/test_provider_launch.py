"""Launch options: the request body -> llama-server argv path, and the GGUF
header reader that bounds it.

The gate under test is that a knob either lands on the command line exactly as
declared, or refuses the launch with a message naming what was wrong. A knob
that is silently dropped is the failure mode worth catching: the server comes up
looking fine, on flags nobody asked for.
"""

from __future__ import annotations

import struct

import pytest

from memsom.providers import gguf
from memsom.providers.base import (
    Capabilities,
    LaunchOption,
    ProviderError,
    coerce_launch_options,
)
from memsom.providers.handlers import handle_provider_action
from memsom.providers.llamacpp import LAUNCH_OPTIONS, LlamaCppAdapter
from memsom.providers.vllm import VllmAdapter


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class FakeProcman:
    """Captures the argv instead of spawning anything."""

    def __init__(self) -> None:
        self.argv = None
        self.kwargs = None

    def start(self, key, argv, **kw):
        self.argv = argv
        self.kwargs = kw
        return {"ok": True, "pid": 4242}

    def stop(self, key):
        return {"ok": True}

    def is_running(self, key):
        return False


@pytest.fixture
def model_file(tmp_path):
    p = tmp_path / "Test-30B-A3B-Q4_K_M.gguf"
    p.write_bytes(_gguf_bytes(block_count=48, expert_count=128,
                              context_length=40960))
    return p


@pytest.fixture
def adapter(tmp_path, model_file):
    pm = FakeProcman()
    a = LlamaCppAdapter({
        "id": "llamacpp", "kind": "llamacpp", "label": "llama.cpp",
        "host": "127.0.0.1", "port": 8081, "exec": "llama-server",
        "models_dirs": [str(tmp_path)],
        "args": ["--flash-attn", "on"],
    }, procman=pm)
    a._pm = pm
    return a


# ---------------------------------------------------------------------------
# argv construction
# ---------------------------------------------------------------------------


def test_start_with_no_options_is_unchanged(adapter, model_file):
    adapter.start(model_file.name)
    assert adapter._pm.argv == [
        "llama-server", "-m", str(model_file),
        "--host", "127.0.0.1", "--port", "8081",
        "--flash-attn", "on",
    ]


def test_every_declared_knob_reaches_the_command_line(adapter, model_file):
    adapter.start(model_file.name, {
        "ctx": 16384, "n_predict": 2048, "temp": 0.6,
        "n_gpu_layers": 48, "n_cpu_moe": 30,
        "override_tensor": r"blk\.(1[5-9])\.ffn_.*_exps\.=CPU",
    })
    argv = adapter._pm.argv
    for flag, value in [("--ctx-size", "16384"), ("--n-predict", "2048"),
                        ("--temp", "0.6"), ("--n-gpu-layers", "48"),
                        ("--n-cpu-moe", "30"),
                        ("--override-tensor",
                         r"blk\.(1[5-9])\.ffn_.*_exps\.=CPU")]:
        assert flag in argv, f"{flag} was dropped"
        assert argv[argv.index(flag) + 1] == value


def test_a_flag_and_its_value_are_separate_argv_elements(adapter, model_file):
    """`--ctx-size 4096` as ONE string would reach llama-server as a single
    unknown argument; keep them split."""
    adapter.start(model_file.name, {"ctx": 4096})
    assert "--ctx-size 4096" not in adapter._pm.argv


def test_bool_knob_is_a_bare_switch(adapter, model_file):
    adapter.start(model_file.name, {"cpu_moe": True})
    argv = adapter._pm.argv
    assert "--cpu-moe" in argv
    # nothing follows it but the profile's own args
    assert argv[argv.index("--cpu-moe") + 1] == "--flash-attn"


def test_false_bool_emits_nothing(adapter, model_file):
    adapter.start(model_file.name, {"cpu_moe": False})
    assert "--cpu-moe" not in adapter._pm.argv


def test_profile_args_still_land_last(adapter, model_file):
    """The profile is trusted config; a panel knob must not be able to displace
    it by ordering."""
    adapter.start(model_file.name, {"ctx": 8192})
    argv = adapter._pm.argv
    assert argv[-2:] == ["--flash-attn", "on"]


def test_none_and_empty_values_mean_engine_default(adapter, model_file):
    adapter.start(model_file.name, {"ctx": None, "override_tensor": ""})
    assert "--ctx-size" not in adapter._pm.argv
    assert "--override-tensor" not in adapter._pm.argv


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_start_without_a_model_still_errors_clearly(adapter):
    with pytest.raises(ProviderError, match="requires a model"):
        adapter.start()


def test_unknown_option_is_refused_by_name(adapter, model_file):
    with pytest.raises(ProviderError, match="unknown launch option"):
        adapter.start(model_file.name, {"n_gpu_layerz": 4})
    assert adapter._pm.argv is None, "refused launch must not spawn"


@pytest.mark.parametrize("options, message", [
    ({"ctx": "abc"}, "must be an integer"),
    ({"temp": "hot"}, "must be a number"),
    ({"n_cpu_moe": -1}, "must be >="),
    ({"temp": 9}, "must be <="),
    ({"cpu_moe": 1}, "must be true or false"),
    ({"ctx": True}, "must be a number"),
])
def test_bad_values_are_refused(adapter, model_file, options, message):
    with pytest.raises(ProviderError, match=message):
        adapter.start(model_file.name, options)
    assert adapter._pm.argv is None


@pytest.mark.parametrize("value", [
    "rm -rf /",                     # spaces
    "blk.0=CPU; shutdown",          # shell punctuation
    "blk.0=CPU\nmalicious",         # newline
    "no-equals-sign",               # not a pattern=buffer pair
    'blk.0="CPU"',                  # quotes
])
def test_override_tensor_rejects_anything_but_a_pattern_pair(
        adapter, model_file, value):
    with pytest.raises(ProviderError, match="not a valid value"):
        adapter.start(model_file.name, {"override_tensor": value})
    assert adapter._pm.argv is None


def test_options_must_be_an_object(adapter, model_file):
    with pytest.raises(ProviderError, match="must be a JSON object"):
        adapter.start(model_file.name, ["--ctx-size", "4096"])


def test_vllm_refuses_launch_options_rather_than_ignoring_them():
    a = VllmAdapter({"id": "vllm", "kind": "vllm", "label": "vLLM",
                     "model": "m"}, procman=FakeProcman())
    with pytest.raises(ProviderError, match="accepts no launch options"):
        a.start("m", {"ctx": 4096})


# ---------------------------------------------------------------------------
# the declaration itself (this is what the UI renders)
# ---------------------------------------------------------------------------


def test_capabilities_serialize_launch_options(adapter):
    d = adapter.capabilities().as_dict()
    assert d["can_start"] is True
    keys = [o["key"] for o in d["launch_options"]]
    assert keys == ["ctx", "n_predict", "temp", "n_gpu_layers",
                    "cpu_moe", "n_cpu_moe", "override_tensor"]
    by_key = {o["key"]: o for o in d["launch_options"]}
    # the MoE knobs must announce that they only apply to MoE models, and
    # --n-cpu-moe must announce its ceiling — the UI has no other way to know.
    assert by_key["n_cpu_moe"]["requires_meta"] == "n_experts"
    assert by_key["cpu_moe"]["requires_meta"] == "n_experts"
    assert by_key["n_cpu_moe"]["max_from_meta"] == "n_layers"
    assert by_key["ctx"]["max_from_meta"] == "ctx_max"


def test_capabilities_default_to_no_launch_options():
    assert Capabilities().as_dict()["launch_options"] == []


def test_coercer_rejects_a_declared_option_it_was_not_given():
    declared = [LaunchOption(key="a", label="a", type="int")]
    assert coerce_launch_options(declared, None) == {}
    assert coerce_launch_options(declared, {}) == {}
    with pytest.raises(ProviderError):
        coerce_launch_options(declared, {"b": 1})


# ---------------------------------------------------------------------------
# model listing carries the architecture the form needs
# ---------------------------------------------------------------------------


def test_list_models_exposes_layers_experts_ctx_and_quant(adapter, model_file):
    m = next(m for m in adapter.list_models() if m.name == model_file.name)
    assert m.quant == "Q4_K_M"
    assert m.ctx_max == 40960
    assert m.meta["n_layers"] == 48
    assert m.meta["n_experts"] == 128
    assert m.meta["path"] == str(model_file)


def test_a_dense_model_reports_no_experts(tmp_path):
    p = tmp_path / "Dense-7B-Q8_0.gguf"
    p.write_bytes(_gguf_bytes(block_count=32, expert_count=None,
                              context_length=8192))
    a = LlamaCppAdapter({"id": "llamacpp", "kind": "llamacpp",
                         "models_dirs": [str(tmp_path)]}, procman=FakeProcman())
    m = next(m for m in a.list_models() if m.name == p.name)
    assert m.meta["n_layers"] == 32
    assert m.meta["n_experts"] is None


def test_an_unreadable_file_costs_only_its_own_metadata(tmp_path):
    """One junk .gguf must not take the whole model list down with it."""
    (tmp_path / "broken-Q4_K_M.gguf").write_bytes(b"not a gguf at all")
    a = LlamaCppAdapter({"id": "llamacpp", "kind": "llamacpp",
                         "models_dirs": [str(tmp_path)]}, procman=FakeProcman())
    models = a.list_models()
    assert [m.name for m in models] == ["broken-Q4_K_M.gguf"]
    assert models[0].meta["n_layers"] is None
    assert models[0].quant == "Q4_K_M"  # filename still tells us this much


# ---------------------------------------------------------------------------
# the HTTP-facing handler
# ---------------------------------------------------------------------------


def test_handler_forwards_options_and_audits_them(adapter, model_file, tmp_path):
    log = tmp_path / "audit.jsonl"
    status, body = handle_provider_action(
        {"llamacpp": adapter}, log, "start",
        {"provider": "llamacpp", "action": "start", "model": model_file.name,
         "options": {"ctx": 8192, "n_cpu_moe": 24}})
    assert status == 200 and body["ok"] is True
    assert "--ctx-size" in adapter._pm.argv
    assert "--n-cpu-moe" in adapter._pm.argv
    assert '"n_cpu_moe": 24' in log.read_text(encoding="utf-8").replace(", ", ", ")


def test_handler_rejects_a_non_object_options_envelope(adapter, tmp_path):
    status, body = handle_provider_action(
        {"llamacpp": adapter}, tmp_path / "audit.jsonl", "start",
        {"provider": "llamacpp", "action": "start", "options": "ctx=1"})
    assert status == 400
    assert "must be an object" in body["error"]


def test_handler_surfaces_the_adapter_refusal_verbatim(adapter, model_file,
                                                       tmp_path):
    status, body = handle_provider_action(
        {"llamacpp": adapter}, tmp_path / "audit.jsonl", "start",
        {"provider": "llamacpp", "action": "start", "model": model_file.name,
         "options": {"bogus": 1}})
    assert status == 400
    assert "unknown launch option(s): bogus" in body["error"]


# ---------------------------------------------------------------------------
# GGUF header reader
# ---------------------------------------------------------------------------


def test_reader_skips_a_tokenizer_array_without_reading_it(tmp_path):
    """The whole point of the reader: a 200k-entry string array must not be
    materialized to learn the layer count."""
    p = tmp_path / "big-Q4_K_M.gguf"
    p.write_bytes(_gguf_bytes(block_count=41, expert_count=256,
                              context_length=262144, tokens=20000))
    meta = gguf.read_meta(p)
    assert meta == {"quant": "Q4_K_M", "n_layers": 41, "n_experts": 256,
                    "n_experts_used": 8, "ctx_max": 262144}


def test_reader_grows_its_prefix_when_the_keys_sit_past_the_first_read(
        tmp_path, monkeypatch):
    monkeypatch.setattr(gguf, "_PREFIX_START", 64)  # force at least one regrow
    p = tmp_path / "late-Q6_K.gguf"
    p.write_bytes(_gguf_bytes(block_count=12, expert_count=None,
                              context_length=4096, tokens=500))
    assert gguf.read_meta(p)["n_layers"] == 12


def test_reader_caches_per_file_identity(tmp_path, monkeypatch):
    p = tmp_path / "cached-Q4_K_M.gguf"
    p.write_bytes(_gguf_bytes(block_count=8, expert_count=None,
                              context_length=2048))
    assert gguf.read_meta(p)["n_layers"] == 8
    calls = []
    real_parse = gguf._parse
    monkeypatch.setattr(gguf, "_parse",
                        lambda *a: (calls.append(1), real_parse(*a))[1])
    gguf.read_meta(p)
    assert calls == [], "second read should be served from the cache"


def test_reader_is_quiet_about_files_it_cannot_understand(tmp_path):
    (tmp_path / "junk.gguf").write_bytes(b"GGUF" + b"\x01\x00\x00\x00" + b"\x00" * 8)
    assert gguf.read_meta(tmp_path / "junk.gguf") == {}      # v1 refused
    assert gguf.read_meta(tmp_path / "nope.gguf") == {}      # missing


@pytest.mark.parametrize("name, quant", [
    ("Qwen3-30B-A3B-Q4_K_M.gguf", "Q4_K_M"),
    ("model-UD-IQ2_XXS.gguf", "IQ2_XXS"),
    ("thing.BF16.gguf", "BF16"),
    ("no-quant-here.gguf", None),
])
def test_quant_is_read_off_the_filename(name, quant):
    assert gguf.quant_from_name(name) == quant


# ---------------------------------------------------------------------------
# a minimal GGUF v3 writer, so the tests own their fixtures
# ---------------------------------------------------------------------------


def _kv_string(key: str) -> bytes:
    b = key.encode()
    return struct.pack("<Q", len(b)) + b


def _kv_u32(key: str, value: int) -> bytes:
    return _kv_string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_str_array(key: str, count: int) -> bytes:
    """A string array like tokenizer.ggml.tokens — the thing the reader must
    walk past without decoding."""
    out = [_kv_string(key), struct.pack("<I", 9), struct.pack("<I", 8),
           struct.pack("<Q", count)]
    for i in range(count):
        tok = f"tok{i}".encode()
        out.append(struct.pack("<Q", len(tok)) + tok)
    return b"".join(out)


def _gguf_bytes(*, block_count, expert_count, context_length, tokens=0) -> bytes:
    kvs = [
        _kv_string("general.architecture") + struct.pack("<I", 8)
        + struct.pack("<Q", 5) + b"qwen3",
        _kv_u32("qwen3.block_count", block_count),
        _kv_u32("qwen3.context_length", context_length),
    ]
    if expert_count is not None:
        kvs.append(_kv_u32("qwen3.expert_count", expert_count))
        kvs.append(_kv_u32("qwen3.expert_used_count", 8))
    if tokens:
        kvs.append(_kv_str_array("tokenizer.ggml.tokens", tokens))
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) \
        + struct.pack("<Q", len(kvs))
    return header + b"".join(kvs)
