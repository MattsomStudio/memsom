"""Parakeet adapter — NVIDIA parakeet-tdt-0.6b-v2 speech-to-text (STT).

The one job: turn a short chunk of compressed microphone audio (opus/webm from
the browser's MediaRecorder) into text, fast, on the local GPU, so a voice tab
never has to ship audio to a cloud STT service. parakeet-tdt-0.6b-v2 is the
current sweet spot — ~0.6B params, near-realtime on a 5070, strong English WER.

RUNTIME CHOICE (documented, per guardrail — not installed here):
Cleanest path is **sherpa-onnx** running the ONNX export of parakeet-tdt-0.6b.
sherpa-onnx is a pip wheel with a CUDA build, no NeMo/PyTorch stack to drag in,
and it exposes an offline recognizer that takes a decoded waveform and returns
text — exactly this adapter's shape. NeMo works too but pulls the full PyTorch +
NeMo toolkit (multi-GB, source-ish on Windows), which is the heavier option and
why sherpa-onnx wins for a load-on-demand panel model.

    Install (deliberately NOT run — see the build report):
      pip install sherpa-onnx            # CUDA wheel: sherpa-onnx-cuda
      # + download the parakeet-tdt-0.6b-v2 ONNX model files into models_dir

Audio decode: the browser sends COMPRESSED audio (opus in a webm/ogg container).
sherpa-onnx wants PCM samples, so the real inference seam decodes the container
to a mono 16 kHz float array first (ffmpeg or soundfile+av). That decode step is
part of the stubbed section below.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

import numpy as np

from memsom.providers.base import ProviderError, run_no_window
from memsom.providers.voice_base import VoiceAdapter

# STT input contract: sherpa-onnx wants mono 16 kHz float32 PCM.
_STT_SR = 16000


class ParakeetAdapter(VoiceAdapter):
    kind = "parakeet"
    default_model = "parakeet-tdt-0.6b-v2"
    param_count = "0.6B"
    runtime_module = "sherpa_onnx"
    install_hint = ("pip install --user sherpa-onnx + download the "
                    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2 ONNX files "
                    "(encoder/decoder/joiner + tokens.txt) into models_dir")

    def __init__(self, spec: dict, procman=None) -> None:
        super().__init__(spec, procman=procman)
        if self.label in ("", self.id):
            self.label = spec.get("label") or "Parakeet (STT)"
        self._recognizer = None  # cached sherpa_onnx.OfflineRecognizer

    # ---- model-file discovery ----

    def _model_files(self) -> dict:
        """Locate the three transducer ONNX files + tokens in models_dir.
        Prefers an int8 export (what we ship) but accepts fp16/fp32 names too.
        Raises ProviderError with a fixable message if any piece is missing."""
        base = Path(self.models_dir or "")
        if not base.is_dir():
            raise ProviderError(
                f"parakeet models_dir not found: {base} — {self.install_hint}")

        def _pick(stem: str) -> Path:
            # e.g. encoder.int8.onnx > encoder.fp16.onnx > encoder.onnx
            for suffix in (".int8.onnx", ".fp16.onnx", ".onnx"):
                hits = sorted(base.glob(f"{stem}*{suffix}"))
                if hits:
                    return hits[0]
            raise ProviderError(
                f"parakeet: no {stem}*.onnx in {base} — {self.install_hint}")

        tokens = base / "tokens.txt"
        if not tokens.is_file():
            raise ProviderError(
                f"parakeet: tokens.txt missing in {base} — {self.install_hint}")
        return {
            "encoder": str(_pick("encoder")),
            "decoder": str(_pick("decoder")),
            "joiner": str(_pick("joiner")),
            "tokens": str(tokens),
        }

    # ---- residency: build/drop the recognizer in-process ----

    def installed(self) -> tuple:
        """Runtime module import AND the model files present on disk — both are
        required before this adapter can actually transcribe."""
        inst, why = super().installed()
        if not inst:
            return inst, why
        try:
            self._model_files()
        except ProviderError as exc:
            return False, str(exc)
        return True, "sherpa_onnx + model files present"

    def _ensure_recognizer(self):
        if self._recognizer is not None:
            return self._recognizer
        import sherpa_onnx
        f = self._model_files()
        # parakeet-tdt is a NeMo Token-and-Duration Transducer. sherpa-onnx runs
        # it on CPU here (the PyPI wheel is CPU onnxruntime) — deliberate: it
        # keeps STT off the ~3.5GB-free card entirely, so only the TTS model
        # ever competes for VRAM. Fast enough (<0.5s for a short clip).
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f["encoder"], decoder=f["decoder"], joiner=f["joiner"],
            tokens=f["tokens"], num_threads=4, sample_rate=_STT_SR,
            feature_dim=80, decoding_method="greedy_search",
            model_type="nemo_transducer")
        return self._recognizer

    def _warm(self, model: str) -> None:
        self._ensure_recognizer()

    def _cool(self) -> None:
        self._recognizer = None

    def transcribe(self, audio_bytes: bytes, fmt: str) -> dict:
        """Compressed audio bytes -> {text, model_installed, ...}.

        Decodes the browser's opus/webm (or any ffmpeg-readable container) to
        mono 16 kHz float32 PCM, then runs a cached sherpa-onnx recognizer."""
        model = self.default_model
        inst, why = self._stub_or_raise()
        if not inst:
            return {
                "text": "",
                "model": model,
                "model_installed": False,
                "detail": f"parakeet STT not installed: {why}",
                "install": self.install_hint,
                "audio_bytes": len(audio_bytes or b""),
                "format": fmt,
            }

        samples = _decode_to_pcm16k(audio_bytes)
        rec = self._ensure_recognizer()
        self._resident = model  # a bare transcribe() call counts as resident
        stream = rec.create_stream()
        stream.accept_waveform(_STT_SR, samples)
        rec.decode_stream(stream)
        return {"text": stream.result.text.strip(), "model": model,
                "model_installed": True}


def _decode_to_pcm16k(audio_bytes: bytes) -> "np.ndarray":
    """Decode any ffmpeg-readable audio (opus-in-webm from MediaRecorder, wav,
    ogg, ...) to a mono 16 kHz float32 numpy array. ffmpeg autodetects the
    container from the stream, so no per-format branch is needed."""
    if not audio_bytes:
        raise ProviderError("no audio bytes to transcribe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ProviderError("ffmpeg not on PATH — needed to decode STT audio")
    proc = run_no_window(
        [ffmpeg, "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(_STT_SR),
         "pipe:1"],
        input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        raise ProviderError(f"ffmpeg failed to decode STT audio: {err}")
    pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    if pcm.size == 0:
        raise ProviderError("decoded audio is empty (0 samples)")
    return pcm


def decode_audio_b64(audio_b64: str) -> bytes:
    """Strict base64 -> bytes. Rejects a data: URL prefix or malformed input
    with a ProviderError the handler turns into a clean 400 (never a 500)."""
    if not isinstance(audio_b64, str) or not audio_b64:
        raise ProviderError("audio_b64 must be a non-empty base64 string")
    s = audio_b64
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1]  # tolerate a data:audio/...;base64, prefix
    try:
        return base64.b64decode(s, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ProviderError(f"audio_b64 is not valid base64: {exc}") from exc
