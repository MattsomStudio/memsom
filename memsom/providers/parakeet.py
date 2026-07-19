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

from memsom.providers.base import ProviderError
from memsom.providers.voice_base import VoiceAdapter


class ParakeetAdapter(VoiceAdapter):
    kind = "parakeet"
    default_model = "parakeet-tdt-0.6b-v2"
    param_count = "0.6B"
    runtime_module = "sherpa_onnx"
    install_hint = ("pip install sherpa-onnx (CUDA wheel) + download "
                    "parakeet-tdt-0.6b-v2 ONNX files into models_dir")

    def __init__(self, spec: dict, procman=None) -> None:
        super().__init__(spec, procman=procman)
        if self.label in ("", self.id):
            self.label = spec.get("label") or "Parakeet (STT)"

    def transcribe(self, audio_bytes: bytes, fmt: str) -> dict:
        """Compressed audio bytes -> {text, model_installed, ...}.

        Returns a clean stub payload while the runtime is absent (the panel
        endpoint stays live and contract-shaped). When sherpa-onnx is installed,
        the real decode+recognize replaces the raise below."""
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

        # === REAL INFERENCE SEAM (wire once sherpa-onnx + model are installed) ===
        #   1. decode `audio_bytes` (opus/webm) -> mono 16kHz float32 PCM
        #   2. feed PCM to a cached sherpa_onnx.OfflineRecognizer for `model`
        #   3. return {"text": result.text, "model": model, "model_installed": True}
        raise ProviderError(
            "parakeet runtime present but inference not yet wired — "
            "implement the decode+recognize seam in ParakeetAdapter.transcribe")


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
