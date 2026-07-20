"""Kokoro adapter — Kokoro-82M text-to-speech (TTS) via kokoro-onnx (CPU).

The one job: turn a sentence of the assistant's reply into speech, per-sentence,
so the voice tab can start talking as soon as the first sentence streams in rather
than waiting for the whole answer. This replaces the earlier Chatterbox adapter.

WHY KOKORO OVER CHATTERBOX (the reason this swap happened):
Chatterbox is a torch/CUDA pipeline whose real cold footprint on the 5070 was
~7 GB (torch context + T3 backbone + S3Gen + voice encoder + PerthNet). That is
FAR above its 0.5B-param weights and, on a 12 GB card, it could not sit next to
the resident 30B chat brain — the LLM had to be evicted first. Kokoro-82M runs as
a plain ONNX model via ``kokoro-onnx`` on the CPU (onnxruntime, no torch, no CUDA),
so it NEVER competes for VRAM. That mirrors the Parakeet STT adapter exactly:
both are small sherpa-/ONNX CPU models that keep voice off the GPU entirely.

    Install (already done on this box):
      python -m pip install --user kokoro-onnx     # pulls onnxruntime (CPU)
      # + download the model + voices files into models_dir:
      #   kokoro-v1.0.onnx   (~310 MB)
      #   voices-v1.0.bin    (~28 MB)
      # from the kokoro-onnx GitHub release assets (model-files-v1.0).

The package exposes ``Kokoro(onnx_path, voices_path)`` and
``.create(text, voice=..., speed=..., lang=...)`` returning a float32 waveform in
[-1, 1] and a sample rate (24 kHz). The seam below turns that into a 16-bit mono
WAV byte buffer via the stdlib ``wave`` module (no soundfile/torchaudio encoder in
the hot path) and base64-encodes it, which is what the browser's <audio> needs.

Output format: WAV (PCM) — every browser decodes it and it avoids an encoder
dependency. The response ``format`` field tells the frontend which it got, so a
later swap to opus over the Mac->PC tunnel is a one-seam change.

Residency: unlike Chatterbox this model lives in CPU RAM, not VRAM, so keeping it
warm is cheap and never starves the LLM. ``has_vram`` is False and there is no
detached model server — the model loads lazily in-process on first synthesize().
"""

from __future__ import annotations

import base64
import io
import wave

import numpy as np
from pathlib import Path

from memsom.providers.base import Capabilities, ProviderError
from memsom.providers.voice_base import VoiceAdapter

# Cap per-call synthesis text. TTS is called per-sentence, so this is a sanity
# fence against a caller shoving a whole document in, not a real limit.
_MAX_TTS_CHARS = 2000

# Kokoro emits 24 kHz mono float32. Used only as a fallback if .create() ever
# stops returning the sample rate alongside the samples.
_KOKORO_SR = 24000


class KokoroAdapter(VoiceAdapter):
    kind = "kokoro"
    default_model = "kokoro-v1.0"
    param_count = "0.082B"
    runtime_module = "kokoro_onnx"
    install_hint = ("python -m pip install --user kokoro-onnx + download "
                    "kokoro-v1.0.onnx and voices-v1.0.bin (kokoro-onnx release "
                    "assets model-files-v1.0) into models_dir")

    def __init__(self, spec: dict, procman=None) -> None:
        super().__init__(spec, procman=procman)
        if self.label in ("", self.id):
            self.label = spec.get("label") or "Kokoro (TTS)"
        self._kokoro = None  # cached kokoro_onnx.Kokoro
        # Voice + speed are profile-tunable; af_heart is a solid neutral default.
        self._voice = spec.get("voice") or "af_heart"
        try:
            self._speed = float(spec.get("speed", 1.0))
        except (TypeError, ValueError):
            self._speed = 1.0
        self._lang = spec.get("lang") or "en-us"

    # ---- capability: CPU model, no VRAM, no server process ----

    def capabilities(self) -> Capabilities:
        # Kokoro is a CPU ONNX model. It holds NO VRAM (that is the whole point of
        # the Chatterbox->Kokoro swap), so the panel never needs to weigh it
        # against the resident 30B. can_load stays True so the UI can pre-warm the
        # in-process weights; has_vram/can_estimate are False.
        return Capabilities(can_start=False, can_load=True, has_vram=False,
                            can_estimate=False, transports=("native",))

    # ---- model-file discovery ----

    def _model_files(self) -> dict:
        """Locate the Kokoro ONNX model + voices .bin in models_dir. Prefers the
        v1.0 asset names but accepts v0_19 / generic names too. Raises
        ProviderError with a fixable message if either piece is missing."""
        base = Path(self.models_dir or "")
        if not base.is_dir():
            raise ProviderError(
                f"kokoro models_dir not found: {base} — {self.install_hint}")

        def _pick(preferred: tuple, glob: str, what: str) -> Path:
            for name in preferred:
                p = base / name
                if p.is_file():
                    return p
            hits = sorted(base.glob(glob))
            if hits:
                return hits[0]
            raise ProviderError(
                f"kokoro: no {what} in {base} — {self.install_hint}")

        onnx = _pick(("kokoro-v1.0.onnx", "kokoro-v0_19.onnx"),
                     "kokoro*.onnx", "kokoro*.onnx model")
        voices = _pick(("voices-v1.0.bin", "voices-v0_19.bin", "voices.bin"),
                       "voices*.bin", "voices*.bin file")
        return {"onnx": str(onnx), "voices": str(voices)}

    # ---- residency: build/drop the model in-process (CPU RAM) ----

    def installed(self) -> tuple:
        """Runtime module import AND the model files present on disk — both are
        required before this adapter can actually synthesize."""
        inst, why = super().installed()
        if not inst:
            return inst, why
        try:
            self._model_files()
        except ProviderError as exc:
            return False, str(exc)
        return True, "kokoro_onnx + model files present"

    def _ensure_kokoro(self):
        if self._kokoro is not None:
            return self._kokoro
        from kokoro_onnx import Kokoro
        f = self._model_files()
        # onnxruntime CPUExecutionProvider — deliberately off the GPU so the 30B
        # chat brain keeps the whole 12 GB card.
        self._kokoro = Kokoro(f["onnx"], f["voices"])
        return self._kokoro

    def _warm(self, model: str) -> None:
        self._ensure_kokoro()

    def _cool(self) -> None:
        # CPU RAM only — just drop the reference and let the GC reclaim it.
        self._kokoro = None

    def synthesize(self, text: str, voice: str = None,
                   speed: float = None) -> dict:
        """Text -> {audio_b64, format, model_installed, ...}.

        Lazily builds+caches the Kokoro ONNX model in CPU RAM, generates a 24 kHz
        float32 waveform, encodes it to 16-bit mono WAV, and base64s it for the
        browser's <audio>."""
        model = self.default_model
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("tts requires non-empty 'text'")
        if len(text) > _MAX_TTS_CHARS:
            raise ProviderError(
                f"tts text too long ({len(text)} > {_MAX_TTS_CHARS} chars); "
                f"synthesize per-sentence")

        inst, why = self._stub_or_raise()
        if not inst:
            return {
                "audio_b64": "",
                "format": "wav",
                "model": model,
                "model_installed": False,
                "detail": f"kokoro TTS not installed: {why}",
                "install": self.install_hint,
                "chars": len(text),
            }

        k = self._ensure_kokoro()
        self._resident = model  # a bare synthesize() call counts as resident
        v = voice or self._voice
        sp = self._speed if speed is None else float(speed)
        samples, sr = k.create(text, voice=v, speed=sp, lang=self._lang)
        sr = int(sr or _KOKORO_SR)
        audio_bytes = _wav_bytes(samples, sr)
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        return {"audio_b64": b64, "format": "wav", "model": model,
                "model_installed": True, "sample_rate": sr,
                "voice": v, "speed": sp}


def _wav_bytes(samples, sample_rate: int) -> bytes:
    """A Kokoro waveform (numpy float32 in [-1, 1], shape (N,) or (1, N)) ->
    a little-endian 16-bit mono WAV byte buffer. Uses the stdlib wave module so
    there is no torchaudio/soundfile encoder dependency in this hot path."""
    arr = np.asarray(samples, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()
