"""Chatterbox adapter — Resemble AI's Chatterbox text-to-speech (TTS).

The one job: turn a sentence of the assistant's reply into speech, per-sentence,
so the voice tab can start talking as soon as the first sentence streams in
rather than waiting for the whole answer. Chatterbox is ~0.5B params, open-weight,
runs on the local GPU, and has a clean pip package — a good fit for a load-on-
demand panel model that shares the card with the 30B chat brain.

RUNTIME CHOICE (documented, per guardrail — not installed here):
Chatterbox ships its own pip package that wraps model download + inference:

    Install (deliberately NOT run — see the build report):
      pip install chatterbox-tts
      # first use downloads the ~0.5B weights to the HF cache

The package exposes ``ChatterboxTTS.from_pretrained(device="cuda")`` and a
``.generate(text)`` returning an audio tensor + sample rate. The real seam below
turns that tensor into a WAV byte buffer and base64-encodes it, which is what the
browser's <audio> needs.

Output format: WAV (PCM) is the safe default — every browser decodes it and it
avoids an encoder dependency. If bandwidth over the Mac->PC tunnel matters later,
swap the encode seam to opus; the response ``format`` field already tells the
frontend which it got.
"""

from __future__ import annotations

import base64
import gc
import io
import wave

import numpy as np

from memsom.providers.base import ProviderError
from memsom.providers.voice_base import VoiceAdapter

# Cap per-call synthesis text. TTS is called per-sentence, so this is a sanity
# fence against a caller shoving a whole document in, not a real limit.
_MAX_TTS_CHARS = 2000

# Measured cold footprint of the Chatterbox stack on the 5070 (torch CUDA
# context + T3 backbone + S3Gen + voice encoder + PerthNet): ~7 GB. That is FAR
# above the naive 0.5B-param weights estimate, so the advisory floor below keeps
# the VRAM gate honest — Chatterbox will NOT fit next to the resident 30B on a
# 12 GB card; the LLM must be evicted first.
_VRAM_FLOOR_MB = 6500


class ChatterboxAdapter(VoiceAdapter):
    kind = "chatterbox"
    default_model = "chatterbox-tts"
    param_count = "0.5B"
    runtime_module = "chatterbox"
    install_hint = ("pip install --user chatterbox-tts --no-deps (protect an "
                    "existing CUDA torch) + hand-install its light deps; first "
                    "run downloads ~3 GB of weights to the HF cache")

    def __init__(self, spec: dict, procman=None) -> None:
        super().__init__(spec, procman=procman)
        if self.label in ("", self.id):
            self.label = spec.get("label") or "Chatterbox (TTS)"
        self._tts = None  # cached ChatterboxTTS
        self._device = spec.get("device") or "cuda"
        # Emotion control (Matt wants this available). Profile-tunable default;
        # 0.5 is Chatterbox's neutral. Threaded per-call via synthesize().
        try:
            self._exaggeration = float(spec.get("exaggeration", 0.5))
        except (TypeError, ValueError):
            self._exaggeration = 0.5

    # ---- VRAM estimate: floor at the measured real footprint ----

    def estimate_vram(self, model: str, ctx: int, kv_type: str = "fp16") -> dict:
        est = super().estimate_vram(model, ctx, kv_type)
        total = est.get("total_mb")
        if total is None or total < _VRAM_FLOOR_MB:
            est["total_mb"] = _VRAM_FLOOR_MB
            est["partial"] = True
            est["note"] = "measured cold footprint floor (~7GB incl torch ctx)"
        return est

    # ---- residency: build/drop the model in-process ----

    def _ensure_tts(self):
        if self._tts is not None:
            return self._tts
        from chatterbox.tts import ChatterboxTTS
        self._tts = ChatterboxTTS.from_pretrained(device=self._device)
        return self._tts

    def _warm(self, model: str) -> None:
        self._ensure_tts()

    def _cool(self) -> None:
        self._tts = None
        gc.collect()
        try:  # free the card so the 30B chat brain can have it back
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def synthesize(self, text: str, exaggeration: float = None) -> dict:
        """Text -> {audio_b64, format, model_installed, ...}.

        Lazily builds+caches ChatterboxTTS on the GPU, generates a waveform,
        encodes it to WAV, and base64s it for the browser's <audio>."""
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
                "detail": f"chatterbox TTS not installed: {why}",
                "install": self.install_hint,
                "chars": len(text),
            }

        tts = self._ensure_tts()
        self._resident = model  # a bare synthesize() call counts as resident
        exag = self._exaggeration if exaggeration is None else float(exaggeration)
        wav = tts.generate(text, exaggeration=exag)
        audio_bytes = _wav_bytes(wav, int(getattr(tts, "sr", 24000)))
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        return {"audio_b64": b64, "format": "wav", "model": model,
                "model_installed": True, "sample_rate": int(getattr(tts, "sr", 24000)),
                "exaggeration": exag}


def _wav_bytes(wav, sample_rate: int) -> bytes:
    """A Chatterbox audio tensor (shape (1, N) or (N,), float32 in [-1, 1]) ->
    a little-endian 16-bit mono WAV byte buffer. Uses the stdlib wave module so
    there is no torchaudio/soundfile encoder dependency in this hot path."""
    arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
    arr = np.squeeze(arr).astype(np.float32)
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
