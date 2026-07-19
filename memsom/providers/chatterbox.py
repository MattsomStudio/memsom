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

from memsom.providers.base import ProviderError
from memsom.providers.voice_base import VoiceAdapter

# Cap per-call synthesis text. TTS is called per-sentence, so this is a sanity
# fence against a caller shoving a whole document in, not a real limit.
_MAX_TTS_CHARS = 2000


class ChatterboxAdapter(VoiceAdapter):
    kind = "chatterbox"
    default_model = "chatterbox-tts"
    param_count = "0.5B"
    runtime_module = "chatterbox"
    install_hint = "pip install chatterbox-tts (first run downloads ~0.5B weights)"

    def __init__(self, spec: dict, procman=None) -> None:
        super().__init__(spec, procman=procman)
        if self.label in ("", self.id):
            self.label = spec.get("label") or "Chatterbox (TTS)"

    def synthesize(self, text: str) -> dict:
        """Text -> {audio_b64, format, model_installed, ...}.

        Returns a clean stub payload while the runtime is absent (endpoint stays
        live and contract-shaped). When chatterbox-tts is installed, the real
        generate+encode replaces the raise below."""
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

        # === REAL INFERENCE SEAM (wire once chatterbox-tts is installed) ===
        #   1. lazily build+cache ChatterboxTTS.from_pretrained(device="cuda")
        #   2. wav = tts.generate(text)  -> audio tensor + sample_rate
        #   3. encode tensor to a WAV byte buffer, base64-encode it
        #   4. return {"audio_b64": b64, "format": "wav", "model": model,
        #              "model_installed": True}
        raise ProviderError(
            "chatterbox runtime present but inference not yet wired — "
            "implement the generate+encode seam in ChatterboxAdapter.synthesize")
