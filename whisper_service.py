import numpy as np
import time
from typing import Optional, Tuple

from faster_whisper import WhisperModel


class WhisperSTT:
    """
    Simple wrapper around faster-whisper for single-utterance transcription.
    Feed a mono Float32/Int16 numpy array at 16 kHz (preferred) or any rate;
    resample externally if needed for best results.
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,
    ) -> None:
        """Initialize faster-whisper model, preferring GPU when available.

        Falls back to CPU/int8 if the requested device/compute_type is unavailable.
        """
        try:
            self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
            print(f"[STT] Loaded Whisper on {device} ({compute_type})")
        except Exception as e:
            print(f"[STT] Failed to load Whisper on {device}/{compute_type}: {e}. Falling back to CPU/int8")
            self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        self.language = language

    def transcribe(self, audio_f32: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """
        Transcribe a single utterance and return (text, avg_prob).
        audio_f32: mono float32 waveform in range [-1, 1].
        sample_rate: the sampling rate of audio_f32.
        """
        start_time = time.time()
        print(f"[STT] Starting transcription of {len(audio_f32)/sample_rate:.2f}s audio...")

        segments, info = self.model.transcribe(
            audio_f32,
            language=self.language,
            task="transcribe",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            beam_size=5,
            best_of=5,
            temperature=0.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            initial_prompt=None,
            without_timestamps=True,
        )
        text_parts = []
        probs = []
        for seg in segments:
            if seg.no_speech_prob is not None and seg.no_speech_prob > 0.8:
                continue
            text_parts.append(seg.text)
            if seg.avg_logprob is not None:
                probs.append(1.0)  # placeholder; faster-whisper doesn't return direct prob
        text = " ".join([t.strip() for t in text_parts if t and t.strip()])
        avg_prob = float(np.mean(probs)) if probs else 0.0

        elapsed = time.time() - start_time
        print(f"[STT] Transcription completed in {elapsed:.2f}s")

        return text.strip(), avg_prob
