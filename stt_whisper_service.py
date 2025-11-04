# stt_whisper_service.py

import asyncio
import threading
import time
import json
import os
from collections import deque
from typing import Optional, Deque

import numpy as np
from faster_whisper import WhisperModel


class STTService:
    """
    Offline STT using faster-whisper (CPU).
    - Input: 8 kHz, 16-bit, mono PCM frames from telephony layer.
    - Internal: upsample to 16 kHz (×2) before Whisper.
    - Emits:
        (False, '...') as soon as voice activity is detected (for barge-in)
        (True,  'final text') when an utterance ends (silence gap).
    """

    def __init__(
        self,
        model_path: str,
        language: Optional[str] = None,   # None=auto, "en" for English-only model
        sample_rate: int = 8000,
        compute_type: str = "int8",
    ):
        self.io_sample_rate = int(sample_rate)   # 8000
        self.model_sr = 16000                    # Whisper expects 16k
        self.language = language
        self.compute_type = compute_type

        self.transcript_queue: asyncio.Queue = asyncio.Queue()
        self._loop = asyncio.get_event_loop()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Whisper model directory not found: {model_path}")

        print(f"Loading faster-whisper from: {model_path} (cuda, compute_type={compute_type})")
        self.model = WhisperModel(
            model_size_or_path=model_path,
            device="cuda",
            compute_type=self.compute_type
        )

        # Buffers + state
        self._audio_buffer: Deque[bytes] = deque()        # incoming 8k frames
        self._utt_buffer = bytearray()                    # current utterance (8k bytes)
        self._recognizer_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_audio_time = 0.0

        # VAD-ish parameters (very lightweight energy gate)
        self._silence_threshold = 0.01         # tweak if needed (0..1, RMS normalized)
        self._silence_duration_s = 1.0         # finalize utterance after this much silence
        self._voice_active = False
        self._last_voice_time = 0.0
        self._emitted_bargein_for_this_utt = False  # ensure we emit partial once per utterance

        print(f"Whisper STT initialized. I/O SR={self.io_sample_rate} Hz -> Model SR={self.model_sr} Hz, Lang={self.language or 'auto'}")

    async def start(self):
        self._running = True
        self._recognizer_thread = threading.Thread(target=self._worker, daemon=True)
        self._recognizer_thread.start()
        await asyncio.sleep(0)  # yield

    def write_audio(self, pcm_chunk: bytes):
        if not self._running:
            return
        self._audio_buffer.append(pcm_chunk)
        self._last_audio_time = self._loop.time()

    async def stop(self):
        self._running = False
        if self._recognizer_thread:
            self._recognizer_thread.join(timeout=2)
        try:
            await self.transcript_queue.put((True, None))
        except Exception:
            pass
        print("Whisper STT Service stopped.")

    # ---------------- internals ----------------

    def _worker(self):
        print("Whisper recognition worker started.")
        while self._running:
            try:
                if not self._audio_buffer:
                    time.sleep(0.01)
                    # If idle long enough, also consider flushing a tail (rare)
                    self._maybe_finalize_on_idle()
                    continue

                chunk8 = self._audio_buffer.popleft()
                if not chunk8:
                    continue

                # Energy check at 8k to detect speech/silence
                if not self._is_silent(chunk8):
                    # Speech detected
                    self._voice_active = True
                    self._last_voice_time = self._loop.time()
                    self._utt_buffer.extend(chunk8)

                    # Emit a one-time partial to trigger barge-in upstream
                    if not self._emitted_bargein_for_this_utt:
                        self._emitted_bargein_for_this_utt = True
                        asyncio.run_coroutine_threadsafe(
                            self.transcript_queue.put((False, "...")),
                            self._loop
                        )
                else:
                    # Silence frame
                    if self._voice_active:
                        # Still in utterance; append but check for end after threshold
                        self._utt_buffer.extend(chunk8)

                # If we have been silent for a while after last voice, finalize utterance
                if self._voice_active and (self._loop.time() - self._last_voice_time) >= self._silence_duration_s:
                    self._finalize_current_utterance()
                    self._voice_active = False
                    self._emitted_bargein_for_this_utt = False

            except Exception as e:
                print(f"Error in Whisper recognition worker: {e}")
                asyncio.run_coroutine_threadsafe(
                    self.transcript_queue.put((True, f"ERROR: {e}")), self._loop
                )
                break

        # Flush anything pending on stop
        if self._utt_buffer:
            try:
                self._finalize_current_utterance()
            except Exception:
                pass
        print("Whisper recognition worker stopped.")

    def _maybe_finalize_on_idle(self):
        # If no audio for a while but an utterance buffer exists, finalize it
        if self._voice_active and (self._loop.time() - self._last_voice_time) >= self._silence_duration_s:
            self._finalize_current_utterance()
            self._voice_active = False
            self._emitted_bargein_for_this_utt = False

    def _finalize_current_utterance(self):
        """Run Whisper on the buffered utterance (8k -> upsample to 16k) and emit final text."""
        if not self._utt_buffer:
            return
        # Convert bytes -> int16
        s8 = np.frombuffer(bytes(self._utt_buffer), dtype=np.int16)
        self._utt_buffer.clear()

        if s8.size == 0:
            return

        # 8k -> 16k upsample (×2 repeat; cheap and OK for phone-band audio)
        s16 = np.repeat(s8, 2).astype(np.int16)

        # int16 -> float32 in [-1, 1]
        audio_f32 = (s16.astype(np.float32) / 32768.0)

        # Transcribe with faster-whisper (blocking in this worker thread)
        segments, _info = self.model.transcribe(
            audio_f32,
            language=self.language,                # None=auto; or "en"
            beam_size=1, best_of=1,                # fast CPU settings
            vad_filter=False,
            condition_on_previous_text=False,      # keep utterances independent
            no_speech_threshold=0.6,               # optional; helps skip non-speech
        )
        text = "".join(seg.text for seg in segments).strip()
        if text:
            asyncio.run_coroutine_threadsafe(
                self.transcript_queue.put((True, text)),
                self._loop
            )
            print(f"Whisper Final: {text}")

    def _is_silent(self, pcm_chunk_8k: bytes) -> bool:
        """Simple RMS gate on 8k chunk."""
        if not pcm_chunk_8k:
            return True
        s = np.frombuffer(pcm_chunk_8k, dtype=np.int16)
        if s.size == 0:
            return True
        rms = np.sqrt(np.mean(s.astype(np.float32) ** 2))
        normalized = rms / 32768.0
        return normalized < self._silence_threshold
