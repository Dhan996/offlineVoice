"""
Hybrid STT Service combining best of both approaches:
- VAD-based utterance segmentation (from stt_whisper_service)
- Barge-in support with immediate voice detection
- 16kHz float32 numpy array input (from streaming_whisper_service)
- medium model for better accuracy
- Async threading architecture for non-blocking operation
"""

import asyncio
import threading
import time
from collections import deque
from typing import Optional, Deque, Dict

import numpy as np
from faster_whisper import WhisperModel


class StreamingWhisperService1:
    """
    Real-time STT using faster-whisper with VAD-based segmentation

    Features:
    - Input: 16 kHz float32 numpy arrays (standard audio format)
    - VAD-based utterance detection (energy + silence threshold)
    - Immediate barge-in signals when voice detected
    - Async queue for results: (is_final: bool, text: str, metadata: dict)
    - Uses Whisper medium for high accuracy
    """

    def __init__(
        self,
        model_name: str = "medium",      # Model name or path
        language: Optional[str] = None,    # None=auto, "en" for English
        sample_rate: int = 16000,
        device: str = "cuda",
        compute_type: str = "float16",     # "float16" for GPU, "int8" for CPU
        silence_threshold: float = 0.02,   # RMS threshold (0..1) - optimized for quality
        silence_duration_s: float = 0.8,   # Finalize after this much silence (faster response)
        voice_frames_needed: int = 2,      # Consecutive frames before barge-in (200ms)
    ):
        """
        Args:
            model_name: Whisper model name ("medium", "medium", "small") or path to local model
            language: Force language (None for auto-detect, "en" for English)
            sample_rate: Expected audio sample rate (16000 recommended)
            device: "cuda" or "cpu"
            compute_type: "float16" (GPU), "int8" (CPU)
            silence_threshold: Energy threshold for voice detection (0.03 = 3% of max, higher = less sensitive)
            silence_duration_s: Seconds of silence before finalizing utterance
            voice_frames_needed: Consecutive voice frames before triggering barge-in (reduces false positives)
        """
        self.sample_rate = int(sample_rate)
        self.language = language
        self.device = device
        self.compute_type = compute_type

        # Async queue for results: (is_final, text, metadata)
        self.transcript_queue: asyncio.Queue = asyncio.Queue()
        self._loop = asyncio.get_event_loop()

        print(f"[StreamingWhisper1] Loading Whisper model: {model_name}")
        print(f"[StreamingWhisper1] Device: {device}, Compute: {compute_type}")

        try:
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type
            )
            print(f"[StreamingWhisper1] Model loaded successfully on {device}")
        except Exception as e:
            print(f"[StreamingWhisper1] Failed to load on {device}/{compute_type}: {e}")
            print(f"[StreamingWhisper1] Falling back to CPU/int8...")
            self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
            self.device = "cpu"
            self.compute_type = "int8"

        # Buffers + state
        self._audio_buffer: Deque[np.ndarray] = deque()  # Incoming float32 chunks
        self._utt_buffer = []                             # Current utterance (list of numpy arrays)
        self._recognizer_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_audio_time = 0.0

        # VAD parameters
        self._silence_threshold = silence_threshold
        self._silence_duration_s = silence_duration_s
        self._voice_active = False
        self._last_voice_time = 0.0
        self._emitted_bargein_for_this_utt = False

        # Barge-in protection: require N consecutive voice frames before triggering
        self._voice_frames_needed = voice_frames_needed
        self._consecutive_voice_frames = 0

        print(f"[StreamingWhisper1] Initialized. SR={self.sample_rate} Hz, Lang={self.language or 'auto'}")
        print(f"[StreamingWhisper1] VAD: threshold={silence_threshold}, silence={silence_duration_s}s, frames_needed={voice_frames_needed}")
        print(f"[StreamingWhisper1] Quality: beam_size=5, best_of=5, vad_filter=enabled")

    async def start(self):
        """Start the recognition worker thread"""
        self._running = True
        self._recognizer_thread = threading.Thread(target=self._worker, daemon=True)
        self._recognizer_thread.start()
        await asyncio.sleep(0)  # Yield to event loop
        # Worker started - reduced verbosity

    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Add audio chunk to processing buffer

        Args:
            audio_chunk: numpy array of float32 audio in [-1, 1] at 16kHz
        """
        if not self._running:
            return

        # Ensure float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Ensure 1D
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()

        self._audio_buffer.append(audio_chunk)
        self._last_audio_time = self._loop.time()

    async def stop(self):
        """Stop the recognition worker and finalize any pending audio"""
        self._running = False
        if self._recognizer_thread:
            self._recognizer_thread.join(timeout=2)
        try:
            # Signal end of stream
            await self.transcript_queue.put((True, None, {'status': 'stopped'}))
        except Exception:
            pass
        print("[StreamingWhisper1] Service stopped")

    # ---------------- Internals ----------------

    def _worker(self):
        """Main recognition worker thread"""
        # Worker started - reduced verbosity

        while self._running:
            try:
                if not self._audio_buffer:
                    time.sleep(0.01)  # 10ms sleep when idle
                    self._maybe_finalize_on_idle()
                    continue

                # Get next chunk
                chunk = self._audio_buffer.popleft()
                if chunk is None or chunk.size == 0:
                    continue

                # VAD: Check if this chunk contains voice
                is_silent = self._is_silent(chunk)

                # Debug: Log first few voice detections
                if not is_silent and not hasattr(self, '_debug_voice_count'):
                    self._debug_voice_count = 0
                if not is_silent:
                    self._debug_voice_count = getattr(self, '_debug_voice_count', 0) + 1
                    if self._debug_voice_count <= 5:
                        rms = np.sqrt(np.mean(chunk ** 2))
                        print(f"[StreamingWhisper1] Voice detected! RMS={rms:.4f}, consecutive={self._consecutive_voice_frames + 1}")

                if not is_silent:
                    # Potential voice detected
                    self._consecutive_voice_frames += 1
                    self._last_voice_time = self._loop.time()
                    self._utt_buffer.append(chunk)

                    # Only activate voice after N consecutive frames (reduces false positives)
                    if not self._voice_active and self._consecutive_voice_frames >= self._voice_frames_needed:
                        self._voice_active = True

                        # Emit one-time barge-in signal
                        if not self._emitted_bargein_for_this_utt:
                            self._emitted_bargein_for_this_utt = True
                            asyncio.run_coroutine_threadsafe(
                                self.transcript_queue.put((
                                    False,  # is_partial
                                    "...",  # placeholder text
                                    {
                                        'type': 'barge_in',
                                        'timestamp': time.time(),
                                        'voice_detected': True
                                    }
                                )),
                                self._loop
                            )
                else:
                    # Silence detected - reset consecutive counter
                    self._consecutive_voice_frames = 0

                    if self._voice_active:
                        # Still in utterance; append silence for context
                        self._utt_buffer.append(chunk)

                # Check if we should finalize the utterance
                if self._voice_active and (self._loop.time() - self._last_voice_time) >= self._silence_duration_s:
                    self._finalize_current_utterance()
                    self._voice_active = False
                    self._emitted_bargein_for_this_utt = False
                    self._consecutive_voice_frames = 0

            except Exception as e:
                print(f"[StreamingWhisper1] Error in worker: {e}")
                import traceback
                traceback.print_exc()
                asyncio.run_coroutine_threadsafe(
                    self.transcript_queue.put((
                        True,
                        f"ERROR: {e}",
                        {'type': 'error', 'error': str(e)}
                    )),
                    self._loop
                )
                break

        # Flush any pending utterance on shutdown
        if self._utt_buffer:
            try:
                self._finalize_current_utterance()
            except Exception as e:
                print(f"[StreamingWhisper1] Error finalizing on shutdown: {e}")

        # Worker stopped - reduced verbosity

    def _maybe_finalize_on_idle(self):
        """Finalize utterance if idle for too long"""
        if self._voice_active and (self._loop.time() - self._last_voice_time) >= self._silence_duration_s:
            self._finalize_current_utterance()
            self._voice_active = False
            self._emitted_bargein_for_this_utt = False
            self._consecutive_voice_frames = 0

    def _finalize_current_utterance(self):
        """Run Whisper on buffered utterance and emit final transcript"""
        print("[StreamingWhisper1] DEBUG: Finalizing utterance...") # Added for debugging
        if not self._utt_buffer:
            return

        start_time = time.time()

        # Concatenate all chunks into single array
        audio_float32 = np.concatenate(self._utt_buffer)
        self._utt_buffer.clear()

        if audio_float32.size == 0:
            return

        # Clip to [-1, 1] range (should already be, but safety check)
        audio_float32 = np.clip(audio_float32, -1.0, 1.0)

        # Transcribe with faster-whisper (optimized for quality)
        try:
            segments, info = self.model.transcribe(
                audio_float32,
                language=self.language,

                # Quality settings (medium optimized)
                beam_size=5,                           # Best balance quality/speed for medium
                best_of=5,                             # Try 5 candidates, pick best
                temperature=0.0,                       # Deterministic output (no randomness)

                # VAD settings
                vad_filter=True,                       # Filter out silence/noise
                vad_parameters=dict(
                    min_silence_duration_ms=300,       # 300ms silence = faster segmentation
                    threshold=0.5,                     # VAD sensitivity
                    speech_pad_ms=400                  # Pad speech with 400ms context
                ),

                # Accuracy settings
                condition_on_previous_text=False,      # Independent utterances (no drift)
                compression_ratio_threshold=2.4,       # Detect repetition/hallucination
                log_prob_threshold=-1.0,               # Accept confident predictions
                no_speech_threshold=0.6,               # Skip non-speech segments

                # Timestamp settings
                word_timestamps=False,                 # Disable for speed
                # Merge-friendly punctuation (multi-language)
                prepend_punctuations="\"'“¿([{-",
                append_punctuations="\"'.。,，!！?？:：”’)}、"
            )

            # Collect transcript
            text = "".join(seg.text for seg in segments).strip()

            processing_time = (time.time() - start_time) * 1000  # ms

            if text:
                # Emit final transcript
                asyncio.run_coroutine_threadsafe(
                    self.transcript_queue.put((
                        True,  # is_final
                        text,
                        {
                            'type': 'final',
                            'processing_time_ms': processing_time,
                            'audio_duration_s': len(audio_float32) / self.sample_rate,
                            'language': info.language if hasattr(info, 'language') else self.language,
                            'language_probability': info.language_probability if hasattr(info, 'language_probability') else None
                        }
                    )),
                    self._loop
                )
                print(f"[StreamingWhisper1] Final ({processing_time:.0f}ms): {text}")
            # Removed "No speech detected" log - was cluttering output

        except Exception as e:
            print(f"[StreamingWhisper1] Transcription error: {e}")
            import traceback
            traceback.print_exc()

    def _is_silent(self, audio_chunk: np.ndarray) -> bool:
        """
        Simple energy-based VAD using RMS

        Args:
            audio_chunk: float32 numpy array in [-1, 1]

        Returns:
            True if silent, False if voice detected
        """
        if audio_chunk.size == 0:
            return True

        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_chunk ** 2))

        # Already normalized since input is in [-1, 1]
        return rms < self._silence_threshold

    def reset(self):
        """Reset buffers for new conversation"""
        self._audio_buffer.clear()
        self._utt_buffer.clear()
        self._voice_active = False
        self._emitted_bargein_for_this_utt = False
        self._last_voice_time = 0.0
        self._consecutive_voice_frames = 0
        # Removed verbose log - was cluttering output


# Example usage
if __name__ == "__main__":
    import soundfile as sf
    import os

    async def test_streaming():
        """Test the streaming service"""

        # Initialize service
        stt = StreamingWhisperService1(
            model_name="medium",                  # or "medium", "small", etc.
            device="cuda",                          # Use "cpu" if no GPU
            compute_type="float16",                 # Use "int8" for CPU
            language="en",
            silence_threshold=0.01,
            silence_duration_s=1.0
        )

        await stt.start()

        print("\n[Test] Loading test audio...")

        # Check if test audio file exists
        if not os.path.exists("test_audio.wav"):
            print("[Test] ERROR: test_audio.wav not found!")
            print("[Test] Please provide a test_audio.wav file in the current directory.")
            print("[Test] Skipping test...")
            await stt.stop()
            return

        # Load test audio file
        audio, sr = sf.read("test_audio.wav")

        # Resample if needed (simple example, use librosa for better quality)
        if sr != 16000:
            print(f"[Test] Warning: Audio is {sr}Hz, expected 16kHz")

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Simulate streaming by sending 100ms chunks
        chunk_size = int(0.1 * 16000)  # 100ms at 16kHz = 1600 samples

        print(f"[Test] Streaming {len(audio)/16000:.2f}s of audio in {chunk_size/16000*1000:.0f}ms chunks\n")

        # Create task to read results
        async def read_results():
            while True:
                is_final, text, metadata = await stt.transcript_queue.get()

                if text is None:
                    print("[Test] Stream ended")
                    break

                if metadata.get('type') == 'barge_in':
                    print(f"[BARGE-IN] Voice detected!")
                elif is_final:
                    proc_time = metadata.get('processing_time_ms', 0)
                    duration = metadata.get('audio_duration_s', 0)
                    lang = metadata.get('language', 'unknown')
                    print(f"[FINAL] ({proc_time:.0f}ms, {duration:.1f}s, {lang}): {text}")

        # Start result reader
        result_task = asyncio.create_task(read_results())

        # Stream audio chunks
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            stt.add_audio_chunk(chunk)
            await asyncio.sleep(0.1)  # Simulate real-time streaming

        # Wait a bit for final processing
        await asyncio.sleep(2)

        # Stop service
        await stt.stop()

        # Wait for result reader to finish
        await result_task

        print("\n[Test] Complete!")

    # Run test
    asyncio.run(test_streaming())
