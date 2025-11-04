"""
Streaming STT using faster-whisper with sliding window approach
Achieves 200-400ms processing time per window
"""
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Tuple, List
from collections import deque
import time


class StreamingWhisperSTT:
    """
    Real-time-ish speech-to-text using faster-whisper with sliding windows

    Strategy:
    - Buffer incoming audio chunks
    - Process 3-second windows with 1-second overlap
    - Send partial transcripts every 2 seconds
    - Merge overlapping results intelligently
    """

    def __init__(
        self,
        model_name: str = "distil-large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = "en",
        window_duration: float = 3.0,
        hop_duration: float = 2.0,
        sample_rate: int = 16000
    ):
        """
        Args:
            model_name: Model to use (distil-large-v3, large-v3, medium.en, etc.)
            device: "cpu" or "cuda"
            compute_type: "int8" (CPU), "float16" (GPU), "int8_float16" (GPU)
            language: Force language (None for auto-detect)
            window_duration: Size of processing window in seconds (3.0 recommended)
            hop_duration: How often to process new window (2.0 = every 2s)
            sample_rate: Audio sample rate (16000 recommended)
        """
        print(f"[StreamingWhisper] Loading {model_name} on {device}...")

        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type
        )

        self.language = language
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration * sample_rate)
        self.hop_samples = int(hop_duration * sample_rate)

        # Circular buffer for audio
        self.audio_buffer = deque(maxlen=self.window_samples * 2)  # Max 6s buffer

        # Track previous transcript for overlap handling
        self.previous_transcript = ""
        self.last_process_time = 0

        print(f"[StreamingWhisper] Model loaded successfully")
        print(f"[StreamingWhisper] Window: {window_duration}s, Hop: {hop_duration}s")

    def add_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[dict]:
        """
        Add audio chunk to buffer and process if enough data accumulated

        Args:
            audio_chunk: numpy array of float32 audio in [-1, 1]

        Returns:
            dict with transcript if window ready, None otherwise:
                {
                    'text': str,
                    'is_partial': bool,
                    'processing_time_ms': float
                }
        """
        # Ensure float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Add to buffer
        self.audio_buffer.extend(audio_chunk)

        # Check if we have enough for a window and enough time has passed
        current_time = time.time()
        time_since_last = current_time - self.last_process_time

        if len(self.audio_buffer) >= self.window_samples and time_since_last >= (self.hop_samples / self.sample_rate):
            return self._process_window()

        return None

    def _process_window(self) -> dict:
        """Process current window and return transcript"""
        start_time = time.time()

        # Get audio window
        audio_window = np.array(list(self.audio_buffer)[-self.window_samples:], dtype=np.float32)

        # Transcribe with faster-whisper
        segments, info = self.model.transcribe(
            audio_window,
            language=self.language,
            beam_size=1,  # Faster (use 5 for better quality)
            vad_filter=True,  # Filter silence
            vad_parameters=dict(min_silence_duration_ms=500),
            without_timestamps=False,
            word_timestamps=False
        )

        # Collect segments
        transcript_parts = []
        for segment in segments:
            transcript_parts.append(segment.text.strip())

        transcript = " ".join(transcript_parts).strip()

        # Handle overlapping text (merge with previous)
        transcript = self._merge_overlapping(transcript)
        self.previous_transcript = transcript

        processing_time = (time.time() - start_time) * 1000
        self.last_process_time = time.time()

        # Remove processed samples (hop forward)
        for _ in range(min(self.hop_samples, len(self.audio_buffer))):
            self.audio_buffer.popleft()

        return {
            'text': transcript,
            'is_partial': True,  # Always partial in streaming mode
            'processing_time_ms': processing_time,
            'language': info.language if hasattr(info, 'language') else self.language
        }

    # def _merge_overlapping(self, new_transcript: str) -> str:
    #     """
    #     Intelligently merge overlapping transcripts

    #     Since windows overlap, the end of previous transcript may match
    #     the beginning of new transcript. We want to avoid repeating text.
    #     """
    #     if not self.previous_transcript or not new_transcript:
    #         return new_transcript

    #     # Simple approach: Find longest common substring at boundary
    #     prev_words = self.previous_transcript.split()
    #     new_words = new_transcript.split()

    #     # Check last N words of previous against first N words of new
    #     max_overlap = min(len(prev_words), len(new_words), 10)

    #     for overlap_len in range(max_overlap, 0, -1):
    #         prev_suffix = " ".join(prev_words[-overlap_len:])
    #         new_prefix = " ".join(new_words[:overlap_len])

    #         if prev_suffix.lower() == new_prefix.lower():
    #             # Found overlap, merge
    #             merged = self.previous_transcript + " " + " ".join(new_words[overlap_len:])
    #             return merged.strip()

    #     # No overlap found, concatenate with space
    #     return (self.previous_transcript + " " + new_transcript).strip()

    # def finalize(self) -> dict:
    #     """
    #     Process remaining buffer and return final transcript
    #     Call this when user stops speaking
    #     """
    #     if len(self.audio_buffer) == 0:
    #         return {'text': self.previous_transcript, 'is_partial': False, 'processing_time_ms': 0}

    #     # Process whatever is left in buffer
    #     audio_remaining = np.array(list(self.audio_buffer), dtype=np.float32)

    #     start_time = time.time()
    #     segments, info = self.model.transcribe(
    #         audio_remaining,
    #         language=self.language,
    #         beam_size=5,  # Higher quality for final
    #         vad_filter=True,
    #         without_timestamps=False
    #     )

    #     transcript_parts = []
    #     for segment in segments:
    #         transcript_parts.append(segment.text.strip())

    #     transcript = " ".join(transcript_parts).strip()
    #     transcript = self._merge_overlapping(transcript)

    #     processing_time = (time.time() - start_time) * 1000

    #     # Clear buffer
    #     self.audio_buffer.clear()
    #     self.previous_transcript = ""
    #     self.last_process_time = 0

    #     return {
    #         'text': transcript,
    #         'is_partial': False,
    #         'processing_time_ms': processing_time,
    #         'language': info.language if hasattr(info, 'language') else self.language
    #     }

    # streaming_whisper_service.py

    def _merge_overlapping(self, new_transcript: str) -> str:
        # If nothing new, keep what we already had
        if not new_transcript:
            return self.previous_transcript  # <-- BUGFIX: don't drop prior partials

        if not self.previous_transcript:
            return new_transcript

        prev_words = self.previous_transcript.split()
        new_words = new_transcript.split()
        max_overlap = min(len(prev_words), len(new_words), 10)

        for overlap_len in range(max_overlap, 0, -1):
            prev_suffix = " ".join(prev_words[-overlap_len:])
            new_prefix = " ".join(new_words[:overlap_len])
            if prev_suffix.lower() == new_prefix.lower():
                return (self.previous_transcript + " " + " ".join(new_words[overlap_len:])).strip()

        return (self.previous_transcript + " " + new_transcript).strip()


    def finalize(self) -> dict:
        if len(self.audio_buffer) == 0:
            # Return what we have from partials
            out = self.previous_transcript
            self.audio_buffer.clear()
            self.previous_transcript = ""
            self.last_process_time = 0
            return {'text': out, 'is_partial': False, 'processing_time_ms': 0, 'language': self.language}

        audio_remaining = np.array(list(self.audio_buffer), dtype=np.float32)

        start_time = time.time()
        # Make final less aggressive: avoid clipping short tails
        segments, info = self.model.transcribe(
            audio_remaining,
            language=self.language,
            beam_size=5,
            vad_filter=False,                 # <-- key change for finals
            without_timestamps=False
        )

        transcript_parts = [seg.text.strip() for seg in segments]
        new_text = " ".join(transcript_parts).strip()
        merged = self._merge_overlapping(new_text)  # keeps previous if new is empty
        processing_time = (time.time() - start_time) * 1000

        # Clear state
        self.audio_buffer.clear()
        self.previous_transcript = ""        # OK to clear after we computed `merged`
        self.last_process_time = 0

        return {
            'text': merged,
            'is_partial': False,
            'processing_time_ms': processing_time,
            'language': getattr(info, 'language', self.language)
        }

    def reset(self):
        """Reset buffer for new conversation"""
        self.audio_buffer.clear()
        self.previous_transcript = ""
        self.last_process_time = 0


# Example usage
if __name__ == "__main__":
    import soundfile as sf

    # Initialize
    stt = StreamingWhisperSTT(
        model_name="distil-large-v3",
        device="cpu",  # Use "cuda" if you have GPU
        compute_type="int8",  # Use "float16" for GPU
        language="en"
    )

    print("\nSimulating streaming audio processing...\n")

    # Load test audio
    audio, sr = sf.read("test_audio.wav")

    # Simulate streaming by chunking (100ms chunks = 1600 samples @ 16kHz)
    chunk_size = 1600

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]

        # Add chunk and get result if ready
        result = stt.add_audio_chunk(chunk)

        if result:
            print(f"[{result['processing_time_ms']:.0f}ms] Partial: {result['text']}")

    # Get final result
    final = stt.finalize()
    print(f"\n[{final['processing_time_ms']:.0f}ms] Final: {final['text']}")
