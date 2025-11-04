"""
Server-side Voice Activity Detection using Silero VAD
High-accuracy VAD model for detecting speech in audio streams
"""
import torch
import numpy as np
from typing import Optional, Tuple
from collections import deque

# --- add near the top ---
import webrtcvad

class WebRTCVAD:
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000, frame_ms: int = 20,
                 min_speech_duration_ms: int = 200, min_silence_duration_ms: int = 900):
        """
        aggressiveness: 0 (least) .. 3 (most aggressive)
        frame_ms: 10/20/30 only (WebRTC requirement)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * (frame_ms / 1000.0)) * 2  # int16 mono
        self.min_speech = min_speech_duration_ms
        self.min_silence = min_silence_duration_ms
        self.is_speaking = False
        self.speech_ms = 0
        self.silence_ms = 0

    def process_chunk(self, audio_chunk_f32: np.ndarray) -> dict:
        # Convert to 16-bit PCM bytes (required by WebRTC VAD), 16 kHz mono
        pcm16 = (np.clip(audio_chunk_f32, -1.0, 1.0) * 32768.0).astype(np.int16).tobytes()

        # Iterate fixed-size frames
        speech_frames = 0; total_frames = 0
        for i in range(0, len(pcm16), self.frame_bytes):
            frame = pcm16[i:i+self.frame_bytes]
            if len(frame) < self.frame_bytes: break
            total_frames += 1
            if self.vad.is_speech(frame, self.sample_rate):
                speech_frames += 1

        prob = (speech_frames / total_frames) if total_frames else 0.0
        is_speech = prob > 0.5  # simple vote

        speech_started = False
        speech_ended = False

        frame_ms = self.frame_bytes / 2 / self.sample_rate * 1000.0
        if is_speech:
            self.speech_ms += total_frames * frame_ms
            self.silence_ms = 0
            if not self.is_speaking and self.speech_ms >= self.min_speech:
                self.is_speaking = True
                speech_started = True
        else:
            self.silence_ms += total_frames * frame_ms
            if self.is_speaking and self.silence_ms >= self.min_silence:
                self.is_speaking = False
                speech_ended = True
                self.speech_ms = 0

        return {
            'is_speech': is_speech,
            'speech_started': speech_started,
            'speech_ended': speech_ended,
            'probability': float(prob),
            'is_speaking': self.is_speaking
        }

class SileroVAD:
    """
    Voice Activity Detection using Silero VAD model

    Advantages over frontend VAD:
    - Much more accurate (ML-based, not just RMS threshold)
    - Consistent behavior across all clients
    - Easy to tune from server config
    - Handles noise, music, background sounds better
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 600,  # 1 second for faster response
        sample_rate: int = 16000
    ):
        """
        Args:
            threshold: Speech probability threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech duration to trigger
            min_silence_duration_ms: Silence duration before stopping
            sample_rate: Audio sample rate (must be 8000 or 16000)
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate

        # Load Silero VAD model
        print("[VAD] Loading Silero VAD model...")
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.model = model
            self.model.eval()

            # Utils functions
            (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
            self.get_speech_timestamps = get_speech_timestamps

            print("[VAD] Silero VAD loaded successfully")

        except Exception as e:
            print(f"[VAD] Failed to load Silero VAD: {e}")
            print("[VAD] Falling back to simple RMS-based VAD")
            self.model = None

        # State tracking
        self.reset()

    def reset(self):
        """Reset VAD state for new session"""
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 2))  # 2s buffer

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Process audio chunk and return VAD decision

        Args:
            audio_chunk: numpy array of float32 audio in [-1, 1]

        Returns:
            dict with:
                - 'is_speech': bool, whether current chunk contains speech
                - 'speech_started': bool, speech just started
                - 'speech_ended': bool, speech just ended
                - 'probability': float, speech probability (0.0-1.0)
        """
        # Ensure float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Add to buffer
        self.audio_buffer.extend(audio_chunk)

        # Get speech probability
        if self.model is not None:
            probability = self._silero_vad(audio_chunk)
        else:
            probability = self._simple_vad(audio_chunk)

        is_speech = probability > self.threshold

        # State machine
        speech_started = False
        speech_ended = False

        if is_speech:
            if not self.is_speaking:
                # Speech just started
                self.is_speaking = True
                self.speech_start_time = 0  # Will track in milliseconds
                speech_started = True
                print(f"[VAD] Speech started (prob={probability:.3f})")

            self.silence_start_time = None
            self.speech_start_time += len(audio_chunk) / self.sample_rate * 1000

        else:  # Silence
            if self.is_speaking:
                if self.silence_start_time is None:
                    self.silence_start_time = 0

                self.silence_start_time += len(audio_chunk) / self.sample_rate * 1000

                # Check if silence long enough to end speech
                if (self.silence_start_time >= self.min_silence_duration_ms and
                    self.speech_start_time >= self.min_speech_duration_ms):
                    # Speech ended
                    self.is_speaking = False
                    speech_ended = True
                    print(f"[VAD] Speech ended (duration={self.speech_start_time:.0f}ms)")
                    self.speech_start_time = None
                    self.silence_start_time = None

        return {
            'is_speech': is_speech,
            'speech_started': speech_started,
            'speech_ended': speech_ended,
            'probability': probability,
            'is_speaking': self.is_speaking
        }

    def _silero_vad(self, audio_chunk: np.ndarray) -> float:
        """Use Silero VAD model to get speech probability"""
        try:
            # Silero VAD expects chunks of specific size (512 samples for 16kHz)
            # For simplicity, we'll process the whole chunk
            audio_tensor = torch.from_numpy(audio_chunk)

            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()

            return speech_prob

        except Exception as e:
            print(f"[VAD] Silero VAD error: {e}, falling back to RMS")
            return self._simple_vad(audio_chunk)

    def _simple_vad(self, audio_chunk: np.ndarray) -> float:
        """Simple RMS-based VAD fallback"""
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        # Map RMS to probability-like value
        # Typical speech RMS is 0.01-0.3
        probability = min(1.0, rms / 0.1)
        return probability

    def get_buffered_audio(self) -> np.ndarray:
        """Get buffered audio for transcription"""
        return np.array(list(self.audio_buffer), dtype=np.float32)


class SimpleVAD:
    """
    Simple RMS-based VAD (fallback if Silero not available)
    Fast and lightweight, good enough for most cases
    """

    def __init__(
        self,
        threshold: float = 0.01,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 600,  # 1 second for faster response
        sample_rate: int = 16000
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate
        self.reset()
        print(f"[VAD] Using simple RMS-based VAD (threshold={threshold})")

    def reset(self):
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 2))

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        self.audio_buffer.extend(audio_chunk)

        # Calculate RMS
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        is_speech = rms > self.threshold

        speech_started = False
        speech_ended = False

        if is_speech:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = 0
                speech_started = True
                print(f"[VAD] Speech started (RMS={rms:.4f})")

            self.silence_start_time = None
            self.speech_start_time += len(audio_chunk) / self.sample_rate * 1000

        else:
            if self.is_speaking:
                if self.silence_start_time is None:
                    self.silence_start_time = 0

                self.silence_start_time += len(audio_chunk) / self.sample_rate * 1000

                if (self.silence_start_time >= self.min_silence_duration_ms and
                    self.speech_start_time >= self.min_speech_duration_ms):
                    self.is_speaking = False
                    speech_ended = True
                    print(f"[VAD] Speech ended (duration={self.speech_start_time:.0f}ms)")
                    self.speech_start_time = None
                    self.silence_start_time = None

        return {
            'is_speech': is_speech,
            'speech_started': speech_started,
            'speech_ended': speech_ended,
            'probability': min(1.0, rms / 0.1),
            'is_speaking': self.is_speaking
        }

    def get_buffered_audio(self) -> np.ndarray:
        return np.array(list(self.audio_buffer), dtype=np.float32)


def create_vad(use_silero: bool = True,use_webrtc: bool = False, **kwargs):
    if use_webrtc:
        return WebRTCVAD(
            aggressiveness=kwargs.get("aggressiveness", 2),
            sample_rate=kwargs.get("sample_rate", 16000),
            frame_ms=kwargs.get("frame_ms", 20),
            min_speech_duration_ms=kwargs.get("min_speech_duration_ms", 200),
            min_silence_duration_ms=kwargs.get("min_silence_duration_ms", 900),
        )
    """Factory function to create VAD instance"""
    if use_silero:
        try:
            return SileroVAD(**kwargs)
        except Exception as e:
            print(f"[VAD] Failed to create Silero VAD: {e}")
            print("[VAD] Falling back to SimpleVAD")
            return SimpleVAD(**kwargs)
    else:
        return SimpleVAD(**kwargs)


# Example usage
if __name__ == "__main__":
    import soundfile as sf

    # Create VAD
    vad = create_vad(
        use_silero=True,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=1500
    )

    # Load test audio
    audio, sr = sf.read("test_audio.wav")

    # Process in chunks
    chunk_size = 1600  # 100ms @ 16kHz
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        result = vad.process_chunk(chunk)

        if result['speech_started']:
            print("🎤 Speech started!")
        if result['speech_ended']:
            print("🔇 Speech ended!")
            print(f"   Buffered audio: {len(vad.get_buffered_audio())/sr:.2f}s")
