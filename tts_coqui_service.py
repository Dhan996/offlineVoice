import os
import platform
import shutil
import time
import numpy as np
import sounddevice as sd
from typing import Optional

from TTS.api import TTS


def _ensure_espeak_on_windows() -> None:
    """Best-effort setup for eSpeak-NG on Windows.

    Coqui TTS uses an eSpeak-NG backend via a DLL. On Windows this DLL is
    typically named either "libespeak-ng.dll" or "espeak-ng.dll". If the
    environment variable "PHONEMIZER_ESPEAK_LIBRARY" is not set, try common
    installation paths and set it automatically so ctypes can load the DLL.
    """
    if platform.system().lower() != "windows":
        return

    # 1) Try to ensure the CLI executable is discoverable (preferred by this TTS version)
    if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
        exe_dirs = [
            r"C:\\Program Files\\eSpeak NG",
            r"C:\\Program Files\\eSpeak NG\\bin",
            r"C:\\Program Files (x86)\\eSpeak NG",
            r"C:\\Program Files (x86)\\eSpeak NG\\bin",
            r"C:\\espeak-ng",
            r"C:\\espeak-ng\\bin",
        ]
        for d in exe_dirs:
            if os.path.isdir(d):
                # if exe exists here, prepend to PATH
                for exe in ("espeak-ng.exe", "espeak.exe"):
                    if os.path.exists(os.path.join(d, exe)):
                        path = os.environ.get("PATH", "")
                        if d not in path:
                            os.environ["PATH"] = d + os.pathsep + path
                        break

    # 2) Set the DLL path as a fallback for other phonemizers that use ctypes
    env_key = "PHONEMIZER_ESPEAK_LIBRARY"
    if os.environ.get(env_key):
        return

    candidates = [
        r"C:\\Program Files\\eSpeak NG\\libespeak-ng.dll",
        r"C:\\Program Files\\eSpeak NG\\espeak-ng.dll",
        r"C:\\Program Files\\eSpeak NG\\bin\\libespeak-ng.dll",
        r"C:\\Program Files\\eSpeak NG\\bin\\espeak-ng.dll",
        r"C:\\Program Files\\eSpeak NG\\lib\\libespeak-ng.dll",
        r"C:\\Program Files\\eSpeak NG\\lib\\espeak-ng.dll",
        r"C:\\Program Files (x86)\\eSpeak NG\\libespeak-ng.dll",
        r"C:\\Program Files (x86)\\eSpeak NG\\espeak-ng.dll",
        r"C:\\Program Files (x86)\\eSpeak NG\\bin\\libespeak-ng.dll",
        r"C:\\Program Files (x86)\\eSpeak NG\\bin\\espeak-ng.dll",
        r"C:\\Program Files (x86)\\eSpeak NG\\lib\\libespeak-ng.dll",
        r"C:\\Program Files (x86)\\eSpeak NG\\lib\\espeak-ng.dll",
        r"C:\\espeak-ng\\libespeak-ng.dll",
        r"C:\\espeak-ng\\espeak-ng.dll",
        r"C:\\espeak-ng\\bin\\libespeak-ng.dll",
        r"C:\\espeak-ng\\bin\\espeak-ng.dll",
        r"C:\\espeak-ng\\lib\\libespeak-ng.dll",
        r"C:\\espeak-ng\\lib\\espeak-ng.dll",
    ]

    for dll in candidates:
        if os.path.exists(dll):
            os.environ[env_key] = dll
            # Help downstream loaders that rely on PATH search
            dll_dir = os.path.dirname(dll)
            path = os.environ.get("PATH", "")
            if dll_dir not in path:
                os.environ["PATH"] = dll_dir + os.pathsep + path
            break


class CoquiTTS:
    """
    Coqui TTS wrapper that synthesizes text to audio and plays via sounddevice.
    """

    def __init__(self, model_name: str = "tts_models/en/vctk/vits", out_rate: int = 22050, speaker: Optional[str] = None):
        self.model_name = model_name
        # On Windows, try to auto-configure eSpeak-NG if not present.
        _ensure_espeak_on_windows()
        self.tts = TTS(model_name)
        self.out_rate = out_rate

        # Set default speaker for multi-speaker models
        self.speaker = speaker
        if self.speaker is None and self.tts.is_multi_speaker:
            # Use the first available speaker as default
            self.speaker = self.tts.speakers[0] if self.tts.speakers else None
            print(f"[TTS] Multi-speaker model detected. Available speakers: {self.tts.speakers[:5]}...")
            print(f"[TTS] Using speaker: {self.speaker}")
        else:
            print(f"[TTS] Single-speaker model or speaker set to: {self.speaker}")

    def synth(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros((0,), dtype=np.float32)

        # Pass speaker parameter if model is multi-speaker
        start_time = time.time()
        print(f"[TTS] Synthesizing: '{text[:50]}...' with speaker: {self.speaker}")

        if self.tts.is_multi_speaker and self.speaker:
            wav = self.tts.tts(text, speaker=self.speaker)
        else:
            wav = self.tts.tts(text)

        wav = np.asarray(wav, dtype=np.float32)
        elapsed = time.time() - start_time

        sample_rate = self.tts.synthesizer.output_sample_rate if hasattr(self.tts, "synthesizer") else self.out_rate
        duration = len(wav) / sample_rate if sample_rate > 0 else 0
        print(f"[TTS] Generated {len(wav)} samples ({duration:.2f}s audio) in {elapsed:.2f}s")

        # Normalize lightly
        peak = np.max(np.abs(wav)) if wav.size > 0 else 1.0
        if peak > 0:
            wav = wav / (peak + 1e-8) * 0.9
        return wav

    def play(self, wav_f32: np.ndarray, sample_rate: Optional[int] = None) -> None:
        if sample_rate is None:
            sample_rate = self.tts.synthesizer.output_sample_rate if hasattr(self.tts, "synthesizer") else self.out_rate
        if wav_f32 is None or wav_f32.size == 0:
            print("[TTS] No audio to play (empty array)")
            return

        start_time = time.time()
        duration = len(wav_f32) / sample_rate
        print(f"[TTS] Playing {len(wav_f32)} samples ({duration:.2f}s) at {sample_rate}Hz")
        sd.play(wav_f32, samplerate=sample_rate, blocking=True)
        elapsed = time.time() - start_time
        print(f"[TTS] Playback completed in {elapsed:.2f}s")
