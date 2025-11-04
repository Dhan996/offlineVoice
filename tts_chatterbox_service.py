import os
import platform
import time
import numpy as np
import sounddevice as sd
from typing import Optional
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import torchaudio
    from chatterbox.tts import ChatterboxTTS as ChatterboxModel
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    print("[TTS] chatterbox-tts not available. Install with: pip install chatterbox-tts")
    ChatterboxModel = None
    ChatterboxMultilingualTTS = None
    torchaudio = None

# Optional audio loading libraries (used as fallbacks)
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None

try:
    import scipy.io.wavfile
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _get_optimal_device() -> str:
    """
    Automatically detect the best available device for TTS inference.
    
    Returns:
        Device string: 'cuda' for NVIDIA GPU, 'cpu' for CPU
    """
    if not TORCH_AVAILABLE:
        print("[TTS] PyTorch not available, using CPU")
        return "cpu"
    
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[TTS] CUDA GPU detected: {gpu_name}")
        return "cuda"
    
    # Check for MPS (Apple Silicon) but don't use it
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("[TTS] Apple Silicon GPU (MPS) detected but not used")
        print("[TTS] Using CPU instead (Apple Silicon CPUs are fast for inference)")
        return "mps"
    
    print("[TTS] Using CPU")
    return "cpu"


class ChatterboxTTS:
    """
    Chatterbox TTS wrapper with voice cloning support.
    
    Chatterbox is a 500M parameter Llama-based TTS model with:
    - Fast inference (RTF ~0.5 on RTX 4090)
    - Voice cloning with short reference audio
    - Emotion exaggeration control
    - Low latency (~400-500ms to first chunk)
    
    Installation: pip install chatterbox-tts
    Repo: https://github.com/resemble-ai/chatterbox
    """

    def __init__(
        self,
        model_name: Optional[str] = None,  # For orchestrator compatibility (unused by Chatterbox)
        multilingual: bool = False,
        out_rate: int = 24000,  # Chatterbox outputs 24kHz
        speaker_wav: Optional[str] = None,
        device: Optional[str] = None,
        use_gpu: bool = True,
        exaggeration: float = 0.5,  # Not used - base API doesn't support this
        cfg: float = 0.5,            # Not used - base API doesn't support this
    ):
        """
        Initialize Chatterbox TTS model with voice cloning support.
        
        Args:
            model_name: Unused (for compatibility with Coqui-style orchestrators)
            multilingual: Use multilingual model (supports 23 languages)
            out_rate: Output sample rate (Chatterbox native is 24kHz)
            speaker_wav: Path to reference audio for voice cloning (3-10 seconds recommended)
            device: Force specific device ('cuda', 'cpu'). If None, auto-detect
            use_gpu: Whether to attempt GPU acceleration (default: True)
            exaggeration: Emotion intensity (stored but not used - base API doesn't support)
            cfg: CFG/pace control (stored but not used - base API doesn't support)
            
        Note: The base Chatterbox-TTS API only supports text and audio_prompt_path.
        Advanced parameters like exaggeration and cfg_weight require custom implementations
        or API wrappers (see chatterbox-tts-api or other server implementations).
        """
        if not CHATTERBOX_AVAILABLE:
            raise ImportError(
                "Chatterbox requires the chatterbox-tts package. "
                "Install with: pip install chatterbox-tts"
            )
        
        # model_name is accepted for orchestrator compatibility but not used
        if model_name is not None:
            print(f"[TTS] Note: model_name '{model_name}' provided but ignored (Chatterbox uses fixed models)")
        
        self.multilingual = multilingual
        self.out_rate = out_rate
        self.speaker_wav = speaker_wav
        # Note: exaggeration and cfg are not supported by the base Chatterbox API
        # These would need custom model modifications or API wrappers to work
        self.exaggeration = exaggeration  # Stored but not used in generation
        self.cfg = cfg  # Stored but not used in generation
        
        # Track if conditionals have been prepared
        self._conditionals_prepared = False
        
        # Determine device
        if device is None:
            self.device = _get_optimal_device() if use_gpu else "cpu"
        else:
            self.device = device
        
        print(f"[TTS] Initializing Chatterbox{'Multilingual' if multilingual else ''}")
        print(f"[TTS] Device: {self.device}")
        print(f"[TTS] Note: Base API does not support exaggeration/cfg parameters")
        
        init_start = time.time()
        
        try:
            # Load the appropriate model
            if multilingual:
                print("[TTS] Loading multilingual model (supports 23 languages)...")
                self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            else:
                print("[TTS] Loading English-only model...")
                self.model = ChatterboxModel.from_pretrained(device=self.device)
            
            # Get sample rate from model
            self.out_rate = self.model.sr
            
        except Exception as e:
            print(f"[TTS] Error loading Chatterbox: {e}")
            print("\nInstallation instructions:")
            print("  pip install chatterbox-tts")
            print("\nOr install from source:")
            print("  git clone https://github.com/resemble-ai/chatterbox.git")
            print("  cd chatterbox")
            print("  pip install -e .")
            raise RuntimeError(f"Failed to load Chatterbox: {e}")
        
        init_time = time.time() - init_start
        print(f"[TTS] Chatterbox loaded in {init_time:.2f}s")
        print(f"[TTS] Sample rate: {self.out_rate}Hz")
        
        if self.speaker_wav:
            if not os.path.exists(self.speaker_wav):
                print(f"[TTS] Warning: Reference audio not found: {self.speaker_wav}")
                self.speaker_wav = None
            else:
                print(f"[TTS] Voice cloning enabled with: {self.speaker_wav}")
                # Call prepare_conditionals once to cache the reference audio
                print(f"[TTS] Preparing conditionals (caching reference audio)...")
                prep_start = time.time()
                self.model.prepare_conditionals(self.speaker_wav, exaggeration=self.exaggeration)
                prep_time = time.time() - prep_start
                print(f"[TTS] Conditionals prepared and cached in {prep_time:.2f}s")
                print(f"[TTS] Reference audio will be reused for all subsequent requests")
                self._conditionals_prepared = True
        else:
            print(f"[TTS] Using default Chatterbox voice (no reference audio)")
            self._conditionals_prepared = False

    def synth(
        self, 
        text: str, 
        speaker_wav: Optional[str] = None,
        language: Optional[str] = None
    ) -> np.ndarray:
        """
        Synthesize text to audio waveform with optional voice cloning.
        
        Args:
            text: Text to synthesize
            speaker_wav: Optional path to reference audio (overrides init setting)
            language: Language code for multilingual model (e.g., 'en', 'es', 'fr')
            
        Returns:
            Audio waveform as float32 numpy array
        """
        if not text or not text.strip():
            return np.zeros((0,), dtype=np.float32)
        
        # Use provided speaker_wav or fall back to instance setting
        use_speaker_wav = speaker_wav or self.speaker_wav
        
        # If a different reference audio is provided, prepare new conditionals
        if speaker_wav and speaker_wav != self.speaker_wav:
            if os.path.exists(speaker_wav):
                print(f"[TTS] Preparing conditionals for new reference audio: {speaker_wav}")
                self.model.prepare_conditionals(speaker_wav, exaggeration=self.exaggeration)
                self._conditionals_prepared = True
            else:
                print(f"[TTS] Warning: Reference audio not found: {speaker_wav}")
                use_speaker_wav = None
        
        start_time = time.time()
        text_preview = text[:50] + "..." if len(text) > 50 else text
        
        if use_speaker_wav:
            print(f"[TTS] Synthesizing with voice cloning: '{text_preview}'")
            if self._conditionals_prepared:
                print(f"[TTS] Using cached conditionals (fast!)")
        else:
            print(f"[TTS] Synthesizing: '{text_preview}'")
        
        try:
            # Generate speech using Chatterbox API
            # If conditionals are prepared, DON'T pass audio_prompt_path - it will use cached conditionals
            if use_speaker_wav and self._conditionals_prepared:
                # Use cached conditionals - much faster!
                if self.multilingual and language:
                    wav = self.model.generate(
                        text,
                        language_id=language,
                        exaggeration=self.exaggeration,
                        cfg_weight=self.cfg,
                    )
                elif self.multilingual:
                    # Multilingual but no language specified - use default
                    wav = self.model.generate(
                        text,
                        language_id="en",
                        exaggeration=self.exaggeration,
                        cfg_weight=self.cfg,
                    )
                else:
                    # Non-multilingual model - don't pass language_id
                    wav = self.model.generate(
                        text,
                        exaggeration=self.exaggeration,
                        cfg_weight=self.cfg,
                    )
            elif use_speaker_wav:
                # Conditionals not prepared, pass the file path (slower - first time only)
                print(f"[TTS] Loading reference audio: {use_speaker_wav}")
                if self.multilingual and language:
                    wav = self.model.generate(
                        text,
                        language_id=language,
                        audio_prompt_path=use_speaker_wav,
                        exaggeration=self.exaggeration,
                        cfg_weight=self.cfg,
                    )
                elif self.multilingual:
                    wav = self.model.generate(
                        text,
                        language_id="en",
                        audio_prompt_path=use_speaker_wav,
                        exaggeration=self.exaggeration,
                        cfg_weight=self.cfg,
                    )
                else:
                    # Non-multilingual model
                    wav = self.model.generate(
                        text,
                        audio_prompt_path=use_speaker_wav,
                        exaggeration=self.exaggeration,
                        cfg_weight=self.cfg,
                    )
            else:
                # Default voice - need to prepare conditionals first
                if not self._conditionals_prepared:
                    # For default voice, we still need conditionals
                    # This shouldn't happen if init was done correctly
                    print(f"[TTS] Warning: No conditionals prepared, this may fail")
                
                if self.multilingual and language:
                    wav = self.model.generate(
                        text,
                        language_id=language,
                        exaggeration=self.exaggeration,
                        cfg_weight=self.cfg,
                    )
                elif self.multilingual:
                    wav = self.model.generate(
                        text,
                        language_id="en",
                        exaggeration=self.exaggeration,
                        cfg_weight=self.cfg,
                    )
                else:
                    # Non-multilingual model
                    wav = self.model.generate(
                        text,
                        exaggeration=self.exaggeration,
                        cfg_weight=self.cfg,
                    )
            
            # Convert to numpy if tensor
            if torch.is_tensor(wav):
                wav = wav.cpu().numpy()
            
            # Ensure 1D array
            if wav.ndim > 1:
                wav = wav.squeeze()
            
            wav = wav.astype(np.float32)
        
        except Exception as e:
            print(f"[TTS] Error during synthesis: {e}")
            print(f"[TTS] Returning silence")
            return np.zeros((self.out_rate,), dtype=np.float32)  # 1 second of silence
        
        elapsed = time.time() - start_time
        duration = len(wav) / self.out_rate if self.out_rate > 0 else 0
        rtf = elapsed / duration if duration > 0 else 0
        
        print(f"[TTS] Generated {len(wav)} samples ({duration:.2f}s audio) in {elapsed:.2f}s (RTF: {rtf:.2f}x)")
        
        # Normalize lightly
        peak = np.max(np.abs(wav)) if wav.size > 0 else 1.0
        if peak > 0:
            wav = wav * (0.9 / (peak + 1e-8))
        
        return wav

    def play(self, wav_f32: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """
        Play audio waveform through speakers.
        
        Args:
            wav_f32: Audio waveform as float32 array
            sample_rate: Sample rate (if None, uses model's output rate)
        """
        if sample_rate is None:
            sample_rate = self.out_rate
        
        if wav_f32 is None or wav_f32.size == 0:
            print("[TTS] No audio to play (empty array)")
            return
        
        start_time = time.time()
        duration = len(wav_f32) / sample_rate
        print(f"[TTS] Playing {len(wav_f32)} samples ({duration:.2f}s) at {sample_rate}Hz")
        sd.play(wav_f32, samplerate=sample_rate, blocking=True)
        elapsed = time.time() - start_time
        print(f"[TTS] Playback completed in {elapsed:.2f}s")
    
    def set_exaggeration(self, value: float) -> None:
        """
        Adjust emotion exaggeration (0.0-1.0).
        This parameter is now properly supported via the generate() method.
        """
        self.exaggeration = max(0.0, min(1.0, value))
        print(f"[TTS] Exaggeration set to: {self.exaggeration}")
        print(f"[TTS] Will be applied on next synthesis")
    
    def set_cfg(self, value: float) -> None:
        """
        Adjust CFG / pacing control (0.0-1.0).
        This parameter is now properly supported via the generate() method.
        """
        self.cfg = max(0.0, min(1.0, value))
        print(f"[TTS] CFG weight set to: {self.cfg}")
        print(f"[TTS] Will be applied on next synthesis")
    
    def get_device_info(self) -> dict:
        """
        Get information about the device being used.
        
        Returns:
            Dictionary with device information
        """
        info = {
            "device": self.device,
            "multilingual": self.multilingual,
            "has_reference_audio": self.speaker_wav is not None,
            "exaggeration": self.exaggeration,
            "cfg": self.cfg,
            "sample_rate": self.out_rate,
        }
        
        if TORCH_AVAILABLE and self.device == "cuda":
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
            info["cuda_memory_reserved"] = f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
        
        return info
    
    def clone_voice(
        self, 
        text: str, 
        reference_audio_path: str,
        language: Optional[str] = None,
        play_audio: bool = False
    ) -> np.ndarray:
        """
        Clone a voice from reference audio and synthesize text.
        
        This is a convenience method that combines voice cloning and synthesis.
        
        Args:
            text: Text to synthesize in the cloned voice
            reference_audio_path: Path to reference audio file (3-10 seconds recommended)
            language: Language code for multilingual model (e.g., 'en', 'es', 'fr')
            play_audio: Whether to play the generated audio immediately
            
        Returns:
            Audio waveform as float32 numpy array
            
        Example:
            >>> tts = ChatterboxTTS()
            >>> audio = tts.clone_voice(
            ...     "Hello, this is a test of voice cloning!",
            ...     "path/to/reference.wav"
            ... )
        """
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
        
        print(f"\n[Voice Cloning] Using reference: {reference_audio_path}")
        wav = self.synth(text, speaker_wav=reference_audio_path, language=language)
        
        if play_audio:
            self.play(wav)
        
        return wav
    
    def set_reference_audio(self, audio_path: Optional[str]) -> None:
        """
        Set or update the default reference audio for voice cloning.
        This will prepare and cache the conditionals for faster synthesis.
        
        Args:
            audio_path: Path to reference audio file, or None to disable voice cloning
        """
        if audio_path and not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
        
        self.speaker_wav = audio_path
        if audio_path:
            print(f"[TTS] Reference audio set to: {audio_path}")
            print(f"[TTS] Preparing conditionals...")
            self.model.prepare_conditionals(audio_path, exaggeration=self.exaggeration)
            self._conditionals_prepared = True
            print(f"[TTS] Conditionals prepared and cached")
        else:
            print(f"[TTS] Reference audio cleared - using default voice")
            self._conditionals_prepared = False
    
    def save_audio(self, wav: np.ndarray, output_path: str, sample_rate: Optional[int] = None) -> None:
        """
        Save audio waveform to a file.
        
        Args:
            wav: Audio waveform as numpy array
            output_path: Path to save the audio file (.wav)
            sample_rate: Sample rate (if None, uses model's output rate)
        """
        if sample_rate is None:
            sample_rate = self.out_rate
        
        try:
            import scipy.io.wavfile as wavfile
            
            # Convert float32 to int16 for WAV file
            wav_int16 = (wav * 32767).astype(np.int16)
            wavfile.write(output_path, sample_rate, wav_int16)
            
            duration = len(wav) / sample_rate
            print(f"[TTS] Audio saved to: {output_path} ({duration:.2f}s, {sample_rate}Hz)")
            
        except ImportError:
            print("[TTS] scipy not available for saving audio. Install with: pip install scipy")
            raise
        except Exception as e:
            print(f"[TTS] Error saving audio: {e}")
            raise
    
    def batch_clone(
        self,
        texts: list[str],
        reference_audio_path: str,
        output_dir: str = "cloned_audio",
        language: Optional[str] = None,
        save_files: bool = True
    ) -> list[np.ndarray]:
        """
        Clone voice for multiple texts in batch.
        
        Args:
            texts: List of texts to synthesize
            reference_audio_path: Path to reference audio file
            output_dir: Directory to save output files (if save_files=True)
            language: Language code for multilingual model
            save_files: Whether to save audio files to disk
            
        Returns:
            List of audio waveforms
        """
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
        
        if save_files:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n[Batch Clone] Processing {len(texts)} texts...")
        results = []
        
        for i, text in enumerate(texts, 1):
            print(f"\n[Batch Clone] {i}/{len(texts)}")
            wav = self.clone_voice(text, reference_audio_path, language=language)
            results.append(wav)
            
            if save_files:
                output_path = os.path.join(output_dir, f"cloned_{i:03d}.wav")
                self.save_audio(wav, output_path)
        
        print(f"\n[Batch Clone] Completed {len(results)} synthesizations")
        return results


def create_test_reference_audio(output_path: str = "test_reference.wav", duration: float = 3.0) -> str:
    """
    Create a simple test reference audio file (sine wave sweep).
    This is just for testing purposes - real voice cloning needs actual speech audio.
    
    Args:
        output_path: Path to save the test audio
        duration: Duration in seconds
        
    Returns:
        Path to the created audio file
    """
    sample_rate = 24000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a frequency sweep (not realistic speech, but good for testing)
    freq_start = 200
    freq_end = 800
    chirp = np.sin(2 * np.pi * (freq_start * t + (freq_end - freq_start) * t**2 / (2 * duration)))
    
    # Add some amplitude modulation to make it more interesting
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    audio = (chirp * envelope * 0.3).astype(np.float32)
    
    # Save using scipy
    try:
        import scipy.io.wavfile as wavfile
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, audio_int16)
        print(f"[Test] Created test reference audio: {output_path}")
        return output_path
    except ImportError:
        print("[Test] scipy not available. Install with: pip install scipy")
        raise