import argparse
import asyncio
import os
import platform
import shutil
import sys
import time
from typing import Optional, List, Dict

import numpy as np
import sounddevice as sd

from whisper_service import WhisperSTT

# Import both TTS services
from tts_coqui_service import CoquiTTS
from tts_chatterbox_service import ChatterboxTTS

# Reuse existing Ollama client from repo
from config import CONFIG as GLOBAL_CONFIG  # type: ignore
from llm_ollama_service import OllamaLLM  # type: ignore


def db_print(*a):
    print(*a, flush=True)


def record_push_to_talk(sample_rate: int, channels: int = 1, max_seconds: int = 20, silence_ms: int = 800, thresh: float = 0.01) -> np.ndarray:
    """
    Record mic until a trailing silence is detected or max duration, using a simple energy VAD.
    Returns mono float32 waveform.
    """
    db_print("[mic] Speak now… (auto-stops on silence)")
    block = 1024
    silence_blocks_needed = int((silence_ms/1000.0) * sample_rate / block)
    silence_run = 0
    captured: List[np.ndarray] = []

    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype='float32', blocksize=block) as stream:
        start = time.time()
        while time.time() - start < max_seconds:
            data, _ = stream.read(block)
            # mono
            if data.ndim > 1:
                data = np.mean(data, axis=1, dtype=np.float32)
            else:
                data = data.astype(np.float32)
            captured.append(data)
            # energy VAD
            level = float(np.sqrt(np.mean(data*data) + 1e-9))
            if level < thresh:
                silence_run += 1
            else:
                silence_run = 0
            if silence_run >= silence_blocks_needed and len(captured) > silence_blocks_needed:
                break
    wav = np.concatenate(captured) if captured else np.zeros((0,), dtype=np.float32)
    db_print(f"[mic] captured {len(wav)/sample_rate:.2f}s")
    return wav


async def query_ollama(user_text: str, model: str, base: str) -> str:
    messages: List[Dict] = [
        {"role": "system", "content": GLOBAL_CONFIG.get("system_prompt", "You are a helpful assistant.")},
        {"role": "user", "content": user_text},
    ]
    # Patch runtime options without mutating global config
    original_model = GLOBAL_CONFIG.get("ollama_model")
    original_base = GLOBAL_CONFIG.get("ollama_base")
    GLOBAL_CONFIG["ollama_model"] = model
    GLOBAL_CONFIG["ollama_base"] = base
    try:
        start_time = time.time()
        db_print(f"[LLM] Querying {model}...")

        llm = OllamaLLM()
        out_parts: List[str] = []
        async for tok in llm.stream(messages):
            out_parts.append(tok)

        result = "".join(out_parts).strip()
        elapsed = time.time() - start_time
        db_print(f"[LLM] Response generated in {elapsed:.2f}s ({len(result)} chars)")

        return result
    finally:
        GLOBAL_CONFIG["ollama_model"] = original_model
        GLOBAL_CONFIG["ollama_base"] = original_base


def pick_io_devices():
    try:
        info = sd.query_devices()
        db_print("[audio] Devices:")
        for i, d in enumerate(info):
            db_print(f"  {i:2d}: {d['name']} (in:{d['max_input_channels']}, out:{d['max_output_channels']})")
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description="Complete offline voice agent: Whisper + Ollama + TTS (Coqui or Chatterbox)")
    p.add_argument('--ollama-base', default='http://localhost:11434')
    p.add_argument('--ollama-model', default='gemma3:4b')
    p.add_argument('--whisper-model', default='large-v3')
    p.add_argument('--whisper-device', default='cuda', choices=['cpu','cuda'])
    p.add_argument('--compute-type', default='float16', help='faster-whisper compute type (e.g., int8, float16)')
    p.add_argument('--lang', default=None)
    
    # TTS Engine Selection
    p.add_argument('--tts-engine', default='coqui', choices=['coqui', 'chatterbox'],
                   help='TTS engine to use: coqui (default) or chatterbox')
    p.add_argument('--tts-model', default='tts_models/en/vctk/vits',
                   help='TTS model (Coqui only)')
    
    # Chatterbox-specific options
    p.add_argument('--chatterbox-multilingual', action='store_true',
                   help='Use Chatterbox multilingual model (supports 23 languages)')
    p.add_argument('--chatterbox-speaker-wav', default=None,
                   help='Path to reference audio for Chatterbox voice cloning')
    p.add_argument('--chatterbox-device', default=None, choices=['cpu', 'cuda', 'mps'],
                   help='Device for Chatterbox (auto-detect if not specified)')
    
    p.add_argument('--sample-rate', type=int, default=16000)
    p.add_argument('--input-device', type=int, default=None)
    p.add_argument('--output-device', type=int, default=None)
    args = p.parse_args()

    # Configure optional I/O device indices
    if args.input_device is not None or args.output_device is not None:
        curr_in, curr_out = None, None
        try:
            curr = sd.default.device
            if isinstance(curr, (list, tuple)):
                curr_in, curr_out = curr[0], curr[1]
            elif isinstance(curr, int):
                curr_in = curr
        except Exception:
            pass
        sd.default.device = (
            args.input_device if args.input_device is not None else curr_in,
            args.output_device if args.output_device is not None else curr_out,
        )

    pick_io_devices()

    # Initialize STT
    stt = WhisperSTT(model_name=args.whisper_model, device=args.whisper_device, compute_type=args.compute_type, language=args.lang)
    
    # Initialize TTS based on selected engine
    db_print(f"\n[TTS] Initializing {args.tts_engine.upper()} engine...")
    
    if args.tts_engine == 'coqui':
        # Coqui TTS initialization with eSpeak checks
        try:
            os_info = f"{platform.system()} {platform.release()}" if hasattr(platform, "release") else platform.system()
        except Exception:
            os_info = platform.system()
        try:
            py_arch = platform.architecture()[0]
        except Exception:
            py_arch = "unknown"
        db_print(f"[check] OS: {os_info} | Python: {py_arch}")
        pre_lib = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
        if pre_lib:
            db_print(f"[check] PHONEMIZER_ESPEAK_LIBRARY={pre_lib} (exists: {os.path.exists(pre_lib)})")
        espeak_ng_path = shutil.which("espeak-ng") or shutil.which("espeak")
        if espeak_ng_path:
            db_print(f"[check] Found espeak executable: {espeak_ng_path}")
        else:
            db_print("[check] espeak-ng/espeak not on PATH (will try auto-detect)")
        
        try:
            tts = CoquiTTS(model_name=args.tts_model)
            post_lib = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
            if post_lib:
                db_print(f"[check] Using eSpeak library: {post_lib}")
        except Exception as e:
            msg = str(e)
            if "No espeak backend" in msg or "espeak" in msg.lower():
                db_print("\n[error] Coqui TTS requires eSpeak-NG available as a command (espeak-ng.exe) on Windows.")
                db_print("Install/verify steps:")
                db_print("  1) Install a 64-bit eSpeak-NG release: https://github.com/espeak-ng/espeak-ng/releases")
                db_print("  2) Make sure the folder containing espeak-ng.exe is on your PATH.")
                db_print("     Common locations: C:\\Program Files\\eSpeak NG or C:\\Program Files\\eSpeak NG\\bin")
                db_print("  3) In PowerShell (current session), you can test by:")
                db_print("       $env:Path = 'C:\\Program Files\\eSpeak NG;' + $env:Path")
                db_print("       espeak-ng --version")
                db_print("  4) Then rerun this agent in the same session.")
                curr = shutil.which("espeak-ng") or shutil.which("espeak")
                db_print(f"[check] which(espeak-ng|espeak) => {curr!r}")
                curr_lib = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
                if curr_lib:
                    db_print(f"[check] PHONEMIZER_ESPEAK_LIBRARY={curr_lib} (exists: {os.path.exists(curr_lib)})")
            raise
    
    elif args.tts_engine == 'chatterbox':
        # Chatterbox TTS initialization (no eSpeak required!)
        try:
            tts = ChatterboxTTS(
                model_name=None,  # Chatterbox doesn't use model_name like Coqui
                multilingual=args.chatterbox_multilingual,
                speaker_wav=args.chatterbox_speaker_wav,
                device=args.chatterbox_device,
                use_gpu=True if args.chatterbox_device != 'cpu' else False,
            )
            db_print(f"[TTS] Chatterbox initialized successfully")
            if args.chatterbox_speaker_wav:
                db_print(f"[TTS] Voice cloning enabled with: {args.chatterbox_speaker_wav}")
        except Exception as e:
            db_print(f"\n[error] Failed to initialize Chatterbox TTS: {e}")
            db_print("Install with: pip install chatterbox-tts")
            raise
    
    else:
        raise ValueError(f"Unknown TTS engine: {args.tts_engine}")

    db_print(f"\nOffline agent ready with {args.tts_engine.upper()} TTS. Press Enter to talk; Ctrl+C to exit.\n")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while True:
            try:
                input("[you] Press Enter to start talking…")
            except (KeyboardInterrupt, EOFError):
                break
            wav = record_push_to_talk(args.sample_rate, channels=1)
            if wav.size == 0:
                db_print("[stt] No audio captured.")
                continue
            text, _ = stt.transcribe(wav, args.sample_rate)
            if not text:
                db_print("[stt] (silence)")
                continue
            db_print(f"[you] {text}")

            reply = loop.run_until_complete(query_ollama(text, model=args.ollama_model, base=args.ollama_base))
            if not reply:
                db_print("[llm] (no reply)")
                continue
            db_print(f"[bot] {reply}")
            audio = tts.synth(reply)
            tts.play(audio)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()