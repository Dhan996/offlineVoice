"""
Enhanced WebSocket server with real-time streaming:
- Streaming STT (faster-whisper with sliding window)
- Streaming LLM (Ollama - already streaming)
- Streaming TTS (sentence-by-sentence)
- Barge-in support
"""
import argparse
import asyncio
import json
import base64
import time
from typing import Optional, Dict, List

import numpy as np
import websockets

from streaming_whisper_service import StreamingWhisperSTT
from streaming_tts_service import StreamingTTS
from llm_ollama_service import OllamaLLM
from vad_service import create_vad

try:
    from config import CONFIG as GLOBAL_CONFIG
except ImportError:
    GLOBAL_CONFIG = {
        "system_prompt": "You are a concise helpful voice assistant. Keep responses brief and conversational.",
        "ollama_model": "llama3:8b",
        "ollama_base": "http://localhost:11434",
        "llm_temperature": 0.6,
        "llm_max_tokens": 256
    }


def db_print(*a):
    print(*a, flush=True)

# --- RESAMPLING HELPERS ---
def _to_mono_f32(pcm16: np.ndarray, channels: int) -> np.ndarray:
    """
    Convert interleaved int16 PCM to mono float32 [-1, 1].
    If channels==1, just scale/return.
    """
    if channels <= 1:
        return (pcm16.astype(np.float32)) / 32768.0
    # De-interleave and average channels
    total_frames = pcm16.size // channels
    pcm16 = pcm16[: total_frames * channels].reshape(total_frames, channels)
    mono = pcm16.mean(axis=1).astype(np.float32) / 32768.0
    return mono

def _resample_f32_to_16k(x: np.ndarray, in_sr: int) -> np.ndarray:
    """
    Resample float32 mono to 16 kHz.
    Tries scipy.signal.resample_poly if available; otherwise uses
    a simple linear-rate numpy fallback (good enough for speech).
    """
    target_sr = 16000
    if in_sr == target_sr or x.size == 0:
        return x

    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(in_sr, target_sr)
        up = target_sr // g
        down = in_sr // g
        return resample_poly(x, up, down).astype(np.float32)
    except Exception:
        # Numpy fallback (linear interpolation)
        import numpy as np
        duration = x.size / float(in_sr)
        new_len = int(round(duration * target_sr))
        if new_len <= 1:  # nothing meaningful to do
            return x
        orig_idx = np.linspace(0.0, 1.0, num=x.size, endpoint=False, dtype=np.float32)
        new_idx = np.linspace(0.0, 1.0, num=new_len, endpoint=False, dtype=np.float32)
        return np.interp(new_idx, orig_idx, x).astype(np.float32)

class StreamingVoiceAgent:
    """Streaming voice agent with real-time STT, LLM, and TTS"""

    def __init__(
        self,
        whisper_model: str = "distil-large-v3",
        whisper_device: str = "cpu",
        compute_type: str = "int8",
        lang: Optional[str] = "en",
        tts_backend: str = "coqui",
        tts_model: str = "tts_models/en/vctk/vits",
        ollama_model: str = "llama3:latest",
        ollama_base: str = "http://localhost:11434",
    ):
        self.ollama_model = ollama_model
        self.ollama_base = ollama_base

        db_print(f"[init] Loading streaming Whisper: {whisper_model}")
        self.stt = StreamingWhisperSTT(
            model_name=whisper_model,
            device=whisper_device,
            compute_type=compute_type,
            language=lang,
            window_duration=2.4,  # 3s windows
            hop_duration=0.3     # Process every 2s
        )

        db_print(f"[init] Loading streaming TTS: {tts_model}")
        self.tts = StreamingTTS(tts_backend=tts_backend, model_name=tts_model)

        db_print("[init] Streaming agent ready")

    async def query_ollama_stream(self, user_text: str):
        """Query Ollama LLM and yield streaming tokens"""
        messages = [
            {"role": "system", "content": GLOBAL_CONFIG.get("system_prompt", "You are a helpful assistant.")},
            {"role": "user", "content": user_text},
        ]

        original_model = GLOBAL_CONFIG.get("ollama_model")
        original_base = GLOBAL_CONFIG.get("ollama_base")
        GLOBAL_CONFIG["ollama_model"] = self.ollama_model
        GLOBAL_CONFIG["ollama_base"] = self.ollama_base

        try:
            start_time = time.time()
            db_print(f"[LLM] Streaming from {self.ollama_model}...")

            llm = OllamaLLM()
            async for token in llm.stream(messages):
                yield token

            elapsed = time.time() - start_time
            db_print(f"[LLM] Stream completed in {elapsed:.2f}s")

        finally:
            GLOBAL_CONFIG["ollama_model"] = original_model
            GLOBAL_CONFIG["ollama_base"] = original_base


class StreamingWebSocketHandler:
    """Handles WebSocket connections with full streaming pipeline"""

    def __init__(self, agent: StreamingVoiceAgent, sample_rate: int = 16000):
        self.agent = agent
        self.sample_rate = sample_rate

        # Per-client state
        self.client_vad_engines: Dict[int, any] = {}  # VAD engines
        self.client_tasks: Dict[int, asyncio.Task] = {}  # Track cancellable tasks
        self.last_partial: Dict[int, str] = {}
    async def handle_client(self, websocket):
        """Handle a single WebSocket client connection"""
        client_id = id(websocket)
        db_print(f"[ws] Client {client_id} connected")

        # Initialize VAD engine with more conservative settings to avoid false positives
        self.client_vad_engines[client_id] = create_vad(
            use_webrtc=True,
            # use_silero=True,  # Use simple VAD for now (faster)
            threshold=0.5,  # Increased from 0.01 to reduce false positives
            min_speech_duration_ms=250,  # Increased from 300ms to filter out noise bursts
            min_silence_duration_ms=600,  # 1 second silence → faster response
            sample_rate=16000
        )

        try:
            await websocket.send(json.dumps({
                "type": "ready",
                "message": "Streaming voice agent ready",
                "sample_rate": self.sample_rate,
                "tts_sample_rate": self.agent.tts.get_sample_rate()
            }))

            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    # Skip logging frequent audio_chunk messages to reduce console spam
                    if msg_type != "audio_chunk":
                        db_print(f"[ws] Client {client_id} msg: {msg_type}")

                    if msg_type == "audio_chunk":
                        await self.handle_streaming_chunk(websocket, client_id, data)
                    elif msg_type == "audio_start":
                        # Acknowledge but ignore - server uses automatic VAD
                        db_print(f"[ws] Audio start signal received (ignored - using automatic VAD)")
                    elif msg_type == "audio_end":
                        await self.handle_stream_finalize(websocket, client_id)
                    elif msg_type == "interrupt":
                        await self.handle_interrupt(websocket, client_id)
                    elif msg_type == "text":
                        await self.handle_text_streaming(websocket, data)
                    else:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"Unknown message type: {msg_type}"
                        }))

                except json.JSONDecodeError as e:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Invalid JSON: {str(e)}"
                    }))
                except Exception as e:
                    db_print(f"[ws] Error processing message: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))

        except websockets.exceptions.ConnectionClosed:
            db_print(f"[ws] Client {client_id} disconnected")
        except Exception as e:
            db_print(f"[ws] Error handling client {client_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            if client_id in self.client_vad_engines:
                del self.client_vad_engines[client_id]
            if client_id in self.client_tasks:
                self.client_tasks[client_id].cancel()
                del self.client_tasks[client_id]
            db_print(f"[ws] Client {client_id} connection closed")

    async def handle_streaming_chunk(self, websocket, client_id: int, data: Dict):
        """Process streaming audio chunk with Backend VAD + partial transcripts"""
        try:
            # Decode audio
            audio_b64 = data.get("audio")
            if not audio_b64:
                return

            audio_bytes = base64.b64decode(audio_b64)
            # --- NEW: detect client-provided metadata (optional) ---
            # Expecting the client to send {"type":"audio_chunk","audio":"...","sample_rate":16000,"channels":1}
            in_sr = int(data.get("sample_rate", self.sample_rate))  # default to server's expected rate (16k)
            channels = int(data.get("channels", 1))

            # Convert bytes -> int16 array
            pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)

            # Convert to mono float32 [-1,1]
            audio_mono_f32 = _to_mono_f32(pcm16, channels)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Get VAD engine
            vad_engine = self.client_vad_engines.get(client_id)

            if not vad_engine:
                db_print(f"[ws] No VAD engine for client {client_id}")
                return

            # Process through VAD first
            vad_result = vad_engine.process_chunk(audio_array)

            # Notify client of speech state changes
            if vad_result['speech_started']:
                await websocket.send(json.dumps({
                    "type": "speech_started",
                    "message": "Speech detected"
                }))

            # Only process STT if speech is happening (use shared agent STT)
            if vad_result['is_speaking']:
                result = self.agent.stt.add_audio_chunk(audio_array)

                if result and result['text'].strip():
                    db_print(f"[ws] Partial [{result['processing_time_ms']:.0f}ms]: {result['text'][:60]}...")
                    text = result['text'].strip()
                    self.last_partial[client_id] = text
                    await websocket.send(json.dumps({
                        "type": "partial_transcript",
                        "text": result['text'],
                        "processing_time_ms": result['processing_time_ms']
                    }))

            # Auto-finalize when speech ends (Backend VAD magic!)
            if vad_result['speech_ended']:
                db_print(f"[ws] Backend VAD detected speech end, auto-finalizing...")
                await self.handle_stream_finalize(websocket, client_id)

        except Exception as e:
            db_print(f"[ws] Error in streaming chunk: {e}")
            import traceback
            traceback.print_exc()

    async def handle_stream_finalize(self, websocket, client_id: int):
        """Finalize STT and start LLM → TTS pipeline"""
        try:
            db_print(f"[ws] Finalizing stream for client {client_id}")

            # Get final transcript from shared agent STT
            final_result = self.agent.stt.finalize()
            # final_text = final_result['text']
            final_text = (final_result.get('text') or "").strip()
            db_print(f"[ws] Final transcript: '{final_text}'")

            if not final_text:
                final_text = self.last_partial.get(client_id, "").strip()
                # await websocket.send(json.dumps({
                #     "type": "transcript",
                #     "text": "",
                #     "message": "No speech detected"
                # }))
                # await websocket.send(json.dumps({"type": "status", "status": "complete"}))
                # return
            db_print(f"[ws] Final transcript: '{final_text}'")

            if not final_text:
                await websocket.send(json.dumps({
                    "type": "transcript",
                    "text": "",
                    "message": "No speech detected"
                }))
                await websocket.send(json.dumps({"type": "status", "status": "complete"}))
                return
            # Send final transcript
            await websocket.send(json.dumps({
                "type": "transcript",
                "text": final_text,
                "is_final": True
            }))

            # Start LLM → TTS streaming pipeline
            await websocket.send(json.dumps({"type": "status", "status": "processing"}))

            # Create cancellable task
            task = asyncio.create_task(
                self._stream_llm_and_tts(websocket, final_text)
            )
            self.client_tasks[client_id] = task

            await task

        except asyncio.CancelledError:
            db_print(f"[ws] Stream pipeline cancelled for client {client_id}")
            await websocket.send(json.dumps({"type": "interrupted"}))
        except Exception as e:
            db_print(f"[ws] Error in finalize: {e}")
            import traceback
            traceback.print_exc()
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Pipeline error: {str(e)}"
            }))

    async def _stream_llm_and_tts(self, websocket, user_text: str):
        """Stream LLM response and synthesize TTS sentence-by-sentence"""
        try:
            # Get LLM stream
            llm_stream = self.agent.query_ollama_stream(user_text)

            # Stream TTS from LLM
            sentence_buffer = ""
            full_response = ""
            tts_started = False

            await websocket.send(json.dumps({"type": "status", "status": "synthesizing"}))

            async for chunk in self.agent.tts.stream_from_llm(llm_stream):
                # Send tts_start on first chunk
                if not tts_started:
                    await websocket.send(json.dumps({
                        "type": "tts_start",
                        "sample_rate": chunk['sample_rate'],
                        "dtype": "float32"
                    }))
                    tts_started = True

                # Send TTS audio chunk
                audio_b64 = base64.b64encode(chunk['audio'].tobytes()).decode('utf-8')

                await websocket.send(json.dumps({
                    "type": "tts_chunk",
                    "audio": audio_b64,
                    "sample_rate": chunk['sample_rate'],
                    "dtype": "float32",
                    "text": chunk['text']
                }))

                full_response += chunk['text'] + " "

            # Send tts_end to signal completion
            if tts_started:
                await websocket.send(json.dumps({"type": "tts_end"}))

            # Send full response text
            await websocket.send(json.dumps({
                "type": "response",
                "text": full_response.strip()
            }))

            await websocket.send(json.dumps({"type": "status", "status": "complete"}))

        except asyncio.CancelledError:
            raise  # Propagate cancellation
        except Exception as e:
            db_print(f"[ws] Error in LLM/TTS stream: {e}")
            raise

    async def handle_interrupt(self, websocket, client_id: int):
        """Handle barge-in: cancel ongoing TTS/LLM"""
        db_print(f"[ws] Interrupt requested for client {client_id}")

        if client_id in self.client_tasks:
            self.client_tasks[client_id].cancel()
            del self.client_tasks[client_id]

        # Reset shared STT engine
        self.agent.stt.reset()

        await websocket.send(json.dumps({"type": "interrupted"}))

    async def handle_text_streaming(self, websocket, data: Dict):
        """Handle direct text input with streaming TTS"""
        text = data.get("text", "").strip()
        if not text:
            return

        db_print(f"[ws] Text input: {text}")

        await websocket.send(json.dumps({"type": "status", "status": "processing"}))

        # Stream response
        async for chunk in self.agent.tts.stream_from_llm(
            self.agent.query_ollama_stream(text)
        ):
            audio_b64 = base64.b64encode(chunk['audio'].tobytes()).decode('utf-8')

            await websocket.send(json.dumps({
                "type": "audio_chunk",
                "audio": audio_b64,
                "sample_rate": chunk['sample_rate'],
                "dtype": "float32"
            }))

        await websocket.send(json.dumps({"type": "status", "status": "complete"}))


async def main_server(host: str, port: int, agent: StreamingVoiceAgent, sample_rate: int):
    """Start WebSocket server"""
    handler = StreamingWebSocketHandler(agent, sample_rate)

    db_print(f"[server] Starting streaming WebSocket server on {host}:{port}")

    async with websockets.serve(handler.handle_client, host, port, max_size=10*1024*1024):
        db_print(f"[server] WebSocket server listening on ws://{host}:{port}")
        await asyncio.Future()  # Run forever


def main():
    p = argparse.ArgumentParser(description="Streaming voice agent with faster-whisper + Ollama + TTS")
    p.add_argument('--host', default='0.0.0.0', help='WebSocket server host')
    p.add_argument('--port', type=int, default=9095, help='WebSocket server port (default: 9095)')
    p.add_argument('--ollama-base', default='http://localhost:11434')
    p.add_argument('--ollama-model', default='llama3:latest')
    p.add_argument('--whisper-model', default='large-v3', help='Whisper model (large-v3, distil-large-v3, etc.)')
    p.add_argument('--whisper-device', default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--compute-type', default='float16', help='int8 (CPU), float16 (GPU)')
    p.add_argument('--lang', default='en', help='Language (en, es, fr, etc.)')
    p.add_argument('--tts-backend', default='coqui', choices=['coqui', 'piper'])
    p.add_argument('--tts-model', default='tts_models/en/vctk/vits')
    p.add_argument('--sample-rate', type=int, default=16000, help='Audio sample rate')
    args = p.parse_args()

    # Initialize agent
    agent = StreamingVoiceAgent(
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        compute_type=args.compute_type,
        lang=args.lang,
        tts_backend=args.tts_backend,
        tts_model=args.tts_model,
        ollama_model=args.ollama_model,
        ollama_base=args.ollama_base
    )

    # Start WebSocket server
    try:
        asyncio.run(main_server(args.host, args.port, agent, args.sample_rate))
    except KeyboardInterrupt:
        db_print("\n[server] Shutting down...")


if __name__ == '__main__':
    main()
