"""
Enhanced WebSocket server using StreamingWhisperService1:
- VAD-based utterance segmentation with immediate barge-in
- Whisper medium for high accuracy
- Streaming LLM (Ollama)
- Streaming TTS (sentence-by-sentence)
- 16kHz float32 audio input
"""
import argparse
import asyncio
import json
import base64
import time
from typing import Optional, Dict

import numpy as np
import websockets

from streaming_whisper_service_1 import StreamingWhisperService1
from streaming_tts_service import StreamingTTS
from llm_ollama_service import OllamaLLM

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


class StreamingVoiceAgent1:
    """Enhanced streaming voice agent with VAD-based STT"""

    def __init__(
        self,
        whisper_model: str = "medium",
        whisper_device: str = "cuda",
        compute_type: str = "float16",
        lang: Optional[str] = "en",
        tts_backend: str = "coqui",
        tts_model: str = "tts_models/en/vctk/vits",
        ollama_model: str = "llama3:latest",
        ollama_base: str = "http://localhost:11434",
        silence_threshold: float = 0.02,
        silence_duration_s: float = 0.8,
        voice_frames_needed: int = 2,
    ):
        self.ollama_model = ollama_model
        self.ollama_base = ollama_base

        db_print(f"[init] Loading StreamingWhisperService1 with model: {whisper_model}")
        self.stt = StreamingWhisperService1(
            model_name=whisper_model,
            language=lang,
            sample_rate=16000,
            device=whisper_device,
            compute_type=compute_type,
            silence_threshold=silence_threshold,
            silence_duration_s=silence_duration_s,
            voice_frames_needed=voice_frames_needed
        )

        db_print(f"[init] Loading streaming TTS: {tts_model}")
        self.tts = StreamingTTS(tts_backend=tts_backend, model_name=tts_model)

        # Will be initialized when service starts
        self.stt_started = False

        db_print("[init] Streaming agent ready")

    async def start_stt(self):
        """Start the STT service"""
        if not self.stt_started:
            await self.stt.start()
            self.stt_started = True
            # STT started - reduced verbosity

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


class StreamingWebSocketHandler1:
    """Handles WebSocket connections with VAD-based STT pipeline"""

    def __init__(self, agent: StreamingVoiceAgent1, sample_rate: int = 16000):
        self.agent = agent
        self.sample_rate = sample_rate

        # Per-client state
        self.client_stt_services: Dict[int, StreamingWhisperService1] = {}  # One STT per client
        self.client_tasks: Dict[int, Dict[str, asyncio.Task]] = {}  # Multiple tasks per client
        self.client_transcripts: Dict[int, str] = {}  # Store final transcripts

    async def handle_client(self, websocket):
        """Handle a single WebSocket client connection"""
        client_id = id(websocket)
        db_print(f"[ws] Client {client_id} connected")

        # Create dedicated STT service for this client (reduced log verbosity)
        client_stt = StreamingWhisperService1(
            model_name=self.agent.stt.model._model_path if hasattr(self.agent.stt.model, '_model_path') else "medium",
            language=self.agent.stt.language,
            sample_rate=self.agent.stt.sample_rate,
            device=self.agent.stt.device,
            compute_type=self.agent.stt.compute_type,
            silence_threshold=self.agent.stt._silence_threshold,
            silence_duration_s=self.agent.stt._silence_duration_s,
            voice_frames_needed=self.agent.stt._voice_frames_needed
        )
        await client_stt.start()
        self.client_stt_services[client_id] = client_stt

        # Initialize client state
        self.client_tasks[client_id] = {}
        self.client_transcripts[client_id] = ""

        try:
            await websocket.send(json.dumps({
                "type": "ready",
                "message": "Streaming voice agent ready (VAD-based STT with automatic barge-in)",
                "sample_rate": self.sample_rate,
                "tts_sample_rate": self.agent.tts.get_sample_rate(),
                "stt_service": "StreamingWhisperService1",
                "features": ["automatic_vad", "barge_in", "utterance_segmentation"]
            }))

            # Start STT result reader task
            stt_reader_task = asyncio.create_task(
                self._stt_result_reader(websocket, client_id)
            )
            self.client_tasks[client_id]['stt_reader'] = stt_reader_task

            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type != "audio_chunk":
                        db_print(f"[ws] Client {client_id} msg: {msg_type}")

                    if msg_type == "audio_chunk":
                        await self.handle_audio_chunk(websocket, client_id, data)
                    elif msg_type == "audio_start":
                        # Acknowledge but ignore - server uses automatic VAD
                        db_print(f"[ws] Audio start signal received (ignored - using automatic VAD)")
                    elif msg_type == "audio_end":
                        # Manual finalization if needed (STT auto-finalizes via VAD)
                        db_print(f"[ws] Audio end signal received for client {client_id}")
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
            if client_id in self.client_tasks:
                for task in self.client_tasks[client_id].values():
                    if not task.done():
                        task.cancel()
                del self.client_tasks[client_id]
            if client_id in self.client_transcripts:
                del self.client_transcripts[client_id]
            if client_id in self.client_stt_services:
                # Stop and cleanup client's STT service
                await self.client_stt_services[client_id].stop()
                del self.client_stt_services[client_id]
            db_print(f"[ws] Client {client_id} disconnected")

    async def _stt_result_reader(self, websocket, client_id: int):
        """Read STT results from the client's dedicated transcript queue"""
        try:
            # Get client's dedicated STT service
            client_stt = self.client_stt_services.get(client_id)
            if not client_stt:
                return

            while True:
                # Get result from CLIENT'S STT service queue
                is_final, text, metadata = await client_stt.transcript_queue.get()

                # Check if service stopped
                if text is None:
                    break

                result_type = metadata.get('type', 'unknown')

                if result_type == 'barge_in':
                    # Voice detected - immediate barge-in signal
                    # Removed verbose log to reduce clutter
                    await websocket.send(json.dumps({
                        "type": "barge_in",
                        "message": "Voice detected - interrupting playback"
                    }))

                    # Cancel any ongoing TTS/LLM
                    await self.handle_interrupt(websocket, client_id, send_response=False)

                elif result_type == 'final':
                    # Final transcript ready
                    proc_time = metadata.get('processing_time_ms', 0)
                    duration = metadata.get('audio_duration_s', 0)
                    language = metadata.get('language', 'unknown')

                    db_print(f"[ws] Final transcript ({proc_time:.0f}ms, {duration:.1f}s, {language}): {text}")

                    # Store transcript
                    self.client_transcripts[client_id] = text

                    # Send final transcript
                    await websocket.send(json.dumps({
                        "type": "transcript",
                        "text": text,
                        "is_final": True,
                        "metadata": {
                            "processing_time_ms": proc_time,
                            "audio_duration_s": duration,
                            "language": language,
                            "language_probability": metadata.get('language_probability')
                        }
                    }))

                    # Start LLM → TTS pipeline
                    if text.strip():
                        await websocket.send(json.dumps({"type": "status", "status": "processing"}))

                        llm_tts_task = asyncio.create_task(
                            self._stream_llm_and_tts(websocket, client_id, text)
                        )
                        self.client_tasks[client_id]['llm_tts'] = llm_tts_task

                elif result_type == 'error':
                    # Error occurred
                    db_print(f"[ws] STT error for client {client_id}: {text}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"STT error: {text}"
                    }))

        except asyncio.CancelledError:
            pass  # Expected on disconnect
        except Exception as e:
            db_print(f"[ws] Error in STT result reader: {e}")
            import traceback
            traceback.print_exc()

    async def handle_audio_chunk(self, websocket, client_id: int, data: Dict):
        """Process incoming audio chunk - STT service handles VAD internally"""
        try:
            # Decode audio
            audio_b64 = data.get("audio")
            if not audio_b64:
                return

            audio_bytes = base64.b64decode(audio_b64)

            # Convert int16 PCM to float32 in [-1, 1]
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Debug: Check audio levels (only log occasionally to avoid spam)
            if not hasattr(self, '_audio_chunk_count'):
                self._audio_chunk_count = {}
            self._audio_chunk_count[client_id] = self._audio_chunk_count.get(client_id, 0) + 1

            # Log every 50th chunk to monitor audio levels
            if self._audio_chunk_count[client_id] % 50 == 1:
                rms = np.sqrt(np.mean(audio_array ** 2))
                db_print(f"[ws] Audio RMS: {rms:.4f} (threshold: 0.03, samples: {len(audio_array)})")

            # Get client's dedicated STT service
            client_stt = self.client_stt_services.get(client_id)
            if client_stt:
                # Feed to CLIENT'S STT - it has built-in VAD for speech detection,
                # utterance segmentation, and automatic barge-in signal generation
                client_stt.add_audio_chunk(audio_array)
            else:
                db_print(f"[ws] Warning: No STT service for client {client_id}")

        except Exception as e:
            db_print(f"[ws] Error in audio chunk: {e}")
            import traceback
            traceback.print_exc()

    async def _stream_llm_and_tts(self, websocket, client_id: int, user_text: str):
        """Stream LLM response and synthesize TTS sentence-by-sentence"""
        try:
            # Get LLM stream
            llm_stream = self.agent.query_ollama_stream(user_text)

            # Stream TTS from LLM
            tts_started = False
            full_response = ""

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
            db_print(f"[ws] LLM/TTS pipeline cancelled for client {client_id}")
            await websocket.send(json.dumps({"type": "interrupted"}))
        except Exception as e:
            db_print(f"[ws] Error in LLM/TTS stream: {e}")
            import traceback
            traceback.print_exc()
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Pipeline error: {str(e)}"
            }))

    async def handle_interrupt(self, websocket, client_id: int, send_response: bool = True):
        """Handle barge-in: cancel ongoing TTS/LLM"""
        # Removed verbose log to reduce clutter

        # Cancel LLM/TTS task if running
        if client_id in self.client_tasks:
            if 'llm_tts' in self.client_tasks[client_id]:
                task = self.client_tasks[client_id]['llm_tts']
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                del self.client_tasks[client_id]['llm_tts']

        # Reset CLIENT'S STT engine
        client_stt = self.client_stt_services.get(client_id)
        if client_stt:
            client_stt.reset()

        if send_response:
            await websocket.send(json.dumps({"type": "interrupted"}))

    async def handle_text_streaming(self, websocket, data: Dict):
        """Handle direct text input with streaming TTS"""
        text = data.get("text", "").strip()
        if not text:
            return

        db_print(f"[ws] Text input: {text}")

        await websocket.send(json.dumps({"type": "status", "status": "processing"}))

        # Stream response
        tts_started = False
        async for chunk in self.agent.tts.stream_from_llm(
            self.agent.query_ollama_stream(text)
        ):
            if not tts_started:
                await websocket.send(json.dumps({
                    "type": "tts_start",
                    "sample_rate": chunk['sample_rate'],
                    "dtype": "float32"
                }))
                tts_started = True

            audio_b64 = base64.b64encode(chunk['audio'].tobytes()).decode('utf-8')

            await websocket.send(json.dumps({
                "type": "tts_chunk",
                "audio": audio_b64,
                "sample_rate": chunk['sample_rate'],
                "dtype": "float32",
                "text": chunk.get('text', '')
            }))

        if tts_started:
            await websocket.send(json.dumps({"type": "tts_end"}))

        await websocket.send(json.dumps({"type": "status", "status": "complete"}))


async def main_server(host: str, port: int, agent: StreamingVoiceAgent1, sample_rate: int):
    """Start WebSocket server"""
    handler = StreamingWebSocketHandler1(agent, sample_rate)

    db_print(f"[server] Starting streaming WebSocket server on {host}:{port}")
    db_print(f"[server] Using StreamingWhisperService1 with VAD-based segmentation")

    async with websockets.serve(handler.handle_client, host, port, max_size=10*1024*1024):
        db_print(f"[server] WebSocket server listening on ws://{host}:{port}")
        await asyncio.Future()  # Run forever


def main():
    p = argparse.ArgumentParser(description="Streaming voice agent with StreamingWhisperService1 + Ollama + TTS")
    p.add_argument('--host', default='0.0.0.0', help='WebSocket server host')
    p.add_argument('--port', type=int, default=9095, help='WebSocket server port (default: 9095)')
    p.add_argument('--ollama-base', default='http://localhost:11434')
    p.add_argument('--ollama-model', default='llama3:latest')
    p.add_argument('--whisper-model', default='medium',
                   help='Whisper model name (medium = best quality, medium = balanced, small = fastest)')
    p.add_argument('--whisper-device', default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--compute-type', default='float16', help='float16 (GPU, best quality), int8 (CPU, faster)')
    p.add_argument('--lang', default='en', help='Language (en, es, fr, etc.) - en gives best results for English')
    p.add_argument('--tts-backend', default='coqui', choices=['coqui', 'piper'])
    p.add_argument('--tts-model', default='tts_models/en/vctk/vits')
    p.add_argument('--sample-rate', type=int, default=16000, help='Audio sample rate (16000 recommended)')
    p.add_argument('--silence-threshold', type=float, default=0.02,
                   help='VAD silence threshold (0.02 = balanced, lower = more sensitive)')
    p.add_argument('--silence-duration', type=float, default=0.8,
                   help='Seconds of silence before finalizing (0.8 = responsive, 1.5 = conservative)')
    p.add_argument('--voice-frames-needed', type=int, default=2,
                   help='Consecutive voice frames before barge-in (2 = 200ms, faster response)')
    args = p.parse_args()

    # Initialize agent
    agent = StreamingVoiceAgent1(
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        compute_type=args.compute_type,
        lang=args.lang,
        tts_backend=args.tts_backend,
        tts_model=args.tts_model,
        ollama_model=args.ollama_model,
        ollama_base=args.ollama_base,
        silence_threshold=args.silence_threshold,
        silence_duration_s=args.silence_duration,
        voice_frames_needed=args.voice_frames_needed
    )

    # Start WebSocket server
    try:
        asyncio.run(main_server(args.host, args.port, agent, args.sample_rate))
    except KeyboardInterrupt:
        db_print("\n[server] Shutting down...")


if __name__ == '__main__':
    main()
