#!/usr/bin/env python3
"""
Mac client for testing the WebSocket voice agent
Captures audio from mic, sends to server, plays TTS responses
"""
import argparse
import asyncio
import base64
import json
import threading
import queue
import sys
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets

# For keyboard input detection
try:
    import termios
    import tty
    UNIX_TERMINAL = True
except ImportError:
    UNIX_TERMINAL = False

def db_print(*args):
    """Debug print with flush"""
    print(*args, flush=True)


class AudioCapture:
    """Handles microphone audio capture"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.stream = None
        
    def start_recording(self):
        """Start capturing audio from microphone"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.audio_queue = queue.Queue()
        
        def callback(indata, frames, time, status):
            if status:
                print(f"[AudioCapture] Status: {status}")
            if self.is_recording:
                # Convert to mono if needed
                if indata.shape[1] > 1:
                    audio = np.mean(indata, axis=1)
                else:
                    audio = indata[:, 0]
                self.audio_queue.put(audio.copy())
        
        self.stream = sd.InputStream(
            callback=callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            device=self.device,
            dtype='int16'
        )
        self.stream.start()
        db_print("[AudioCapture] Recording started")
    
    def stop_recording(self):
        """Stop audio capture"""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        db_print("[AudioCapture] Recording stopped")
    
    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get audio chunk from queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class AudioPlayer:
    """Handles audio playback"""
    
    def __init__(self, device: Optional[int] = None):
        self.device = device
        self.playback_queue = queue.Queue()
        self.is_playing = False
        
    def play_audio(self, audio: np.ndarray, sample_rate: int):
        """Add audio to playback queue"""
        self.playback_queue.put((audio, sample_rate))
        
    def start_playback_thread(self):
        """Start background thread for audio playback"""
        def playback_worker():
            while True:
                try:
                    audio, sample_rate = self.playback_queue.get(timeout=0.1)
                    self.is_playing = True
                    sd.play(audio, samplerate=sample_rate, device=self.device, blocking=True)
                    self.is_playing = False
                except queue.Empty:
                    continue
                except Exception as e:
                    db_print(f"[AudioPlayer] Error: {e}")
                    self.is_playing = False
        
        thread = threading.Thread(target=playback_worker, daemon=True)
        thread.start()
        db_print("[AudioPlayer] Playback thread started")


class WebSocketClient:
    """WebSocket client for voice agent"""
    
    def __init__(self, url: str, audio_capture: AudioCapture, audio_player: AudioPlayer):
        self.url = url
        self.audio_capture = audio_capture
        self.audio_player = audio_player
        self.websocket = None
        self.is_connected = False
        self.tts_buffer = []
        self.tts_sample_rate = None
        self.is_receiving_tts = False
        
    async def connect(self):
        """Connect to WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.url)
            self.is_connected = True
            db_print(f"[WebSocket] Connected to {self.url}")
            
            # Start receiving messages
            asyncio.create_task(self.receive_loop())
            
        except Exception as e:
            db_print(f"[WebSocket] Connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from server"""
        self.is_connected = False
        if self.websocket:
            await self.websocket.close()
            db_print("[WebSocket] Disconnected")
    
    async def receive_loop(self):
        """Receive and handle messages from server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "ready":
                    db_print(f"[Server] Ready: {data.get('message')}")
                    db_print(f"[Server] STT sample rate: {data.get('sample_rate')}Hz")
                    db_print(f"[Server] TTS sample rate: {data.get('tts_sample_rate')}Hz")
                    
                elif msg_type == "speech_started":
                    db_print("[VAD] Speech detected")
                    
                elif msg_type == "partial_transcript":
                    text = data.get("text", "")
                    print(f"\r[Partial] {text}", end="")
                    
                elif msg_type == "transcript":
                    text = data.get("text", "")
                    print(f"\r[You] {text}                    ")
                    
                elif msg_type == "response":
                    text = data.get("text", "")
                    db_print(f"[Agent] {text}")
                    
                elif msg_type == "tts_start":
                    self.is_receiving_tts = True
                    self.tts_sample_rate = data.get("sample_rate", 22050)
                    self.tts_buffer = []
                    db_print(f"[TTS] Starting playback @ {self.tts_sample_rate}Hz")
                    
                elif msg_type == "tts_chunk":
                    if self.is_receiving_tts:
                        # Decode and play audio
                        audio_b64 = data.get("audio")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            audio = np.frombuffer(audio_bytes, dtype=np.float32)
                            
                            # Play immediately for low latency
                            self.audio_player.play_audio(audio, self.tts_sample_rate)
                            
                elif msg_type == "tts_end":
                    self.is_receiving_tts = False
                    db_print("[TTS] Playback complete")
                    
                elif msg_type == "status":
                    status = data.get("status", "")
                    if status and status != "complete":
                        db_print(f"[Status] {status}")
                    
                elif msg_type == "error":
                    db_print(f"[Error] {data.get('message', 'Unknown error')}")
                    
        except websockets.exceptions.ConnectionClosed:
            db_print("[WebSocket] Connection closed by server")
            self.is_connected = False
        except Exception as e:
            db_print(f"[WebSocket] Receive error: {e}")
            self.is_connected = False
    
    async def send_audio_stream(self):
        """Stream audio chunks to server"""
        if not self.is_connected:
            return
            
        db_print("[Stream] Starting audio stream to server...")
        chunk_duration = 0.1  # Send chunks every 100ms
        
        while self.audio_capture.is_recording and self.is_connected:
            # Collect audio chunks
            chunks = []
            deadline = time.time() + chunk_duration
            
            while time.time() < deadline:
                chunk = self.audio_capture.get_audio_chunk(timeout=0.01)
                if chunk is not None:
                    chunks.append(chunk)
            
            if chunks:
                # Combine chunks
                audio = np.concatenate(chunks)
                
                # Convert to base64
                audio_b64 = base64.b64encode(audio.tobytes()).decode('utf-8')
                
                # Send to server
                await self.websocket.send(json.dumps({
                    "type": "audio_chunk",
                    "audio": audio_b64
                }))
            
            await asyncio.sleep(0.01)
        
        # Send audio_end when recording stops
        if self.is_connected:
            await self.websocket.send(json.dumps({"type": "audio_end"}))
            db_print("[Stream] Audio stream ended")
    
    async def send_text(self, text: str):
        """Send text message to server"""
        if not self.is_connected:
            return
            
        await self.websocket.send(json.dumps({
            "type": "text",
            "text": text
        }))
    
    async def interrupt(self):
        """Send interrupt signal"""
        if not self.is_connected:
            return
            
        await self.websocket.send(json.dumps({"type": "interrupt"}))
        db_print("[Interrupt] Sent interrupt signal")


def list_audio_devices():
    """List available audio devices"""
    db_print("\n=== Audio Devices ===")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        db_print(f"{i:3d}: {device['name']} (in:{device['max_input_channels']}, out:{device['max_output_channels']})")
    db_print("")


async def main():
    parser = argparse.ArgumentParser(description="Mac client for WebSocket voice agent")
    parser.add_argument('--server', default='localhost', help='WebSocket server address')
    parser.add_argument('--port', type=int, default=9095, help='WebSocket server port')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--input-device', type=int, default=None, help='Input device index')
    parser.add_argument('--output-device', type=int, default=None, help='Output device index')
    parser.add_argument('--list-devices', action='store_true', help='List audio devices and exit')
    parser.add_argument('--push-to-talk', action='store_true', help='Use push-to-talk mode (hold SPACE)')
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    # Initialize audio components
    audio_capture = AudioCapture(
        sample_rate=args.sample_rate,
        device=args.input_device
    )
    
    audio_player = AudioPlayer(device=args.output_device)
    audio_player.start_playback_thread()
    
    # Connect to WebSocket server
    url = f"ws://{args.server}:{args.port}"
    client = WebSocketClient(url, audio_capture, audio_player)
    
    try:
        await client.connect()
    except Exception as e:
        db_print(f"Failed to connect: {e}")
        return
    
    db_print("\n=== Voice Agent Client ===")
    db_print("Commands:")
    db_print("  SPACE: Start/stop recording (or hold for push-to-talk)")
    db_print("  'i':   Interrupt current response")
    db_print("  't':   Type text input")
    db_print("  'q':   Quit")
    db_print("")
    
    is_recording = False
    
    try:
        while client.is_connected:
            # Simple input handling
            command = input("Press ENTER to start recording (or 'q' to quit): ").strip().lower()
            
            if command == 'q':
                break
                
            elif command == 'i':
                await client.interrupt()
                
            elif command == 't':
                text = input("Enter text: ").strip()
                if text:
                    await client.send_text(text)
                    
            else:  # ENTER pressed - toggle recording
                if not is_recording:
                    db_print("\n🎤 Recording... (press ENTER to stop)")
                    audio_capture.start_recording()
                    is_recording = True
                    
                    # Start streaming audio
                    stream_task = asyncio.create_task(client.send_audio_stream())
                else:
                    db_print("⏹ Stopping recording...")
                    audio_capture.stop_recording()
                    is_recording = False
                    
                    # Wait for stream task to complete
                    try:
                        await stream_task
                    except:
                        pass
            
            # Give async tasks time to run
            await asyncio.sleep(0.1)
            
    except KeyboardInterrupt:
        db_print("\n[Client] Interrupted")
    finally:
        # Clean up
        if is_recording:
            audio_capture.stop_recording()
        
        await client.disconnect()
        db_print("[Client] Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")