"""
Streaming TTS using sentence-by-sentence synthesis
Works with both Piper (fast) and Coqui TTS (existing)
"""
import re
import numpy as np
from typing import Iterator, Optional
import asyncio


class StreamingTTS:
    """
    Wrapper for sentence-level TTS streaming

    Takes streaming text input (from LLM) and yields audio chunks
    as soon as each sentence is complete.
    """

    def __init__(self, tts_backend="coqui", model_name: str = "tts_models/en/vctk/vits"):
        """
        Args:
            tts_backend: "coqui" or "piper"
            model_name: Model name for chosen backend
        """
        self.backend = tts_backend
        self.sentence_buffer = ""

        if tts_backend == "coqui":
            from tts_coqui_service import CoquiTTS
            print(f"[StreamingTTS] Using Coqui TTS: {model_name}")
            self.tts_engine = CoquiTTS(model_name=model_name)

        elif tts_backend == "piper":
            try:
                import piper
                print(f"[StreamingTTS] Using Piper TTS: {model_name}")
                # Piper setup (you'll need to adjust based on piper API)
                self.tts_engine = self._init_piper(model_name)
            except ImportError:
                print("[StreamingTTS] Piper not installed, falling back to Coqui")
                from tts_coqui_service import CoquiTTS
                self.tts_engine = CoquiTTS(model_name="tts_models/en/vctk/vits")
                self.backend = "coqui"

        print("[StreamingTTS] Streaming TTS initialized")

    def _init_piper(self, model_name):
        """Initialize Piper TTS (placeholder - adjust based on your setup)"""
        # TODO: Implement Piper initialization
        # from piper import PiperVoice
        # return PiperVoice.load(model_name)
        raise NotImplementedError("Piper integration pending")

    def synthesize_chunk(self, text_chunk: str) -> np.ndarray:
        """Synthesize single text chunk to audio"""
        if not text_chunk or not text_chunk.strip():
            return np.zeros((0,), dtype=np.float32)

        if self.backend == "coqui":
            return self.tts_engine.synth(text_chunk)
        elif self.backend == "piper":
            # TODO: Implement Piper synthesis
            return self.tts_engine.synthesize(text_chunk)

    def get_sample_rate(self) -> int:
        """Get TTS output sample rate"""
        if self.backend == "coqui":
            return self.tts_engine.out_rate
        elif self.backend == "piper":
            return 22050  # Piper default
        return 22050

    async def stream_from_llm(
        self,
        llm_token_stream,
        yield_partial: bool = False
    ) -> Iterator[dict]:
        """
        Stream TTS audio from LLM token stream

        Args:
            llm_token_stream: Async iterator yielding text tokens
            yield_partial: If True, yield audio for incomplete sentences

        Yields:
            dict with:
                - 'audio': np.ndarray of audio samples
                - 'sample_rate': int
                - 'text': str (the text that was synthesized)
                - 'is_sentence_end': bool
        """
        sentence_buffer = ""

        # Sentence end patterns
        sentence_end_pattern = re.compile(r'[.!?]+\s*')

        async for token in llm_token_stream:
            sentence_buffer += token

            # Check for sentence end
            if sentence_end_pattern.search(sentence_buffer):
                # Split into sentences
                sentences = sentence_end_pattern.split(sentence_buffer)

                # Process all complete sentences
                for i, sentence in enumerate(sentences[:-1]):  # Skip last (incomplete)
                    sentence = sentence.strip()
                    if sentence:
                        print(f"[StreamingTTS] Synthesizing: '{sentence}'")

                        # Synthesize
                        audio = self.synthesize_chunk(sentence)

                        if audio.size > 0:
                            yield {
                                'audio': audio,
                                'sample_rate': self.get_sample_rate(),
                                'text': sentence,
                                'is_sentence_end': True
                            }

                # Keep last incomplete part
                sentence_buffer = sentences[-1] if sentences else ""

        # Process remaining text
        if sentence_buffer.strip():
            print(f"[StreamingTTS] Synthesizing final: '{sentence_buffer.strip()}'")
            audio = self.synthesize_chunk(sentence_buffer.strip())

            if audio.size > 0:
                yield {
                    'audio': audio,
                    'sample_rate': self.get_sample_rate(),
                    'text': sentence_buffer.strip(),
                    'is_sentence_end': False
                }

    def stream_from_text(self, text: str) -> Iterator[dict]:
        """
        Stream TTS audio from complete text (split by sentences)

        Args:
            text: Complete text to synthesize

        Yields:
            dict with audio chunks
        """
        # Split into sentences
        sentence_pattern = re.compile(r'([.!?]+)')
        parts = sentence_pattern.split(text)

        # Reconstruct sentences with punctuation
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentence = (parts[i] + (parts[i + 1] if i + 1 < len(parts) else '')).strip()
            if sentence:
                sentences.append(sentence)

        # Add last part if no punctuation
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())

        # Synthesize each sentence
        for i, sentence in enumerate(sentences):
            print(f"[StreamingTTS] Synthesizing sentence {i+1}/{len(sentences)}: '{sentence[:50]}...'")

            audio = self.synthesize_chunk(sentence)

            if audio.size > 0:
                yield {
                    'audio': audio,
                    'sample_rate': self.get_sample_rate(),
                    'text': sentence,
                    'is_sentence_end': i < len(sentences) - 1
                }


# Example usage
if __name__ == "__main__":
    # Test with complete text
    tts = StreamingTTS(tts_backend="coqui")

    test_text = "Hello! This is a test of streaming TTS. It should synthesize sentence by sentence. Much faster than waiting for the whole response."

    print("\n=== Streaming from complete text ===\n")
    for chunk in tts.stream_from_text(test_text):
        print(f"Audio chunk: {chunk['audio'].shape[0]} samples, '{chunk['text'][:30]}...'")

    print("\n=== Simulating LLM stream ===\n")

    async def mock_llm_stream():
        """Simulate LLM streaming tokens"""
        text = "The weather today is sunny. It will be warm and pleasant. Perfect for outdoor activities!"
        for token in text:
            await asyncio.sleep(0.02)  # Simulate streaming delay
            yield token

    async def test_llm_streaming():
        async for chunk in tts.stream_from_llm(mock_llm_stream()):
            print(f"Audio chunk: {chunk['audio'].shape[0]} samples, '{chunk['text']}'")

    # Run async test
    asyncio.run(test_llm_streaming())
