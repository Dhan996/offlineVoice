# Offline Voice Agent with Chatterbox TTS

Complete offline voice agent using Whisper STT, Ollama LLM, and Chatterbox TTS with voice cloning support.

## Features

- 🎤 **Speech-to-Text**: Whisper (faster-whisper)
- 🤖 **LLM**: Ollama (local inference)
- 🔊 **Text-to-Speech**: Chatterbox TTS with voice cloning
- 🎭 **Voice Cloning**: Clone any voice with 3-10 seconds of reference audio
- ⚡ **Fast**: Optimized caching for reference audio
- 🚀 **GPU Support**: CUDA acceleration for all components

## Installation

### Prerequisites

```bash
# Python 3.11+
python --version

# Ollama (for LLM)
# Install from: https://ollama.ai
ollama pull gemma3:4b
```

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd offline-voice-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1
# OR for CPU only:
# pip install torch torchaudio

# Install other requirements
pip install chatterbox-tts faster-whisper sounddevice numpy
```

## Usage

### Basic Usage

```bash
# Run with Chatterbox TTS
python orchestrator_updated.py \
    --tts-engine chatterbox \
    --chatterbox-device cuda

# Run with voice cloning
python orchestrator_updated.py \
    --tts-engine chatterbox \
    --chatterbox-speaker-wav path/to/reference_voice.wav \
    --chatterbox-device cuda
```

### Command-Line Arguments

#### TTS Options
- `--tts-engine`: Choose 'coqui' or 'chatterbox' (default: coqui)
- `--chatterbox-device`: Device for Chatterbox ('cpu', 'cuda', 'mps')
- `--chatterbox-speaker-wav`: Path to reference audio for voice cloning
- `--chatterbox-multilingual`: Use multilingual model (23 languages)

#### Whisper STT Options
- `--whisper-model`: Model size (default: large-v3)
- `--whisper-device`: Device ('cpu', 'cuda')
- `--compute-type`: Compute type (float16, int8)

#### LLM Options
- `--ollama-model`: Ollama model (default: gemma3:4b)
- `--ollama-base`: Ollama API base URL

#### Audio Options
- `--sample-rate`: Sample rate for recording (default: 16000)
- `--input-device`: Audio input device index
- `--output-device`: Audio output device index

## Voice Cloning

### Preparing Reference Audio

For best voice cloning results:
- **Duration**: 3-10 seconds of clear speech
- **Quality**: Single speaker, minimal background noise
- **Format**: WAV recommended (24kHz or higher)

```bash
# Convert audio to optimal format
ffmpeg -i input.mp3 -ar 24000 -ac 1 reference.wav
```

### Example

```bash
python orchestrator_updated.py \
    --tts-engine chatterbox \
    --chatterbox-speaker-wav my_voice.wav \
    --chatterbox-device cuda \
    --whisper-device cuda
```

## Testing

### Test Chatterbox Caching

```bash
python test_chatterbox_caching.py
```

This will:
- Initialize Chatterbox with reference audio
- Run 3 synthesis requests
- Compare timing to verify caching is working

## Project Structure

```
.
├── orchestrator_updated.py           # Main orchestrator
├── tts_chatterbox_service.py        # Chatterbox TTS wrapper
├── tts_coqui_service.py             # Coqui TTS wrapper (legacy)
├── whisper_service.py               # Whisper STT wrapper
├── llm_ollama_service.py            # Ollama LLM wrapper
├── config.py                        # Configuration
├── test_chatterbox_caching.py       # Caching test script
├── USAGE_GUIDE.md                   # Detailed usage guide
├── PROPER_CACHING_SOLUTION.md       # Caching explanation
└── README.md                        # This file
```

## Performance

### Chatterbox TTS
- **RTF**: ~0.3-0.5x on RTX 4090
- **Latency**: ~400-500ms to first chunk
- **Caching**: Reference audio loaded once, reused for all requests

### Whisper STT
- **Model**: large-v3
- **Device**: CUDA recommended
- **Latency**: ~2-3s for 5s audio on GPU

## Troubleshooting

### CUDA Not Available
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Audio Device Issues
```bash
# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Specify devices manually
python orchestrator_updated.py --input-device 0 --output-device 1
```

### Ollama Connection
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

## GPU Server Setup

### Quick Start on GPU Server

```bash
# SSH to GPU server
ssh user@gpu-server

# Clone and setup
git clone <repo-url>
cd offline-voice-agent
python -m venv .venv
source .venv/bin/activate

# Install with CUDA support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install chatterbox-tts faster-whisper sounddevice numpy

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Run
python orchestrator_updated.py \
    --tts-engine chatterbox \
    --chatterbox-device cuda \
    --whisper-device cuda
```

## Documentation

- [Usage Guide](USAGE_GUIDE.md) - Detailed usage examples
- [Caching Solution](PROPER_CACHING_SOLUTION.md) - How reference audio caching works

## License

[Your License Here]

## Acknowledgments

- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [Ollama](https://ollama.ai)