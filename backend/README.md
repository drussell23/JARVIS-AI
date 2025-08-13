# AI-Powered Chatbot with Voice & NLP

A sophisticated AI chatbot with advanced NLP capabilities and JARVIS-like voice interaction.

## Features

### 🤖 Core Chatbot
- Multiple LLM support (GPT-2, BLOOM, DialoGPT, Llama-2)
- Conversation memory and context management
- Customizable personalities and system prompts
- Real-time streaming responses

### 🧠 Advanced NLP
- Intent recognition (15+ intent types)
- Named entity extraction
- Sentiment analysis
- Conversation flow management
- Task planning capabilities
- Response quality enhancement

### 🎤 Voice Capabilities
- **Speech-to-Text**: OpenAI Whisper integration
- **Text-to-Speech**: Multiple engines (Edge TTS, Google TTS, pyttsx3)
- **Wake Word Detection**: "Hey JARVIS" activation
- **Voice Command Processing**: Complete voice interaction pipeline

## Installation

### Prerequisites
- Python 3.8+
- Node.js 14+ (for frontend)
- FFmpeg (for audio processing)
- PortAudio (for microphone access)

### Quick Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-Powered-Chatbot/backend
```

2. Run the setup script:
```bash
./setup.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### System Dependencies

**macOS:**
```bash
brew install portaudio ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg
```

**Windows:**
- Download PyAudio wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
- Install FFmpeg from [here](https://ffmpeg.org/download.html)

## Usage

### Starting the Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the API server
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Interactive API docs: `http://localhost:8000/docs`

### Voice Demo

Open `voice_demo.html` in your browser for a complete voice interaction demo.

## API Endpoints

### Chat Endpoints
- `POST /chat` - Send message and get response
- `POST /chat/stream` - Streaming responses
- `GET /chat/history` - Get conversation history
- `POST /chat/analyze` - Analyze text for NLP insights
- `POST /chat/plan` - Create task plans

### Voice Endpoints
- `POST /voice/tts` - Text-to-speech
- `POST /voice/stt` - Speech-to-text
- `POST /voice/command` - Process voice command
- `WebSocket /voice/stream` - Real-time voice streaming

## Configuration

### Chatbot Models

Supported models:
- `gpt2` (default)
- `gpt2-medium`
- `dialogpt-small`
- `dialogpt-medium`
- `bloom-560m`
- `bloom-1b1`
- `distilgpt2`

Change model:
```python
POST /chat/config
{
    "model_name": "gpt2-medium",
    "system_prompt": "You are a helpful assistant..."
}
```

### Voice Configuration

```python
POST /voice/config
{
    "wake_word": "hey jarvis",
    "tts_engine": "edge_tts",
    "voice_name": "en-US-JennyNeural",
    "speech_rate": 1.0
}
```

## Examples

### Python Client

```python
import requests

# Chat
response = requests.post("http://localhost:8000/chat", 
    json={"user_input": "Hello, how are you?"})
print(response.json())

# Voice Command
import base64
with open("audio.wav", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode()
    
response = requests.post("http://localhost:8000/voice/command",
    json={"audio_data": audio_data, "format": "wav"})
```

### JavaScript Client

```javascript
// Chat
const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({user_input: 'Hello!'})
});

// Text-to-Speech
const ttsResponse = await fetch('http://localhost:8000/voice/tts', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        text: 'Hello, I am your AI assistant',
        engine: 'edge_tts'
    })
});
const audioBlob = await ttsResponse.blob();
const audio = new Audio(URL.createObjectURL(audioBlob));
audio.play();
```

## Troubleshooting

### Import Errors
If you see import errors, ensure you've activated the virtual environment:
```bash
source venv/bin/activate
```

### Audio Issues
- **macOS**: Grant microphone permissions in System Preferences
- **Linux**: Add user to `audio` group: `sudo usermod -a -G audio $USER`
- **Windows**: Run as administrator if microphone access is denied

### Model Download Issues
Some models require authentication:
- Llama-2: Requires Hugging Face account and acceptance of license

## Development

### Project Structure
```
backend/
├── chatbot.py          # Core chatbot engine
├── nlp_engine.py       # NLP processing
├── voice_engine.py     # Voice capabilities
├── voice_api.py        # Voice API endpoints
├── main.py             # FastAPI application
├── requirements.txt    # Python dependencies
└── voice_demo.html     # Voice interaction demo
```

### Adding New Features

1. **New Intent Types**: Edit `nlp_engine.py` → `Intent` enum
2. **New TTS Engine**: Edit `voice_engine.py` → `TTSEngine` enum
3. **New Model**: Edit `chatbot.py` → `SUPPORTED_MODELS` dict

## License

[Your License Here]

## Contributors

[Your Name/Team]