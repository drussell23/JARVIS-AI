# 🤖 AI-Powered Chatbot System

A comprehensive AI chatbot system featuring advanced NLP, voice interaction, automation capabilities, RAG (Retrieval-Augmented Generation), and custom LLM development tools.

## 🌟 Features

- **Advanced Conversational AI**: Multiple model support (GPT-2, BLOOM, custom models)
- **Voice Interaction**: Speech recognition and synthesis with wake word detection
- **Task Automation**: Calendar management, weather, news, home automation
- **RAG System**: Knowledge base with semantic search and learning capabilities
- **Custom LLM Development**: Train and fine-tune your own language models
- **Multi-modal Support**: Handle text, voice, and structured data

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js (optional, for React frontend)
- 8GB+ RAM recommended
- CUDA-capable GPU (optional, for faster training)

### Installation & Running

#### Option 1: Using the Python Launcher (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd AI-Powered-Chatbot

# Run the system (installs dependencies automatically)
python start_system.py

# Or skip dependency installation if already installed
python start_system.py --skip-install
```

#### Option 2: Using Shell Script (macOS/Linux)

```bash
# Make the script executable
chmod +x quick_start.sh

# Run the system
./quick_start.sh

# Options:
# ./quick_start.sh --skip-install   # Skip dependency installation
# ./quick_start.sh --backend-only   # Start only backend services
# ./quick_start.sh --install-only   # Only install dependencies
```

#### Option 3: Using Batch File (Windows)

```cmd
# Run the system
start_system.bat

# Options:
# start_system.bat --skip-install   # Skip dependency installation
# start_system.bat --backend-only   # Start only backend services
```

#### Option 4: Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Download required models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Start backend services
cd backend
python main.py &          # Main API (port 8000)
python training_interface.py &  # Training API (port 8001)

# Start frontend (if available)
cd ../frontend
npm install && npm start  # Or python -m http.server 3000
```

## 📚 Accessing the System

Once running, access the following services:

### Main Services
- **Frontend**: http://localhost:3000
- **Main API**: http://localhost:8000
- **Training API**: http://localhost:8001

### Demo Interfaces
- **API Documentation**: http://localhost:8000/docs
- **Voice Assistant**: http://localhost:8000/voice_demo.html
- **Automation Demo**: http://localhost:8000/automation_demo.html
- **RAG System**: http://localhost:8000/rag_demo.html
- **LLM Training Studio**: http://localhost:8001/llm_demo.html

## 🛑 Stopping the System

### Using the launcher:
Press `Ctrl+C` in the terminal where the system is running.

### Using shell script:
```bash
./stop_system.sh
```

### Using batch file:
```cmd
stop_system.bat
```

### Manual:
```bash
# Find and kill Python processes
pkill -f "python.*main.py"
pkill -f "python.*training_interface.py"

# Or on Windows:
taskkill /F /IM python.exe
```

## 📖 Documentation

### API Endpoints

#### Chat API
- `POST /chat` - Send a message and get a response
- `POST /chat/stream` - Get streaming responses
- `GET /chat/history` - Get conversation history
- `POST /chat/analyze` - Analyze text for NLP insights

#### Voice API
- `POST /voice/transcribe` - Transcribe audio to text
- `POST /voice/synthesize` - Convert text to speech
- `GET /voice/wake-word/start` - Start wake word detection

#### Automation API
- `POST /automation/command` - Process natural language commands
- `POST /automation/calendar/events` - Create calendar events
- `POST /automation/weather/current` - Get weather information
- `POST /automation/home/devices/control` - Control smart devices

#### Knowledge/RAG API
- `POST /knowledge/add` - Add documents to knowledge base
- `POST /knowledge/search` - Search knowledge base
- `POST /knowledge/feedback` - Provide feedback for learning

#### Training API
- `POST /training/train` - Start model training
- `POST /training/fine-tune` - Fine-tune existing model
- `POST /training/evaluate` - Evaluate model performance
- `GET /training/models` - List available models

## 🏗️ Project Structure

```
AI-Powered-Chatbot/
├── backend/
│   ├── main.py                 # Main FastAPI application
│   ├── chatbot.py             # Core chatbot implementation
│   ├── nlp_engine.py          # NLP processing
│   ├── voice_engine.py        # Voice processing
│   ├── automation_engine.py   # Task automation
│   ├── rag_engine.py          # RAG implementation
│   ├── custom_model.py        # Custom model architecture
│   ├── training_pipeline.py   # Model training
│   ├── fine_tuning.py         # Fine-tuning utilities
│   └── requirements.txt       # Python dependencies
├── frontend/
│   └── index.html             # Basic frontend (or React app)
├── start_system.py            # Python launcher script
├── quick_start.sh             # Shell launcher script
├── start_system.bat           # Windows launcher script
└── README.md                  # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# API Keys (optional)
OPENWEATHER_API_KEY=your_api_key
NEWS_API_KEY=your_api_key

# Model Configuration
DEFAULT_MODEL=gpt2
MAX_TOKENS=150
TEMPERATURE=0.7

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1
```

### Custom Model Configuration

Edit `backend/custom_model.py` to modify model architecture:

```python
config = CustomChatbotConfig(
    vocab_size=50257,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    use_domain_embeddings=True,
    use_intent_aware_attention=True
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Kill process on port (macOS/Linux)
   lsof -ti:8000 | xargs kill -9
   
   # Windows
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   ```

2. **Module not found errors**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   
   # Reinstall dependencies
   pip install -r backend/requirements.txt
   ```

3. **CUDA/GPU errors**
   ```bash
   # Install CPU-only versions
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Memory errors during model training**
   - Reduce batch size in training configuration
   - Use gradient accumulation
   - Enable gradient checkpointing

### Logs

Check log files for detailed error information:
- `logs/main_api.log` - Main API logs
- `logs/training_api.log` - Training API logs
- `logs/frontend.log` - Frontend logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers for pre-trained models
- OpenAI Whisper for speech recognition
- FAISS for efficient similarity search
- The open-source community for various tools and libraries

## 📞 Support

For issues and questions:
- Check the [API Documentation](http://localhost:8000/docs)
- Review logs in the `logs/` directory
- Open an issue on GitHub

---

Built with ❤️ using FastAPI, PyTorch, and modern AI technologies.
