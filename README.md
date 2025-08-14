# 🤖 JARVIS - AI-Powered Chatbot System

A comprehensive AI chatbot system inspired by JARVIS, featuring conversational AI, voice interaction, automation capabilities, and knowledge management with special optimizations for Apple Silicon (M1/M2) Macs.

## 📊 Current Status

### ✅ What's Working
- **Basic Chat Interface**: Simple conversational AI with context awareness
- **API Endpoints**: RESTful API with interactive documentation
- **M1/M2 Support**: Optimized performance using llama.cpp (when configured)
- **Auto-Setup**: One-command installation and startup
- **Web Interface**: Basic frontend for testing

### 🚧 In Development
- **Voice Interaction**: APIs present but require additional setup
- **Knowledge Base**: RAG system structure in place
- **Advanced NLP**: Intent recognition and entity extraction
- **Task Automation**: Framework ready for expansion

### 🔄 Planned
- **Model Training**: Custom model fine-tuning interface
- **Multi-modal Input**: Image and document processing
- **Agent Actions**: Autonomous task execution

## 🌟 Features

- **Conversational AI**: Simple, reliable chatbot with context awareness
- **Voice Interaction**: Speech recognition and synthesis capabilities
- **Task Automation**: Natural language command processing
- **Knowledge Management**: Document storage and retrieval system
- **M1/M2 Optimized**: Native performance on Apple Silicon using llama.cpp
- **Easy Setup**: One-command installation and startup

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js (optional, for React frontend)
- 8GB+ RAM recommended
- macOS (Intel or Apple Silicon), Linux, or Windows

### Installation & Running

#### Quick Start (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd AI-Powered-Chatbot

# Run the system (installs dependencies automatically)
python start_system.py
```

The startup script will:
- ✅ Install all dependencies automatically
- ✅ Set up the backend and frontend
- ✅ Detect M1/M2 Macs and configure optimizations
- ✅ Start all services
- ✅ Open your browser to the frontend

#### Advanced Options

```bash
# Skip dependency installation (if already installed)
python start_system.py --skip-install

# Don't open browser automatically
python start_system.py --no-browser

# Start only backend services
python start_system.py --backend-only
```

#### M1/M2 Mac Special Setup

For Apple Silicon Macs, the system can use llama.cpp for better performance:

```bash
# First time setup for llama.cpp (optional but recommended)
cd backend
./setup_llama_m1.sh

# Then run normally
cd ..
python start_system.py
```

The system will automatically use llama.cpp if available, providing:
- 🚀 Native M1/M2 performance
- 🔧 No PyTorch compatibility issues  
- 💾 Lower memory usage
- ⚡ Faster response times

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

Press `Ctrl+C` in the terminal where start_system.py is running. This will gracefully shut down all services.

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
│   ├── simple_chatbot.py       # Simple chatbot implementation
│   ├── chatbot.py             # Advanced chatbot (with M1 fixes)
│   ├── nlp_engine.py          # NLP processing
│   ├── voice_api.py           # Voice interaction endpoints
│   ├── automation_api.py      # Automation endpoints
│   ├── rag_engine.py          # Knowledge management
│   ├── training_interface.py  # Model training API
│   ├── setup_llama_m1.sh      # M1/M2 setup script
│   └── requirements.txt       # Python dependencies
├── frontend/
│   └── index.html             # Web interface
├── start_system.py            # Main launcher script
└── README.md                  # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# API Keys (optional)
OPENWEATHER_API_KEY=your_api_key
NEWS_API_KEY=your_api_key

# Force llama.cpp usage on M1 (optional)
FORCE_LLAMA=1

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

### M1/M2 Configuration

The system automatically detects Apple Silicon and optimizes accordingly. To customize:

1. **Use llama.cpp** (recommended for M1/M2):
   ```bash
   cd backend
   ./setup_llama_m1.sh
   ```

2. **Force specific backend**:
   ```bash
   # Force llama.cpp
   export FORCE_LLAMA=1
   python start_system.py
   ```

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**
   - The startup script automatically handles port conflicts
   - If issues persist, manually kill the process:
   ```bash
   lsof -ti:8000 | xargs kill -9  # macOS/Linux
   ```

2. **M1/M2 Mac Bus Errors**
   - Install llama.cpp: `cd backend && ./setup_llama_m1.sh`
   - The system will automatically use it when available

3. **Module not found errors**
   ```bash
   # Re-run with dependency installation
   python start_system.py
   ```

4. **"SimpleChatbot" import error**
   - Ensure you're in the project root directory
   - Check that backend/simple_chatbot.py exists

5. **llama.cpp server not starting**
   - Verify model downloaded: `ls ~/Documents/ai-models/`
   - Check port 8080 is free: `lsof -i :8080`

### Logs

Check log files for detailed error information:
- `logs/main_api.log` - Main API logs
- `logs/training_api.log` - Training API logs
- `logs/frontend.log` - Frontend logs

## 💡 Tips

- **For best performance on M1/M2 Macs**: Run `./backend/setup_llama_m1.sh` first
- **To use your own model**: Update model paths in configuration
- **For production deployment**: Use environment variables for sensitive data
- **Memory optimization**: The system automatically manages memory on M1/M2

## 🚀 Roadmap - JARVIS Evolution

### Phase 1: Core Enhancements (Q1 2025)
- [ ] **Improved Language Models**
  - Integrate Llama 3 70B for better responses
  - Add support for Mixtral and other modern models
  - Implement model hot-swapping without restart
  
- [ ] **Enhanced Memory System**
  - Long-term conversation memory with vector database
  - User preference learning and personalization
  - Context-aware responses based on chat history

- [ ] **Better Voice Interaction**
  - Real-time voice conversation (no push-to-talk)
  - Multiple voice options and personalities
  - Emotion detection in voice input

### Phase 2: Intelligence Upgrade (Q2 2025)
- [ ] **Multi-Modal Capabilities**
  - Image understanding and generation
  - Document analysis (PDF, DOCX, etc.)
  - Screen reading and UI automation
  
- [ ] **Advanced Reasoning**
  - Chain-of-thought reasoning
  - Task decomposition and planning
  - Code generation and execution
  
- [ ] **Knowledge Integration**
  - Wikipedia and web search integration
  - Custom knowledge base training
  - Real-time fact checking

### Phase 3: Agent Capabilities (Q3 2025)
- [ ] **Autonomous Actions**
  - Email management and composition
  - Calendar scheduling with conflict resolution
  - File organization and management
  
- [ ] **Tool Integration**
  - Browser automation for web tasks
  - API integration framework
  - Smart home device control
  
- [ ] **Proactive Assistance**
  - Predictive task suggestions
  - Automated routine handling
  - Intelligent notifications

### Phase 4: JARVIS Vision (Q4 2025)
- [ ] **True AI Assistant**
  - Natural conversation flow
  - Multi-step task execution
  - Learning from user feedback
  
- [ ] **Security & Privacy**
  - Local-first processing
  - End-to-end encryption
  - Privacy-preserving analytics
  
- [ ] **Ecosystem Integration**
  - Mobile app companion
  - Browser extension
  - Desktop widget/overlay

### Future Possibilities
- **Distributed Processing**: Multi-device coordination
- **Custom Personalities**: Train JARVIS on specific domains
- **AR/VR Integration**: Spatial computing interface
- **Collaborative AI**: Multiple JARVIS instances working together

## 🛠️ Next Steps for Developers

### Quick Wins (Can implement today)
1. **Improve Chat Responses**
   - Update `simple_chatbot.py` with more sophisticated responses
   - Add personality traits and conversation styles
   - Implement better context handling

2. **Enable llama.cpp Models**
   - Run `./backend/setup_llama_m1.sh` for M1/M2 Macs
   - Configure different models in `chatbot_m1_llama.py`
   - Test with various GGUF models

3. **Enhance Frontend**
   - Improve the chat UI in `frontend/`
   - Add dark mode and themes
   - Implement typing indicators

### Medium-term Goals
1. **Voice Integration**
   - Complete `voice_api.py` implementation
   - Add wake word detection
   - Implement voice synthesis options

2. **Knowledge Base**
   - Activate RAG system in `rag_engine.py`
   - Create document upload interface
   - Implement semantic search

3. **Task Automation**
   - Build out `automation_api.py` capabilities
   - Add calendar integration
   - Create task templates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- llama.cpp for M1/M2 optimization
- The open-source community for various tools and libraries

## 📞 Support

For issues and questions:
- Check the [API Documentation](http://localhost:8000/docs)
- Review the troubleshooting section above
- Open an issue on GitHub

---

Built with ❤️ for simplicity and performance on all platforms.
