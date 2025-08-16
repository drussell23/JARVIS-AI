# 🤖 JARVIS - AI-Powered Chatbot System

A comprehensive AI chatbot system inspired by JARVIS, featuring advanced conversational AI with LangChain integration, voice interaction, automation capabilities, and intelligent memory management with special optimizations for Apple Silicon (M1/M2) Macs.

## 🎯 Key Features

### 🆕 Latest Updates
- **LangChain Integration**: Advanced reasoning with tools (Calculator, Web Search, Wikipedia)
- **Smart Dependency Management**: 10x faster startup with intelligent package checking
- **Dynamic Mode Switching**: Automatically adjusts capabilities based on available memory
- **Enhanced Math Support**: Natural language math calculations ("What is 2+2?" → "4")
- **Memory-Safe Architecture**: Never crashes, gracefully scales features based on resources
- **Intelligent Memory Optimization**: Automatically frees memory to enable advanced features

## 📊 Current Status

### ✅ What's Working
- **Advanced Chat Interface**: LangChain-powered conversational AI with tool usage
- **Mathematical Calculations**: Natural language math support through Calculator tool
- **Web Search**: Real-time information retrieval via DuckDuckGo
- **Knowledge Queries**: Wikipedia integration for factual information
- **Dynamic Intelligence**: Automatic switching between Simple → Intelligent → LangChain modes
- **API Endpoints**: RESTful API with interactive documentation
- **M1/M2 Support**: Optimized performance using llama.cpp
- **Smart Setup**: Only installs missing dependencies
- **Memory Management**: Proactive monitoring and component lifecycle management

### 🚧 In Development
- **Voice Interaction**: APIs present, integration ongoing
- **RAG System**: Knowledge base structure ready for activation
- **Task Automation**: Framework ready for expansion

### 🔄 Planned
- **Model Training**: Custom model fine-tuning interface
- **Multi-modal Input**: Image and document processing
- **Agent Actions**: Autonomous task execution

## 🌟 Architecture

```
┌─────────────────────────────────────────────────────┐
│                    JARVIS System                     │
├─────────────────────────────────────────────────────┤
│                 Dynamic Chatbot                      │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │   Simple    │ │ Intelligent  │ │  LangChain   │ │
│  │   (Basic)   │ │   (NLP)      │ │  (Advanced)  │ │
│  └─────────────┘ └──────────────┘ └──────────────┘ │
│         ↑               ↑               ↑           │
│         └───────────────┴───────────────┘           │
│              Memory-Based Switching                  │
├─────────────────────────────────────────────────────┤
│              Memory Management System                │
│  - Proactive monitoring (HEALTHY/WARNING/CRITICAL)  │
│  - Component priority system                        │
│  - Automatic resource optimization                  │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js (optional, for React frontend)
- 8GB+ RAM (16GB recommended for full features)
- macOS (Intel or Apple Silicon), Linux, or Windows

### Installation & Running

#### Quick Start (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd AI-Powered-Chatbot

# Check system memory status
python start_system.py --memory-status

# Run the system (installs dependencies automatically)
python start_system.py

# For faster startup (skip dependency check)
python start_system.py --skip-install
```

The enhanced startup script will:
- ✅ Check installed packages and only install missing ones
- ✅ Show progress: `[1/5] Installing: package-name`
- ✅ Set up backend and frontend
- ✅ Detect M1/M2 Macs and configure optimizations
- ✅ Start all services with proper memory management
- ✅ Open your browser to the frontend

#### Advanced Options

```bash
# Check dependencies without starting
python start_system.py --check-deps

# Skip dependency installation (fastest startup)
python start_system.py --skip-install

# Don't open browser automatically
python start_system.py --no-browser

# Use async mode for faster installation
python start_system.py --async-mode

# Combine options
python start_system.py --skip-install --no-browser

# Check memory status before starting
python start_system.py --memory-status

# Optimize memory before starting
python start_system.py --optimize-memory
```

## 🧠 Memory Management

JARVIS includes intelligent memory management that automatically adjusts features based on available system memory:

### Memory Modes

| Mode | Memory Usage | Features | Capabilities |
|------|--------------|----------|--------------|
| **LangChain** | < 50% | Full | Math, Web Search, Wikipedia, Advanced Reasoning |
| **Intelligent** | < 65% | Enhanced | NLP, Intent Recognition, Entity Extraction |
| **Simple** | > 80% | Basic | Simple pattern matching, Basic responses |

### Memory Optimization

#### Quick Memory Check
```bash
# Check current memory status
python check_memory.py

# Or use the optimization tool
python optimize_memory.py
```

#### Automatic Optimization
When memory is too high for advanced features, JARVIS can automatically:
- Clear Python caches and run garbage collection
- Kill helper processes (browser helpers, etc.)
- Suspend background applications
- Clear system caches
- Optimize browser memory usage

```bash
# Trigger optimization via API
curl -X POST http://localhost:8000/chat/optimize-memory

# Response shows what was freed:
{
  "success": true,
  "memory_freed_mb": 2156,
  "actions_taken": [
    {"strategy": "kill_helpers", "freed_mb": 1831},
    {"strategy": "clear_caches", "freed_mb": 200}
  ]
}
```

#### Manual Optimization Tips
1. **Close Browser Tabs**: Each tab can use 100-500MB
2. **Quit IDEs**: Cursor/VS Code can use 1-3GB each
3. **Close Chat Apps**: Slack, Discord use 200-500MB
4. **Docker Desktop**: Can use 2-4GB if running

### Startup Memory Check
The startup script now checks memory and provides recommendations:

```bash
python start_system.py
# Will show:
# - Current memory usage
# - Available modes
# - Optimization suggestions if needed
```

#### M1/M2 Mac Optimization

For Apple Silicon Macs, the system uses llama.cpp for optimal performance:

```bash
# Download a model for LangChain (one-time setup)
mkdir -p ~/Documents/ai-models
cd ~/Documents/ai-models
curl -L -o tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Start JARVIS
cd ~/path/to/AI-Powered-Chatbot
python start_system.py
```

Benefits:
- 🚀 Native M1/M2 performance with Metal acceleration
- 🔧 No PyTorch compatibility issues or bus errors
- 💾 Efficient memory usage
- ⚡ Fast response times

## 📚 Using JARVIS

### Chat Examples

#### Mathematical Calculations
```
User: What is 2+2?
JARVIS: 4

User: Calculate 15 * 23 + 47
JARVIS: 392

User: What's 100 divided by 4?
JARVIS: 25
```

#### Web Search
```
User: Search for the latest AI news
JARVIS: [Searches and returns current AI news]

User: What's the weather like in San Francisco?
JARVIS: [Provides current weather information]
```

#### Knowledge Queries
```
User: Tell me about quantum computing
JARVIS: [Provides Wikipedia-based explanation]
```

### Memory Management

JARVIS automatically adjusts its capabilities based on available memory:

| Memory Usage | Mode | Features |
|-------------|------|----------|
| < 50% | LangChain | Full capabilities with all tools |
| 50-65% | Intelligent | Enhanced NLP without tools |
| 65-80% | Intelligent | Basic NLP features |
| > 80% | Simple | Pattern-based responses |

## 📖 API Documentation

### Core Endpoints

#### Chat API
- `POST /chat` - Send a message and get a response
- `POST /chat/stream` - Get streaming responses
- `GET /chat/history` - Get conversation history
- `GET /chat/mode` - Check current chatbot mode
- `POST /chat/mode` - Force a specific mode
- `GET /chat/capabilities` - Get current capabilities

#### Memory API
- `GET /memory/status` - Get memory status
- `GET /memory/report` - Get detailed memory report
- `POST /memory/optimize` - Trigger memory optimization
- `GET /memory/components` - List loaded components

#### Knowledge API (LangChain-enhanced)
- `POST /knowledge/add` - Add documents to knowledge base
- `POST /knowledge/search` - Search with semantic understanding
- `POST /knowledge/feedback` - Improve responses with feedback

### Example API Usage

```python
import requests

# Chat with JARVIS
response = requests.post("http://localhost:8000/chat", 
    json={"user_input": "What is 2+2?"})
print(response.json())
# {"response": "4", "chatbot_mode": "langchain", ...}

# Check capabilities
caps = requests.get("http://localhost:8000/chat/capabilities").json()
print(f"LangChain enabled: {caps['langchain']['enabled']}")
print(f"Available tools: {caps['langchain']['tools']}")
```

## 🏗️ Project Structure

```
AI-Powered-Chatbot/
├── backend/
│   ├── chatbots/
│   │   ├── simple_chatbot.py      # Basic pattern matching
│   │   ├── intelligent_chatbot.py # NLP-enhanced chatbot
│   │   ├── langchain_chatbot.py   # LangChain integration
│   │   └── dynamic_chatbot.py     # Automatic mode switching
│   ├── memory/
│   │   ├── memory_manager.py      # M1-optimized memory management
│   │   ├── memory_api.py          # Memory monitoring endpoints
│   │   └── memory_safe_components.py # Safe component loading
│   ├── api/
│   │   ├── voice_api.py           # Voice interaction
│   │   └── automation_api.py      # Task automation
│   ├── utils/
│   │   └── intelligent_cache.py   # Smart caching system
│   ├── main.py                    # FastAPI application
│   ├── run_server.py              # Server runner with proper paths
│   └── requirements.txt           # Python dependencies
├── frontend/
│   └── [React/HTML interface]
├── start_system.py                # Enhanced launcher script
└── README.md                      # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Enable dynamic chatbot (recommended)
USE_DYNAMIC_CHATBOT=1

# Force llama.cpp usage on M1
FORCE_LLAMA=1

# API Keys (optional)
OPENWEATHER_API_KEY=your_api_key
NEWS_API_KEY=your_api_key

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

### LangChain Models

Place GGUF models in `~/Documents/ai-models/`:
- `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` - Lightweight, fast
- `mistral-7b-instruct-v0.1.Q4_K_M.gguf` - Balanced performance
- `mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf` - Maximum capability

## 🐛 Troubleshooting

### Common Issues

1. **"What is 2+2?" not working**
   - Ensure memory usage is below 50% for LangChain mode
   - Check with: `curl http://localhost:8000/chat/mode`
   - Force LangChain: `POST /chat/mode {"mode": "langchain"}`

2. **Slow startup**
   - Use `--skip-install` flag after first run
   - Check dependencies: `python start_system.py --check-deps`

3. **M1/M2 Mac Issues**
   - Ensure llama-cpp-python is installed
   - Download at least one GGUF model
   - Set `FORCE_LLAMA=1` in environment

4. **Import errors**
   - Run from project root directory
   - Use `python start_system.py` instead of running main.py directly

5. **Memory warnings**
   - Normal behavior - system automatically adjusts
   - Check status: `GET /memory/status`

### Logs

- Backend logs show mode switches and component loading
- Check console output for LangChain tool usage
- Memory state changes are logged in real-time

## 💡 Advanced Features

### LangChain Tools

When in LangChain mode (memory < 50%), JARVIS can:
- **Calculate**: Any mathematical expression
- **Search**: Current information from the web
- **Research**: Detailed knowledge from Wikipedia
- **Analyze**: System status and memory usage

### Dynamic Switching

JARVIS seamlessly transitions between modes:
```
High Memory → Simple (fast, basic)
     ↓
Medium Memory → Intelligent (NLP-enhanced)
     ↓
Low Memory → LangChain (full capabilities)
```

### Memory Optimization

The system includes:
- Proactive memory monitoring
- Component priority system
- Automatic garbage collection
- Emergency shutdown procedures

## 🚀 Roadmap

### Completed ✅
- [x] LangChain integration with tools
- [x] Memory-safe architecture
- [x] M1/M2 optimization with llama.cpp
- [x] Smart dependency management
- [x] Mathematical reasoning
- [x] Web search capabilities

### Next Steps
- [ ] Voice interaction completion
- [ ] RAG system activation
- [ ] Custom tool creation
- [ ] Multi-modal inputs
- [ ] Agent autonomy

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
- LangChain for advanced AI capabilities
- llama.cpp for M1/M2 optimization
- The open-source community for various tools and libraries

## 📞 Support

For issues and questions:
- Check the [API Documentation](http://localhost:8000/docs)
- Review the troubleshooting section
- Test with: `python test_langchain_integration.py`
- Open an issue on GitHub

---

Built with ❤️ for intelligence, performance, and reliability on all platforms.