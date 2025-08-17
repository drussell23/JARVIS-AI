# 🤖 JARVIS - AI Assistant Powered by Claude

<p align="center">
  <img src="https://img.shields.io/badge/AI-Claude%203-blue" alt="Claude AI">
  <img src="https://img.shields.io/badge/UI-Iron%20Man%20Inspired-red" alt="Iron Man UI">
  <img src="https://img.shields.io/badge/Platform-Cloud%20Based-green" alt="Cloud Based">
</p>

## Overview

JARVIS is an AI-powered assistant with an Iron Man-inspired interface, now exclusively powered by Anthropic's Claude AI. Experience superior language understanding, accurate calculations, and cloud-based processing with no local memory constraints.

![JARVIS Interface](https://via.placeholder.com/800x400?text=JARVIS+Iron+Man+Interface)

## ✨ Key Features

- **🎯 Claude AI Integration**: Powered by Anthropic's Claude for superior intelligence
- **🎨 Iron Man UI**: Futuristic holographic interface inspired by JARVIS
- **🧮 Accurate Calculations**: Handles math correctly (2 + 2 * 2 = 6, not 8!)
- **☁️ Cloud-Based**: No local memory usage - perfect for M1 Macs
- **🚀 Fast Responses**: Low latency with Claude's optimized API
- **📚 200k Context**: Handle long conversations and documents

## 🚀 Quick Start

### 1. Get Claude API Key
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create account and generate API key
3. Add credits to your account

### 2. Setup
```bash
# Clone repository
git clone https://github.com/yourusername/AI-Powered-Chatbot.git
cd AI-Powered-Chatbot

# Create .env file
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env

# Install dependencies (one time)
pip install anthropic python-dotenv fastapi uvicorn pydantic psutil

# For the React frontend
cd frontend && npm install && cd ..
```

### 3. Launch JARVIS
```bash
python3 start_system.py
```

This will:
- ✅ Verify Claude API configuration
- ✅ Start the backend API
- ✅ Launch the JARVIS React interface
- ✅ Open your browser to the Iron Man UI

## 🖥️ Interfaces

| Interface | URL | Description |
|-----------|-----|-------------|
| **JARVIS UI** | http://localhost:3000/ | Iron Man-inspired chat interface |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Basic Chat** | http://localhost:8000/ | Simple chat interface |

## 💬 Example Interactions

```
You: What is 2 + 2 * 2?
JARVIS: Following the order of operations (PEMDAS), I need to multiply first, 
then add: 2 * 2 = 4, then 2 + 4 = 6

You: Calculate the square root of 144
JARVIS: The square root of 144 is 12.

You: Explain quantum computing in simple terms
JARVIS: Quantum computing is like having a magical computer that can try many 
solutions at once...
```

## 🔧 Configuration

### Environment Variables (.env)
```env
# Required
ANTHROPIC_API_KEY=sk-ant-api03-...

# Optional
CLAUDE_MODEL=claude-3-haiku-20240307  # or sonnet/opus
CLAUDE_MAX_TOKENS=1024
CLAUDE_TEMPERATURE=0.7
```

### Available Models
- `claude-3-haiku-20240307` - Fast & cost-effective (default)
- `claude-3-sonnet-20240229` - Balanced performance
- `claude-3-opus-20240229` - Most capable

## 📊 API Usage

### Chat Endpoint
```python
import requests

response = requests.post("http://localhost:8000/chat", 
    json={"user_input": "Hello JARVIS!"})
print(response.json()["response"])
```

### Check Status
```bash
curl http://localhost:8000/health
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| "No credits" error | Add credits at https://console.anthropic.com/settings/plans |
| API key not found | Ensure `.env` file exists with `ANTHROPIC_API_KEY` |
| Port already in use | The launcher will auto-kill existing processes |
| React won't compile | Run `cd frontend && npm install` |

## 📁 Project Structure

```
AI-Powered-Chatbot/
├── backend/
│   ├── chatbots/
│   │   └── claude_chatbot.py    # Claude API integration
│   ├── main.py                  # FastAPI application
│   └── run_server.py           # Server runner
├── frontend/
│   ├── src/
│   │   ├── App.js              # JARVIS UI component
│   │   └── App.css             # Iron Man styling
│   └── package.json
├── start_system.py             # Main launcher
├── .env                        # API configuration
└── README.md                   # This file
```

## 🎯 Why Claude?

Previously, JARVIS supported multiple local models with complex memory management. We've simplified to Claude-only because:

- ✅ **Accurate Math**: No more calculation errors
- ✅ **Better Understanding**: Superior context awareness
- ✅ **No Memory Issues**: Cloud-based processing
- ✅ **Consistent Quality**: Same performance every time
- ✅ **Simpler Setup**: No model downloads or management

## 💰 Costs

Claude API is pay-as-you-go:
- Haiku: ~$0.25 per million input tokens
- Typical conversation: < $0.01
- Monitor usage in Anthropic console
- Set spending limits for safety

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/awesome-feature`)
3. Commit changes (`git commit -m 'Add awesome feature'`)
4. Push branch (`git push origin feature/awesome-feature`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Anthropic for Claude AI
- FastAPI for the web framework
- React for the frontend
- Iron Man/Marvel for UI inspiration

---

<p align="center">
Built with ❤️ for superior AI assistance
</p>