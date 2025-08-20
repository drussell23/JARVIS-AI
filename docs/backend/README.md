# Backend - JARVIS API Server

This directory contains the FastAPI backend server that powers JARVIS with Claude AI integration.

## Structure

```
backend/
├── chatbots/
│   └── claude_chatbot.py    # Claude AI integration
├── main.py                  # FastAPI application & endpoints
├── run_server.py           # Server runner with proper paths
├── logs/                   # Application logs
└── static/                 # Static files and demos
```

## Key Components

### main.py
- FastAPI application setup
- Chat endpoints (`/chat`, `/chat/stream`, `/chat/history`)
- Health check endpoint (`/health`)
- Static file serving

### claude_chatbot.py
- Anthropic Claude API integration
- Conversation history management
- Streaming response support
- Token usage tracking

## Running the Backend

```bash
# From project root
python3 backend/run_server.py

# Or use the main launcher
python3 start_system.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/chat` | POST | Send message and get response |
| `/chat/stream` | POST | Get streaming response |
| `/chat/history` | GET | Get conversation history |
| `/chat/history` | DELETE | Clear conversation history |
| `/chat/mode` | GET | Get current mode (always "claude") |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |

## Environment Variables

Required in `.env` file at project root:
```env
ANTHROPIC_API_KEY=your-api-key-here
CLAUDE_MODEL=claude-3-haiku-20240307
CLAUDE_MAX_TOKENS=1024
CLAUDE_TEMPERATURE=0.7
```

## Dependencies

Core requirements:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `anthropic` - Claude API client
- `python-dotenv` - Environment variables
- `pydantic` - Data validation

## Development

### Adding New Endpoints
1. Add route in `main.py` ChatbotAPI class
2. Add to router: `self.router.add_api_route(...)`
3. Implement method with proper type hints

### Testing
```bash
# Test chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Hello JARVIS!"}'

# Check health
curl http://localhost:8000/health
```

## Notes
- All AI processing happens via Claude API (no local models)
- Conversation history is maintained in memory
- CORS is enabled for frontend integration
- Static files served from `/static` directory