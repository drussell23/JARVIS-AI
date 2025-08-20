# 🤖 JARVIS - Claude AI Powered Assistant

## Overview
JARVIS is now a Claude-exclusive AI assistant system, providing superior language understanding, accurate calculations, and cloud-based processing. Perfect for M1 Macs and systems with limited memory.

## Why Claude-Only?
- **Accurate Calculations**: Claude handles math and order of operations correctly (no more 2+2*2=8 errors!)
- **Superior Understanding**: Better context awareness and reasoning
- **No Memory Issues**: All processing happens in the cloud
- **Consistent Quality**: Same high-quality responses every time
- **200k Token Context**: Handle long conversations and documents

## Setup Requirements

### 1. Get Claude API Key
1. Visit https://console.anthropic.com/
2. Create an account or sign in
3. Generate an API key
4. Add credits to your account

### 2. Configure Environment
Create a `.env` file in the project root:
```env
ANTHROPIC_API_KEY=your-api-key-here
CLAUDE_MODEL=claude-3-haiku-20240307  # Optional: change model
CLAUDE_MAX_TOKENS=1024                 # Optional: adjust response length
CLAUDE_TEMPERATURE=0.7                 # Optional: adjust creativity (0-1)
```

### 3. Install Dependencies
```bash
pip install anthropic python-dotenv fastapi uvicorn pydantic psutil
```

## Starting JARVIS

### Quick Start
```bash
python3 start_system.py
```

This will:
1. Check Claude API configuration
2. Start the backend API
3. Start the JARVIS React interface
4. Open your browser to the Iron Man UI

### Command Line Options
- `--check-only`: Verify setup without starting services
- `--skip-install`: Skip dependency checks
- `--no-browser`: Don't auto-open browser

## Available Interfaces
- **JARVIS UI**: http://localhost:3000/ - Iron Man-inspired chat interface
- **API Docs**: http://localhost:8000/docs - Interactive API documentation
- **Basic Chat**: http://localhost:8000/ - Simple chat interface

## Testing Math Accuracy
```bash
python3 test_claude_math.py
```

This will verify Claude handles calculations correctly:
- Order of operations (2 + 2 * 2 = 6)
- Parentheses, exponents, percentages
- Complex calculations

## Available Models
- `claude-3-haiku-20240307` - Fast and cost-effective (default)
- `claude-3-sonnet-20240229` - Balanced performance
- `claude-3-opus-20240229` - Most capable

## API Usage and Costs
- Claude API is pay-as-you-go
- Haiku: ~$0.25 per million input tokens
- Monitor usage in your Anthropic console
- Set spending limits for safety

## Troubleshooting

### "No credits" error
Add credits to your Anthropic account at https://console.anthropic.com/settings/plans

### API key not found
Make sure your `.env` file is in the project root and contains:
```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Port already in use
The launcher will attempt to kill existing processes. If issues persist:
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

## Features Removed
Since we're using Claude exclusively, these features have been removed:
- Local model support (SimpleChatbot, IntelligentChatbot)
- Dynamic mode switching
- Memory optimization for local models
- Training API
- Model downloading

This simplification means:
- More consistent performance
- No memory management issues
- No model loading delays
- Always accurate responses

Enjoy your Claude-powered JARVIS assistant! 🚀