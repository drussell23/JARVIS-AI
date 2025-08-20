# Claude API Integration for JARVIS

## Overview

JARVIS now supports Anthropic's Claude API, providing high-quality AI responses without any local memory constraints. This is perfect for M1 MacBooks with limited RAM since all processing happens in the cloud.

## Benefits

- **No Memory Constraints**: Claude runs in the cloud, so your 16GB M1 Mac won't struggle
- **Superior Intelligence**: Claude provides more sophisticated, nuanced responses
- **Cost-Effective**: Start with Claude Haiku for fast, affordable responses
- **Easy Setup**: Just add your API key and go!

## Setup

### 1. Get an API Key

1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Create an API key
4. Copy the key (starts with `sk-ant-api...`)

### 2. Set Environment Variable

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

Or create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-api-key-here
```

### 3. Install Dependencies

```bash
pip install anthropic
```

Or install all requirements:
```bash
pip install -r backend/requirements.txt
```

## Usage

### Quick Start with Claude

```bash
python start_jarvis_claude.py
```

This starts JARVIS in Claude-only mode - perfect for M1 Macs!

### Using with Dynamic Mode Switching

```bash
# Enable Claude in dynamic mode
export USE_CLAUDE=1
python start_jarvis.py
```

### Programmatic Usage

```python
from backend.chatbots.claude_chatbot import ClaudeChatbot

# Create Claude chatbot
claude = ClaudeChatbot(
    api_key="your-api-key",
    model="claude-3-haiku-20240307",  # or claude-3-sonnet-20240229, claude-3-opus-20240229
    max_tokens=1024,
    temperature=0.7
)

# Generate response
response = await claude.generate_response("Hello, JARVIS!")
print(response)
```

### With Dynamic Chatbot

```python
from backend.chatbots.dynamic_chatbot import DynamicChatbot
from backend.memory.memory_manager import M1MemoryManager

# Create with Claude enabled
chatbot = DynamicChatbot(
    memory_manager=M1MemoryManager(),
    use_claude=True,
    claude_api_key="your-api-key"
)

# Force Claude mode
await chatbot.force_mode("claude")

# Use as normal
response = await chatbot.generate_response("Hello!")
```

## Available Models

1. **Claude 3 Haiku** (`claude-3-haiku-20240307`)
   - Fastest and most cost-effective
   - Great for quick responses
   - Recommended for general use

2. **Claude 3 Sonnet** (`claude-3-sonnet-20240229`)
   - Balanced performance
   - More capable than Haiku
   - Good for complex tasks

3. **Claude 3 Opus** (`claude-3-opus-20240229`)
   - Most capable model
   - Best for very complex tasks
   - Higher cost

## Cost Considerations

Claude API pricing (as of 2024):
- **Haiku**: ~$0.25 per million input tokens
- **Sonnet**: ~$3 per million input tokens  
- **Opus**: ~$15 per million input tokens

For typical JARVIS usage (conversations, questions, coding help), Haiku is very cost-effective.

## Testing

Run the test suite to verify everything works:

```bash
python test_claude_integration.py
```

## Troubleshooting

### API Key Not Found
```
❌ Error: ANTHROPIC_API_KEY not set!
```
**Solution**: Set your API key as shown in the setup section.

### Import Error
```
ImportError: No module named 'anthropic'
```
**Solution**: Install the anthropic package:
```bash
pip install anthropic
```

### API Errors
- **401 Unauthorized**: Check your API key is correct
- **429 Rate Limited**: You're sending too many requests
- **500 Server Error**: Anthropic's servers are having issues

## Environment Variables

- `ANTHROPIC_API_KEY`: Your Claude API key (required)
- `USE_CLAUDE`: Set to "1" to enable Claude in dynamic mode
- `CLAUDE_MODEL`: Override default model (optional)

## Integration with JARVIS Modes

The Claude integration works seamlessly with JARVIS's mode system:

1. **Claude Mode**: Always uses Claude API
2. **Simple Mode**: Local pattern matching (no AI)
3. **Intelligent Mode**: Local LLM (uses RAM)
4. **LangChain Mode**: Local LLM with tools (uses more RAM)

When Claude is enabled, it overrides other modes since it doesn't use local resources.

## Best Practices

1. **Start with Haiku**: It's fast and affordable for most tasks
2. **Use Streaming**: For better UX, enable response streaming
3. **Monitor Usage**: Check your Anthropic dashboard for usage stats
4. **Handle Errors**: Always have fallback behavior for API failures
5. **Cache Responses**: Consider caching common responses to save costs

## Example Session

```
🚀 Starting JARVIS with Claude API
==================================================
✅ Claude API key found
🧠 Using Claude for all responses (no local memory constraints!)
==================================================

🤖 JARVIS is ready! (Powered by Claude)
Type 'exit' or 'quit' to stop.
--------------------------------------------------

👤 You: Hello JARVIS! How are you running today?

🤖 JARVIS: Good day! I'm operating at peak efficiency, thank you for asking. I'm currently running via Claude's API, which means I can provide you with sophisticated responses without taxing your local system resources. This cloud-based configuration is particularly advantageous for your M1 MacBook Pro, as all the heavy computational work happens on Anthropic's servers rather than your local RAM. How may I assist you today?

👤 You: Can you help me write a Python function to calculate fibonacci numbers?

🤖 JARVIS: Certainly! I'll provide you with an efficient Python function to calculate Fibonacci numbers. Here's an implementation with multiple approaches:

```python
def fibonacci(n):
    """
    Calculate the nth Fibonacci number using dynamic programming.
    Time complexity: O(n), Space complexity: O(1)
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    # Use two variables to track last two numbers
    prev, curr = 0, 1
    
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr

# Alternative: Recursive with memoization
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_memoized(n):
    """
    Recursive Fibonacci with memoization.
    More elegant but uses more memory.
    """
    if n <= 1:
        return n
    return fibonacci_memoized(n - 1) + fibonacci_memoized(n - 2)

# Example usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
```

The first approach uses iteration with constant space complexity, making it very efficient. The second approach uses recursion with memoization, which is more elegant but uses O(n) space. Choose based on your specific needs!

👤 You: exit

👋 Goodbye! Thanks for using JARVIS.
```