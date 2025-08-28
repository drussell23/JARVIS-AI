#!/usr/bin/env python3
"""
Test weather command routing to find unpacking issue
"""

import asyncio
import os
import sys
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from voice.intelligent_command_handler import IntelligentCommandHandler

async def test_weather_routing():
    """Test weather command routing"""
    
    # Set API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY")
        return
    
    handler = IntelligentCommandHandler()
    
    weather_queries = [
        "what's the weather for today",
        "what's the weather",
        "tell me the weather",
        "is it going to rain today",
        "what's the temperature"
    ]
    
    for query in weather_queries:
        print(f"\n{'='*50}")
        print(f"Testing: {query}")
        print(f"{'='*50}")
        
        try:
            response, handler_type = await handler.handle_command(query)
            print(f"✅ Success!")
            print(f"Handler: {handler_type}")
            print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_weather_routing())