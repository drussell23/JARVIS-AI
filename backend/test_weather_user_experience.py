#!/usr/bin/env python3
"""
Test weather command as user would experience it
"""

import asyncio
import os
import sys
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

async def test_user_experience():
    """Test weather command as user would experience it"""
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY")
        return
    
    from voice.intelligent_command_handler import IntelligentCommandHandler
    
    handler = IntelligentCommandHandler()
    
    queries = [
        "what's the weather for today",
        "what's the temperature",
        "is it going to rain today",
        "tell me the weather"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"You: {query}")
        print("JARVIS: Processing...")
        
        try:
            response, handler_type = await handler.handle_command(query)
            print(f"JARVIS: {response}")
        except Exception as e:
            print(f"JARVIS: I encountered an error: {e}")

if __name__ == "__main__":
    asyncio.run(test_user_experience())