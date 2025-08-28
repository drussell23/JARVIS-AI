#!/usr/bin/env python3
"""
Debug weather command execution
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

async def test_weather_debug():
    """Debug weather command execution"""
    
    # Set API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY")
        return
    
    # Import after path setup
    from system_control.claude_command_interpreter import ClaudeCommandInterpreter
    
    interpreter = ClaudeCommandInterpreter(os.getenv("ANTHROPIC_API_KEY"))
    
    print("Testing weather command interpretation...")
    
    try:
        # Test interpreting the command
        context = {'intent': 'weather', 'user': 'Sir'}
        intent = await interpreter.interpret_command("what's the weather today", context)
        
        print(f"\nIntent:")
        print(f"  Action: {intent.action}")
        print(f"  Target: {intent.target}")
        print(f"  Category: {intent.category}")
        print(f"  Confidence: {intent.confidence}")
        
        # Test executing the intent
        print("\nExecuting intent...")
        result = await interpreter.execute_intent(intent)
        
        print(f"\nResult:")
        print(f"  Success: {result.success}")
        print(f"  Message: {result.message}")
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_weather_debug())