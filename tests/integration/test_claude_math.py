#!/usr/bin/env python3
"""
Test script to verify Claude API handles math correctly
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from backend.chatbots.claude_chatbot import ClaudeChatbot


async def test_math_calculations():
    """Test various math calculations with Claude"""
    
    print("🧪 Testing Claude API Math Calculations\n")
    
    # Initialize Claude chatbot
    try:
        bot = ClaudeChatbot()
        print("✅ Claude chatbot initialized\n")
    except Exception as e:
        print(f"❌ Failed to initialize Claude: {e}")
        return
    
    # Test cases
    test_cases = [
        ("What is 2 + 2 * 2?", "6"),  # Order of operations
        ("Calculate 10 / 2 + 3", "8"),  # Division and addition
        ("What is (5 + 3) * 2?", "16"),  # Parentheses
        ("Solve: 3^2 + 4^2", "25"),  # Exponents (Pythagorean)
        ("What is 15% of 200?", "30"),  # Percentage
        ("Calculate √144", "12"),  # Square root
    ]
    
    print("Running math tests...\n")
    
    for question, expected in test_cases:
        print(f"Q: {question}")
        
        try:
            response = await bot.generate_response(question)
            print(f"A: {response}")
            
            # Check if expected answer is in response
            if expected.lower() in response.lower():
                print("✅ Correct!\n")
            else:
                print(f"⚠️  Expected {expected} in response\n")
                
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
    # Get usage stats
    stats = bot.get_usage_stats()
    print(f"\n📊 API Usage:")
    print(f"  • API calls: {stats['api_calls']}")
    print(f"  • Total tokens: {stats['total_tokens']}")
    print(f"  • Model: {stats['model']}")


if __name__ == "__main__":
    asyncio.run(test_math_calculations())