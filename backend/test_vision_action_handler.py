#!/usr/bin/env python3
"""
Test script to debug vision action handler issue
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

async def test_vision_action_handler():
    """Test the vision action handler directly"""
    
    print("🔍 Testing Vision Action Handler")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found!")
        return
    print("✅ API key found")
    
    # Import and test vision action handler
    try:
        from system_control.vision_action_handler import get_vision_action_handler
        
        handler = get_vision_action_handler()
        print("✅ Vision action handler initialized")
        
        # Test describe_screen directly
        print("\n📷 Testing describe_screen...")
        result = await handler.describe_screen()
        
        print(f"Success: {result.success}")
        print(f"Description: {result.description[:200]}...")
        if result.error:
            print(f"Error: {result.error}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vision_action_handler())