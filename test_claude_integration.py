#!/usr/bin/env python3
"""
Test script for Claude API integration in JARVIS
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from chatbots.claude_chatbot import ClaudeChatbot
from chatbots.dynamic_chatbot import DynamicChatbot
from memory.memory_manager import M1MemoryManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_claude_direct():
    """Test Claude chatbot directly"""
    print("\n=== Testing Claude Chatbot Directly ===")
    
    # Check if API key is set
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set. Please set it to test Claude integration.")
        print("   You can get an API key from: https://console.anthropic.com/")
        return False
        
    try:
        # Create Claude chatbot
        claude = ClaudeChatbot(
            api_key=api_key,
            model="claude-3-haiku-20240307",  # Fast and cost-effective
            max_tokens=512
        )
        
        # Test simple conversation
        test_prompts = [
            "Hello! I'm testing JARVIS with Claude API. Can you introduce yourself?",
            "What's the capital of France?",
            "Can you write a simple Python function to calculate factorial?",
        ]
        
        for prompt in test_prompts:
            print(f"\n👤 User: {prompt}")
            response = await claude.generate_response(prompt)
            print(f"🤖 Claude: {response}")
            
        # Show usage stats
        stats = claude.get_usage_stats()
        print(f"\n📊 Usage Stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Claude: {e}")
        return False


async def test_dynamic_with_claude():
    """Test Dynamic chatbot with Claude mode"""
    print("\n=== Testing Dynamic Chatbot with Claude Mode ===")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set. Cannot test Claude mode.")
        return False
        
    try:
        # Create memory manager
        memory_manager = M1MemoryManager()
        
        # Create dynamic chatbot with Claude enabled
        chatbot = DynamicChatbot(
            memory_manager=memory_manager,
            use_claude=True,  # Enable Claude
            claude_api_key=api_key,
            auto_switch=False  # Disable auto-switching for testing
        )
        
        # Start monitoring
        await chatbot.start_monitoring()
        
        # Force Claude mode
        print("Switching to Claude mode...")
        await chatbot.force_mode("claude")
        
        # Test conversation
        test_prompts = [
            "Hello JARVIS! I'm using you with Claude API now.",
            "Can you help me understand how your memory management works?",
            "What makes you different when using Claude API vs local models?",
        ]
        
        for prompt in test_prompts:
            print(f"\n👤 User: {prompt}")
            response = await chatbot.generate_response(prompt)
            print(f"🤖 JARVIS (Claude): {response}")
            
        # Show metrics
        print(f"\n📊 Chatbot Metrics: {chatbot.metrics}")
        
        # Cleanup
        await chatbot.stop_monitoring()
        await chatbot.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Dynamic chatbot with Claude: {e}")
        return False


async def test_mode_switching():
    """Test switching between different modes including Claude"""
    print("\n=== Testing Mode Switching with Claude ===")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set. Testing without Claude.")
        use_claude = False
    else:
        use_claude = True
        
    try:
        # Create memory manager
        memory_manager = M1MemoryManager()
        
        # Create dynamic chatbot
        chatbot = DynamicChatbot(
            memory_manager=memory_manager,
            use_claude=use_claude,
            claude_api_key=api_key,
            auto_switch=False  # Manual control for testing
        )
        
        await chatbot.start_monitoring()
        
        # Test different modes
        modes = ["simple", "intelligent"]
        if use_claude:
            modes.append("claude")
            
        for mode in modes:
            print(f"\n--- Testing {mode.upper()} mode ---")
            await chatbot.force_mode(mode)
            
            response = await chatbot.generate_response(
                f"Hello! I'm in {mode} mode. Can you tell me what mode you're running in?"
            )
            print(f"Response: {response}")
            
        # Show final metrics
        print(f"\n📊 Final Metrics: {chatbot.metrics}")
        
        # Cleanup
        await chatbot.stop_monitoring()
        await chatbot.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing mode switching: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 JARVIS Claude Integration Test Suite")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Direct Claude Test", test_claude_direct),
        ("Dynamic Chatbot with Claude", test_dynamic_with_claude),
        ("Mode Switching", test_mode_switching),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 50}")
        print(f"Running: {test_name}")
        print('=' * 50)
        
        success = await test_func()
        results.append((test_name, success))
        
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        
    # Instructions
    print("\n" + "=" * 50)
    print("📝 To use Claude with JARVIS:")
    print("=" * 50)
    print("1. Get an API key from: https://console.anthropic.com/")
    print("2. Set the environment variable:")
    print("   export ANTHROPIC_API_KEY='your-api-key-here'")
    print("3. Run JARVIS with Claude enabled:")
    print("   - Set USE_CLAUDE=1 environment variable")
    print("   - Or pass use_claude=True when creating DynamicChatbot")
    print("4. Claude mode will be used automatically (no memory constraints!)")
    

if __name__ == "__main__":
    asyncio.run(main())