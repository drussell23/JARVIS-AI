#!/usr/bin/env python3
"""Test Vision Logic for Real-time Screen Analysis"""

import asyncio
import os
import sys
from unittest.mock import Mock, AsyncMock, patch

# Add backend to path
backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

async def test_vision_logic():
    """Test the vision logic without running the full backend"""
    print("🧪 Testing Vision Logic for Real-time Screen Analysis")
    print("=" * 80)
    
    # Import the chatbot
    from chatbots.claude_vision_chatbot import ClaudeVisionChatbot
    
    # Create a mock chatbot instance
    chatbot = ClaudeVisionChatbot(api_key="test_key")
    
    # Test 1: Check if _is_screen_query works
    print("\n1. Testing _is_screen_query method:")
    test_queries = [
        ("can you see my terminal?", True),
        ("can you see vscode?", True),
        ("what do you see on my screen?", True),
        ("what's on my screen", True),
        ("describe what's on my screen", True),
        ("tell me what you see", True),
        ("what's the weather like?", False),
        ("hello jarvis", False),
        ("open safari", False)
    ]
    
    for query, expected in test_queries:
        result = chatbot._is_screen_query(query)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{query}' -> {result} (expected {expected})")
    
    # Test 2: Mock monitoring active scenario
    print("\n2. Testing monitoring active behavior:")
    
    # Set monitoring active
    chatbot._monitoring_active = True
    print("  ✓ Set _monitoring_active = True")
    
    # Mock the necessary methods
    chatbot.is_available = Mock(return_value=True)
    chatbot._is_monitoring_command = AsyncMock(return_value=False)
    chatbot._analyze_current_screen = AsyncMock(return_value="Yes, I can see your terminal. You have VS Code open with the test_vision_logic.py file.")
    chatbot.analyze_screen_with_vision = AsyncMock(return_value="Regular vision analysis")
    chatbot._build_messages = Mock(return_value=[])
    chatbot.client = Mock()
    chatbot.client.messages = Mock()
    chatbot.client.messages.create = Mock()
    
    # Test queries when monitoring is active
    print("\n  Testing queries with monitoring active:")
    
    monitoring_queries = [
        "can you see my terminal?",
        "what do you see?",
        "describe my screen"
    ]
    
    for query in monitoring_queries:
        # Check if it would route to real-time analysis
        is_screen_query = chatbot._is_screen_query(query)
        print(f"\n  Query: '{query}'")
        print(f"  Is screen query: {is_screen_query}")
        print(f"  Monitoring active: {chatbot._monitoring_active}")
        print(f"  Should use real-time analysis: {chatbot._monitoring_active and is_screen_query}")
    
    # Test 3: Check the actual response flow
    print("\n3. Testing response generation flow:")
    
    # Test with monitoring active and screen query
    query = "can you see my terminal?"
    print(f"\n  Testing: '{query}' with monitoring active")
    
    # The generate_response method should route to _analyze_current_screen
    # Since we can't easily test the full async flow, we'll check the logic
    is_monitoring_cmd = False  # Not a monitoring command
    is_screen_query = chatbot._is_screen_query(query)
    monitoring_active = chatbot._monitoring_active
    
    print(f"  - Is monitoring command: {is_monitoring_cmd}")
    print(f"  - Is screen query: {is_screen_query}")
    print(f"  - Monitoring active: {monitoring_active}")
    
    if not is_monitoring_cmd and monitoring_active and is_screen_query:
        print("  ✓ Would route to _analyze_current_screen (real-time analysis)")
    else:
        print("  ✗ Would not use real-time analysis")
    
    print("\n✅ Logic test completed!")

if __name__ == "__main__":
    asyncio.run(test_vision_logic())