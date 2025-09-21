#!/usr/bin/env python3
"""
Test Vision Edge Cases - Multi-window, Ambiguity, and Dynamic Content
Tests JARVIS's ability to handle complex visual scenarios intelligently
"""

import asyncio
import sys
import os
import logging

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from backend.chatbots.claude_vision_chatbot import ClaudeVisionChatbot

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_edge_case_queries():
    """Test various edge case scenarios"""
    
    # Initialize chatbot
    chatbot = ClaudeVisionChatbot()
    
    if not chatbot.is_available():
        logger.error("Claude API not available. Please set ANTHROPIC_API_KEY")
        return
    
    # Edge case test scenarios
    test_scenarios = [
        {
            'category': 'Multiple UI Elements',
            'setup': 'Open a browser with a screenshot of a desktop showing time/battery',
            'queries': [
                "What's my battery percentage?",
                "What time is it?",
                "Can you see my status bar?"
            ]
        },
        {
            'category': 'Overlapping Windows',
            'setup': 'Have multiple windows open with one partially covering the menu bar',
            'queries': [
                "What's my battery level?",
                "Can you see the time?",
                "What windows are open?"
            ]
        },
        {
            'category': 'App Visibility',
            'setup': 'Have Slack minimized or on another desktop',
            'queries': [
                "Can you see my Slack messages?",
                "Is Slack open?",
                "What's happening in Slack?"
            ]
        },
        {
            'category': 'Multi-Monitor Awareness',
            'setup': 'Reference apps on other monitors',
            'queries': [
                "What's on my second monitor?",
                "Can you see my other screen?",
                "Check my external display"
            ]
        },
        {
            'category': 'Dynamic Content',
            'setup': 'Have loading spinners or animations',
            'queries': [
                "What's loading on my screen?",
                "Is the page still loading?",
                "What's changing on screen?"
            ]
        },
        {
            'category': 'Ambiguous Requests',
            'setup': 'Multiple browsers or similar apps open',
            'queries': [
                "What website am I looking at?",
                "Which browser is active?",
                "What tabs do I have open?"
            ]
        }
    ]
    
    print("\n🔬 Testing Vision Edge Cases with Pure Intelligence\n")
    print("=" * 70)
    
    for scenario in test_scenarios:
        print(f"\n📋 Category: {scenario['category']}")
        print(f"💡 Setup: {scenario['setup']}")
        print("-" * 70)
        
        for query in scenario['queries']:
            print(f"\n❓ Query: {query}")
            
            try:
                # Test the vision analysis
                response = await chatbot.analyze_screen_with_vision(query)
                
                # Analyze the response for edge case handling
                print(f"\n🤖 JARVIS Response:")
                print(response)
                
                # Check for intelligent edge case handling
                edge_case_indicators = [
                    "I see multiple",
                    "appears to be",
                    "I'm certain",
                    "I'm fairly certain", 
                    "I'm uncertain",
                    "I believe",
                    "partially visible",
                    "obscured",
                    "hidden",
                    "minimized",
                    "another monitor",
                    "different screen",
                    "not visible",
                    "I can only see",
                    "primary display",
                    "loading",
                    "changing"
                ]
                
                intelligence_shown = any(indicator in response for indicator in edge_case_indicators)
                
                if intelligence_shown:
                    print("\n✅ Intelligence Check: Response shows awareness of edge cases")
                else:
                    print("\n⚠️  Intelligence Check: Response may be too generic")
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Test failed: {e}", exc_info=True)
            
            await asyncio.sleep(2)  # Pause between queries
        
        print("\n" + "=" * 70)
    
    print("\n✨ Edge case testing completed!")


async def test_confidence_expressions():
    """Test confidence expression in ambiguous situations"""
    
    chatbot = ClaudeVisionChatbot()
    
    if not chatbot.is_available():
        return
    
    print("\n\n🎯 Testing Confidence Expression\n")
    print("=" * 70)
    
    # Queries that should trigger confidence expressions
    confidence_queries = [
        "What's that number in the corner?",
        "Can you read the small text?",
        "What's behind that window?",
        "Is that the system time or a screenshot?",
        "Which battery indicator is the real one?"
    ]
    
    for query in confidence_queries:
        print(f"\n❓ Ambiguous Query: {query}")
        
        try:
            response = await chatbot.analyze_screen_with_vision(query)
            print(f"\n🤖 JARVIS: {response}")
            
            # Check for confidence expressions
            confidence_words = ["certain", "believe", "likely", "unsure", "unclear", "appears", "seems"]
            has_confidence = any(word in response.lower() for word in confidence_words)
            
            if has_confidence:
                print("✅ Confidence expression detected")
            else:
                print("⚠️  No explicit confidence expression")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        await asyncio.sleep(2)


async def test_monitoring_edge_cases():
    """Test edge cases during continuous monitoring"""
    
    chatbot = ClaudeVisionChatbot()
    
    if not chatbot.is_available():
        return
    
    print("\n\n🔄 Testing Monitoring Mode Edge Cases\n")
    print("=" * 70)
    
    # Start monitoring
    print("Starting continuous monitoring...")
    await chatbot.generate_response("Start monitoring my screen")
    
    await asyncio.sleep(2)
    
    # Test edge case queries while monitoring
    monitoring_queries = [
        {
            'query': "Is Slack visible?",
            'context': "When Slack might be minimized"
        },
        {
            'query': "What's my battery percentage?",
            'context': "With potential UI ambiguity"
        },
        {
            'query': "What just changed?",
            'context': "For dynamic content awareness"
        },
        {
            'query': "Can you see my second monitor?",
            'context': "Multi-monitor limitation"
        }
    ]
    
    for item in monitoring_queries:
        print(f"\n📍 Query: {item['query']}")
        print(f"   Context: {item['context']}")
        
        response = await chatbot.generate_response(item['query'])
        print(f"\n🤖 JARVIS: {response}")
        
        await asyncio.sleep(2)
    
    # Stop monitoring
    print("\nStopping monitoring...")
    await chatbot.generate_response("Stop monitoring")


async def main():
    """Run all edge case tests"""
    
    print("🧪 JARVIS Vision Edge Case Test Suite")
    print("Testing pure vision intelligence for complex scenarios")
    print("=" * 70)
    
    # Instructions for testers
    print("\n📝 INSTRUCTIONS FOR TESTING:")
    print("1. Set up various edge case scenarios as described")
    print("2. Have multiple windows open, some overlapping")
    print("3. Open screenshots of desktops in browsers")
    print("4. Minimize some apps or put them on other desktops")
    print("5. Have some dynamic content (videos, loading pages)")
    print("\nPress Enter when ready to start...")
    input()
    
    # Run edge case tests
    await test_edge_case_queries()
    
    # Test confidence expressions
    await test_confidence_expressions()
    
    # Test monitoring edge cases
    user_input = input("\n\nTest monitoring mode edge cases? (y/n): ")
    if user_input.lower() == 'y':
        await test_monitoring_edge_cases()
    
    print("\n\n✨ All edge case tests completed!")
    print("\nKey things to verify:")
    print("- JARVIS distinguishes between system UI and app content")
    print("- Expresses uncertainty when appropriate")
    print("- Acknowledges when things are not visible")
    print("- Handles multi-window scenarios intelligently")
    print("- Provides specific values when confident")


if __name__ == "__main__":
    asyncio.run(main())