#!/usr/bin/env python3
"""
Test script to verify multi-space integration
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.pure_vision_intelligence import PureVisionIntelligence


async def test_multi_space_queries():
    """Test multi-space query detection and handling"""
    
    # Create mock Claude client
    class MockClaudeClient:
        async def analyze_image_with_prompt(self, image, prompt, max_tokens):
            # Check if multi-space context is in prompt
            if "Multi-Space Context:" in prompt:
                return {
                    'content': "I can see multiple desktop spaces. VSCode is open on Desktop 2 with your Python project, and Chrome is on Desktop 1."
                }
            else:
                return {
                    'content': "I can see your current desktop with various applications open."
                }
    
    # Initialize intelligence with multi-space enabled
    intelligence = PureVisionIntelligence(MockClaudeClient(), enable_multi_space=True)
    
    # Test queries
    test_queries = [
        # Multi-space queries
        "Is VSCode open anywhere?",
        "Where is Chrome?",
        "What's on Desktop 2?",
        "Show me all my workspaces",
        "What am I working on across all desktops?",
        
        # Single-space queries
        "What do you see?",
        "Describe the screen",
        "What's the battery level?"
    ]
    
    print("Testing multi-space query detection...\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        
        # Check if multi-space handling is triggered
        if intelligence._should_use_multi_space(query):
            print("✓ Multi-space handling ENABLED")
        else:
            print("  Single-space handling")
            
        # Get response (with mock screenshot)
        try:
            response = await intelligence.understand_and_respond(
                screenshot="mock_screenshot",
                user_query=query
            )
            print(f"Response: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Check multi-space summary
    print("\nMulti-space system summary:")
    summary = intelligence.get_multi_space_summary()
    print(f"Enabled: {summary.get('multi_space_enabled', False)}")
    
    if summary.get('multi_space_enabled'):
        print(f"Current space: {summary.get('current_space', 'Unknown')}")
        print(f"Total spaces: {summary.get('total_spaces', 0)}")


def test_imports():
    """Test that all multi-space components can be imported"""
    print("Testing imports...\n")
    
    try:
        from vision.multi_space_window_detector import MultiSpaceWindowDetector
        print("✓ MultiSpaceWindowDetector imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MultiSpaceWindowDetector: {e}")
    
    try:
        from vision.multi_space_intelligence import MultiSpaceIntelligenceExtension
        print("✓ MultiSpaceIntelligenceExtension imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MultiSpaceIntelligenceExtension: {e}")
    
    try:
        from vision.space_screenshot_cache import SpaceScreenshotCache
        print("✓ SpaceScreenshotCache imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SpaceScreenshotCache: {e}")
    
    try:
        from vision.minimal_space_switcher import MinimalSpaceSwitcher
        print("✓ MinimalSpaceSwitcher imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MinimalSpaceSwitcher: {e}")
    
    print()


if __name__ == "__main__":
    print("=== JARVIS Multi-Space Integration Test ===\n")
    
    # Test imports first
    test_imports()
    
    # Test multi-space functionality
    asyncio.run(test_multi_space_queries())
    
    print("\n=== Test Complete ===")