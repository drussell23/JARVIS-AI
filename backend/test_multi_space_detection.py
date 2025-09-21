#!/usr/bin/env python3
"""Test multi-space detection for various queries"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.pure_vision_intelligence import PureVisionIntelligence
from vision.multi_space_intelligence import MultiSpaceIntelligenceExtension

async def test_detection():
    """Test if multi-space queries are detected correctly"""
    print("🔍 Testing Multi-Space Query Detection")
    print("=" * 60)
    
    # Create components
    extension = MultiSpaceIntelligenceExtension()
    
    # Test queries
    test_queries = [
        "Where is Terminal?",
        "can you see if VSCode is open on another desktop space?",
        "What do you see?",
        "Show me all my workspaces",
        "What's on Desktop 2?",
        "Find Chrome across all spaces",
        "Is VSCode open?",
        "Look for VSCode in other spaces",
        "can you see the Cursor IDE in the other desktop space?",
        "is Chrome open in another space?",
        "what's on the other desktop?",
    ]
    
    print("\n📋 Query Detection Results:")
    print("-" * 60)
    
    for query in test_queries:
        should_use = extension.should_use_multi_space(query)
        print(f"Query: {query}")
        print(f"  → Multi-space: {'YES ✅' if should_use else 'NO ❌'}")
        
        # Also show the intent details
        intent = extension.query_detector.detect_intent(query)
        print(f"  → Query type: {intent.query_type}")
        print(f"  → Target app: {intent.target_app}")
        print(f"  → Target space: {intent.target_space}")
        print()
    
    # Test with PureVisionIntelligence
    print("\n🧠 Testing with PureVisionIntelligence:")
    print("-" * 60)
    
    class MockClaudeClient:
        async def analyze_image_with_prompt(self, image, prompt, max_tokens=500):
            return {'content': 'Mock response'}
    
    intelligence = PureVisionIntelligence(MockClaudeClient(), enable_multi_space=True)
    
    print(f"Multi-space enabled: {intelligence.multi_space_enabled}")
    print(f"Has multi_space_extension: {hasattr(intelligence, 'multi_space_extension')}")
    print(f"Has capture_engine: {hasattr(intelligence, 'capture_engine')}")
    
    if intelligence.multi_space_enabled:
        query = "can you see if VSCode is open on another desktop space?"
        should_use = intelligence._should_use_multi_space(query)
        print(f"\nQuery: '{query}'")
        print(f"Should use multi-space: {'YES ✅' if should_use else 'NO ❌'}")

if __name__ == "__main__":
    asyncio.run(test_detection())