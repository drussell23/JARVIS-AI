#!/usr/bin/env python3
"""Test specific 'other space' query handling"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.multi_space_intelligence import MultiSpaceIntelligenceExtension
from api.pure_vision_intelligence import PureVisionIntelligence

async def test_other_space_query():
    """Test the specific user query about Cursor IDE in other space"""
    print("🔍 Testing 'Other Space' Query Handling")
    print("=" * 60)
    
    # Create extension
    extension = MultiSpaceIntelligenceExtension()
    
    # The exact query from the user
    query = "can you see the Cursor IDE in the other desktop space?"
    
    print(f"\nQuery: '{query}'")
    print("-" * 60)
    
    # Test detection
    should_use = extension.should_use_multi_space(query)
    print(f"Should use multi-space: {'YES ✅' if should_use else 'NO ❌'}")
    
    # Get intent details
    intent = extension.query_detector.detect_intent(query)
    print(f"Query type: {intent.query_type}")
    print(f"Target app: {intent.target_app}")
    print(f"Requires screenshot: {intent.requires_screenshot}")
    print(f"Metadata sufficient: {intent.metadata_sufficient}")
    
    # Test with similar variations
    print("\n📝 Testing variations:")
    print("-" * 60)
    
    variations = [
        "is Cursor open in another desktop?",
        "where is the Cursor IDE?",
        "can you see Cursor on the other space?",
        "what's on the other desktop?",
        "show me what's in the other desktop space"
    ]
    
    for variant in variations:
        should_use = extension.should_use_multi_space(variant)
        intent = extension.query_detector.detect_intent(variant)
        print(f"\n'{variant}'")
        print(f"  → Multi-space: {'YES ✅' if should_use else 'NO ❌'}")
        print(f"  → Type: {intent.query_type.value}")
        print(f"  → App: {intent.target_app}")
    
    # Simulate processing with mock data
    print("\n🧪 Simulating response with mock window data:")
    print("-" * 60)
    
    # Mock window data showing Cursor on Desktop 2
    mock_window_data = {
        'current_space': {'id': 1, 'window_count': 3},
        'spaces': [
            {'space_id': 1, 'is_current': True, 'applications': {'Chrome': ['tab1'], 'Terminal': ['bash']}},
            {'space_id': 2, 'is_current': False, 'applications': {'Cursor': ['project.py'], 'Spotify': ['Music']}}
        ],
        'windows': [
            {'window_id': 1, 'app_name': 'Google Chrome', 'window_title': 'tab1', 'space_id': 1},
            {'window_id': 2, 'app_name': 'Terminal', 'window_title': 'bash', 'space_id': 1},
            {'window_id': 3, 'app_name': 'Cursor', 'window_title': 'project.py', 'space_id': 2},
            {'window_id': 4, 'app_name': 'Spotify', 'window_title': 'Music', 'space_id': 2}
        ],
        'space_window_map': {
            1: [1, 2],
            2: [3, 4]
        }
    }
    
    response_data = extension.process_multi_space_query(query, mock_window_data)
    print(f"Can answer with current data: {response_data['can_answer']}")
    print(f"Confidence level: {response_data['confidence']}")
    if response_data['suggested_response']:
        print(f"\nSuggested response:")
        print(f"  {response_data['suggested_response']}")
    
    # Show what enhanced prompt would look like
    print("\n📄 Enhanced Prompt Preview:")
    print("-" * 60)
    base_prompt = "Analyze this desktop screenshot and respond to the user's query."
    enhanced = extension.prompt_enhancer.enhance_prompt(base_prompt, intent, mock_window_data)
    print(enhanced[:500] + "..." if len(enhanced) > 500 else enhanced)

if __name__ == "__main__":
    asyncio.run(test_other_space_query())