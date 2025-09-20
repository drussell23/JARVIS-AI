#!/usr/bin/env python3
"""Test multi-space vision fixes"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from api.pure_vision_intelligence import PureVisionIntelligence
from vision.multi_space_window_detector import MultiSpaceWindowDetector

async def test_multi_space_query():
    """Test that multi-space queries work without errors"""
    print("🔍 Testing Multi-Space Vision Fixes")
    print("=" * 60)
    
    # Create mock Claude client
    class MockClaudeClient:
        async def analyze_image_with_prompt(self, image, prompt, max_tokens=500):
            return {'content': 'Terminal is on Desktop 2'}
    
    # Initialize components
    print("\n1️⃣ Initializing PureVisionIntelligence...")
    intelligence = PureVisionIntelligence(MockClaudeClient())
    
    # Check capture engine
    print(f"   Capture engine initialized: {intelligence.capture_engine is not None}")
    print(f"   Multi-space enabled: {intelligence.multi_space_enabled}")
    
    if not intelligence.capture_engine:
        print("   ❌ ERROR: Capture engine not initialized!")
        return
    
    # Test window detection
    print("\n2️⃣ Testing window detection...")
    detector = MultiSpaceWindowDetector()
    window_data = detector.get_all_windows_across_spaces()
    
    print(f"   Found {len(window_data.get('windows', []))} windows")
    print(f"   Found {len(window_data.get('spaces', []))} spaces")
    
    # Test multi-space query analysis
    print("\n3️⃣ Testing multi-space query analysis...")
    
    try:
        if intelligence.multi_space_extension:
            query_analysis = intelligence.multi_space_extension.process_multi_space_query(
                "Where is Terminal?",
                window_data
            )
            print("   ✅ Query analysis succeeded!")
            if hasattr(query_analysis, 'intent'):
                intent = query_analysis.intent
                print(f"   Query type: {intent.query_type if hasattr(intent, 'query_type') else 'N/A'}")
                print(f"   Needs multi-space: {intent.requires_multi_space if hasattr(intent, 'requires_multi_space') else 'N/A'}")
            elif isinstance(query_analysis, dict):
                print(f"   Query type: {query_analysis.get('intent', {}).get('query_type')}")
                print(f"   Needs multi-space: {query_analysis.get('intent', {}).get('requires_multi_space')}")
        else:
            print("   ❌ Multi-space extension not available")
    except Exception as e:
        print(f"   ❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
    # Test capture request creation
    print("\n4️⃣ Testing capture request...")
    try:
        from vision.multi_space_capture_engine import SpaceCaptureRequest, CaptureQuality
        
        request = SpaceCaptureRequest(
            space_ids=[1, 2],
            quality=CaptureQuality.OPTIMIZED,
            use_cache=False
        )
        
        print("   ✅ SpaceCaptureRequest created successfully")
        print(f"   Quality type: {type(request.quality)}")
        print(f"   Quality value: {request.quality.value}")
        
    except Exception as e:
        print(f"   ❌ ERROR: {type(e).__name__}: {e}")
        
    print("\n✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_multi_space_query())