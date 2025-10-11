#!/usr/bin/env python3
"""
Test multi-space vision response to verify it uses Claude API instead of templates
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_multispace_vision():
    """Test the multi-space vision response"""

    # Import after path setup
    from backend.api.pure_vision_intelligence import PureVisionIntelligence
    from backend.vision.multi_space_window_detector import MultiSpaceWindowDetector as MultiSpaceDetector
    from backend.vision.multi_space_intelligence import EnhancedMultiSpaceIntelligence
    from backend.vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer

    print("\n" + "="*70)
    print("🧪 Testing Multi-Space Vision Intelligence")
    print("="*70 + "\n")

    # Initialize components
    print("📦 Initializing components...")
    detector = MultiSpaceDetector()
    multi_space_extension = EnhancedMultiSpaceIntelligence()

    # Initialize Claude vision (this will use mock if no API key)
    claude = ClaudeVisionAnalyzer()

    # Initialize Pure Vision Intelligence
    vision = PureVisionIntelligence(
        claude=claude,
        multi_space_detector=detector,
        multi_space_extension=multi_space_extension
    )

    # Test query
    test_query = "What's happening across my desktop spaces?"
    print(f"📝 Test Query: '{test_query}'")
    print("-" * 50)

    # Get mock screenshot (just a placeholder)
    import numpy as np
    mock_screenshot = np.zeros((100, 100, 3), dtype=np.uint8)  # Black image

    # Test the response
    print("🤖 Getting response from JARVIS...")
    response = await vision.understand_and_respond(mock_screenshot, test_query)

    print("\n📊 Response Analysis:")
    print("-" * 50)
    print(f"Response Length: {len(response)} characters")
    print(f"Response Type: {'Template-based' if 'workspace appears well-organized' in response.lower() else 'Claude API-based'}")
    print("\n💬 JARVIS Response:")
    print("-" * 50)
    print(response)
    print("\n" + "="*70)

    # Check if response is using Claude API
    if "workspace appears well-organized" in response.lower():
        print("⚠️  WARNING: Still using template-based response!")
        print("   The fix needs more work to properly route to Claude API")
        return False
    else:
        print("✅ SUCCESS: Using intelligent Claude API response!")
        return True

if __name__ == "__main__":
    success = asyncio.run(test_multispace_vision())
    sys.exit(0 if success else 1)