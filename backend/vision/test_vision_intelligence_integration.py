#!/usr/bin/env python3
"""
Test script for Vision Intelligence System integration with ClaudeVisionAnalyzer
Demonstrates dynamic state learning and multi-language processing
"""

import asyncio
import os
import numpy as np
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the enhanced ClaudeVisionAnalyzer
from claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig

async def main():
    """Test Vision Intelligence integration"""
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Please set ANTHROPIC_API_KEY environment variable")
        return
    
    # Create config with Vision Intelligence enabled
    config = VisionConfig(
        enable_vision_intelligence=True,
        vision_intelligence_learning=True,
        vision_intelligence_consensus=True,
        state_persistence_enabled=True,
        enable_metrics=True
    )
    
    # Initialize analyzer
    analyzer = ClaudeVisionAnalyzer(api_key=api_key, config=config)
    
    print("🚀 Vision Intelligence Integration Test")
    print("=" * 50)
    
    # Test 1: Basic screenshot analysis with state tracking
    print("\n📸 Test 1: Analyzing screenshot with state tracking...")
    
    # Create a test image (you can replace this with an actual screenshot)
    test_image = Image.new('RGB', (800, 600), color='white')
    
    try:
        # Analyze with state tracking
        result, metrics = await analyzer.analyze_with_state_tracking(
            test_image,
            "What application is shown on screen? Describe what you see.",
            app_id="test_app"
        )
        
        print(f"✅ Analysis completed in {metrics.total_time:.2f}s")
        print(f"   - Claude API time: {metrics.api_call_time:.2f}s")
        print(f"   - Vision Intelligence time: {metrics.vision_intelligence_time:.2f}s")
        
        # Check if Vision Intelligence was used
        if '_vision_intelligence' in result:
            vi_data = result['_vision_intelligence']
            print(f"\n🧠 Vision Intelligence Results:")
            print(f"   - State ID: {vi_data['state'].get('state_id', 'unknown')}")
            print(f"   - Confidence: {vi_data['confidence']:.2%}")
            print(f"   - Components used: {', '.join(vi_data['components_used'])}")
            print(f"   - Consensus: {'Yes' if vi_data['state'].get('consensus') else 'No'}")
        
        # Check for application state in entities
        if 'entities' in result and 'application_state' in result['entities']:
            app_state = result['entities']['application_state']
            print(f"\n📊 Application State:")
            print(f"   - State: {app_state['state_id']}")
            print(f"   - Sources: {', '.join(app_state.get('sources', []))}")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test 2: Get Vision Intelligence insights
    print("\n\n📈 Test 2: Getting Vision Intelligence insights...")
    
    try:
        insights = await analyzer.get_vision_intelligence_insights()
        
        print(f"🔍 System Insights:")
        print(f"   - Enabled: {insights.get('enabled', False)}")
        
        if 'components' in insights:
            components = insights['components']
            print(f"   - Python VSMS: {'Available' if components.get('python_vsms') else 'Not available'}")
            print(f"   - Rust available: {'Yes' if components.get('rust_available') else 'No'}")
            print(f"   - Swift available: {'Yes' if components.get('swift_available') else 'No'}")
        
        if 'tracked_applications' in insights:
            print(f"   - Tracked applications: {len(insights['tracked_applications'])}")
            for app in insights['tracked_applications']:
                print(f"     • {app}")
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    # Test 3: Train Vision Intelligence on a labeled state
    print("\n\n🎯 Test 3: Training Vision Intelligence...")
    
    try:
        # Create another test image
        test_image_2 = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        
        train_result = await analyzer.train_vision_intelligence(
            screenshot=test_image_2,
            app_id="test_app",
            state_id="loading_state",
            state_type="loading"
        )
        
        if train_result['success']:
            print("✅ Training successful!")
            if 'result' in train_result:
                print(f"   - Trained components: {train_result['result'].get('trained_components', [])}")
        else:
            print(f"❌ Training failed: {train_result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    # Test 4: Analyze multiple screenshots to build state understanding
    print("\n\n🔄 Test 4: Building state understanding through multiple analyses...")
    
    try:
        # Simulate analyzing different states of the same app
        states = ["idle", "loading", "active", "error"]
        
        for i, state in enumerate(states):
            # Create different test images
            color = (255 - i*60, i*60, 128)
            test_image = Image.new('RGB', (800, 600), color=color)
            
            result, metrics = await analyzer.analyze_screenshot(
                test_image,
                f"This shows the {state} state of the application",
                use_cache=False  # Disable cache for testing
            )
            
            print(f"   • Analyzed {state} state (time: {metrics.total_time:.2f}s)")
            
            # Small delay between analyses
            await asyncio.sleep(0.5)
        
        # Get insights after multiple analyses
        final_insights = await analyzer.get_vision_intelligence_insights("test_app")
        
        if 'requested_app' in final_insights:
            app_data = final_insights['requested_app']
            print(f"\n📊 Learned State Information:")
            print(f"   - Total states: {app_data.get('total_states', 0)}")
            
            if 'most_common_states' in app_data:
                print("   - Most common states:")
                for state_info in app_data['most_common_states']:
                    print(f"     • {state_info['state_id']} (count: {state_info['count']})")
        
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
    
    # Test 5: Save learned states
    print("\n\n💾 Test 5: Saving learned states...")
    
    try:
        if analyzer.save_vision_intelligence_states():
            print("✅ Successfully saved Vision Intelligence states")
        else:
            print("⚠️  Failed to save states (may not be enabled)")
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
    
    # Cleanup
    print("\n\n🧹 Cleaning up...")
    await analyzer.cleanup_all_components()
    
    print("\n✅ Vision Intelligence integration test completed!")

if __name__ == "__main__":
    asyncio.run(main())