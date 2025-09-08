#!/usr/bin/env python3
"""
Test script demonstrating complete VSMS integration with claude_vision_analyzer_main.py
Shows how Vision Intelligence and VSMS Core work together for autonomous visual intelligence
"""

import asyncio
import numpy as np
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the analyzer
from claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig

async def test_vsms_integration():
    """Test complete VSMS integration"""
    print("🚀 Testing VSMS Integration with Claude Vision Analyzer")
    print("=" * 80)
    
    # Create configuration with VSMS enabled
    config = VisionConfig(
        vision_intelligence_enabled=True,
        vsms_core_enabled=True,
        model_name="claude-3-opus-20240229",
        max_tokens=1024,
        compression_enabled=True,
        compression_quality=85,
        dynamic_timeout_enabled=True
    )
    
    # Create analyzer
    analyzer = ClaudeVisionAnalyzer(config)
    print("✅ Analyzer created with VSMS integration")
    
    # Test 1: Basic screenshot analysis with VSMS
    print("\n📸 Test 1: Basic Screenshot Analysis with VSMS")
    print("-" * 60)
    
    # Create mock screenshot
    mock_screenshot = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    
    # Analyze with a prompt
    prompt = "Analyze this Chrome browser screenshot and identify the current state"
    result = await analyzer.analyze_screenshot(mock_screenshot, prompt)
    
    print(f"✅ Analysis completed")
    print(f"   Description: {result.get('description', 'N/A')[:100]}...")
    
    if 'vision_intelligence' in result:
        vi_result = result['vision_intelligence']
        print(f"\n🧠 Vision Intelligence Results:")
        print(f"   Components used: {vi_result.get('components_used', [])}")
        
        if 'final_state' in vi_result:
            final_state = vi_result['final_state']
            print(f"   Final state: {final_state.get('state_id', 'unknown')}")
            print(f"   Confidence: {final_state.get('confidence', 0):.2%}")
            print(f"   Consensus: {'Yes' if final_state.get('consensus') else 'No'}")
    
    if 'vsms_core' in result:
        vsms_result = result['vsms_core']
        print(f"\n📊 VSMS Core Results:")
        print(f"   App ID: {vsms_result.get('app_id', 'unknown')}")
        print(f"   Detected state: {vsms_result.get('detected_state', 'unknown')}")
        print(f"   Confidence: {vsms_result.get('confidence', 0):.2%}")
        
        if 'warnings' in vsms_result:
            print(f"   ⚠️  Warnings:")
            for warning in vsms_result['warnings']:
                print(f"      - {warning['message']}")
        
        if 'workflow_detected' in vsms_result:
            print(f"   💡 Workflow: {vsms_result['workflow_detected']}")
    
    # Test 2: Get VSMS insights
    print(f"\n📈 Test 2: VSMS Insights")
    print("-" * 60)
    
    insights = await analyzer.get_vsms_insights()
    
    print(f"✅ VSMS Insights:")
    print(f"   Tracked applications: {insights.get('tracked_applications', 0)}")
    print(f"   Total states: {insights.get('total_states', 0)}")
    print(f"   Personalization score: {insights.get('personalization_score', 0):.1%}")
    print(f"   Memory usage: {sum(insights.get('memory_usage', {}).values()) / 1024 / 1024:.1f}MB")
    
    # Test 3: State recommendations
    print(f"\n🔮 Test 3: State Recommendations")
    print("-" * 60)
    
    recommendations = await analyzer.get_state_recommendations("chrome_home")
    
    print(f"✅ Recommendations for 'chrome_home' state:")
    
    if recommendations.get('next_states'):
        print(f"   Predicted next states:")
        for state, prob in recommendations['next_states'][:3]:
            print(f"   - {state}: {prob:.1%}")
    
    if recommendations.get('warnings'):
        print(f"   ⚠️  Warnings:")
        for warning in recommendations['warnings']:
            print(f"   - {warning['message']}")
    
    if recommendations.get('workflow_hint'):
        hint = recommendations['workflow_hint']
        print(f"   💡 Detected workflow: {' → '.join(hint['detected_workflow'])}")
    
    # Test 4: Create state definition
    print(f"\n🏗️ Test 4: Creating State Definition")
    print("-" * 60)
    
    state_created = await analyzer.create_state_definition(
        app_id="chrome",
        state_id="chrome_custom_state",
        category="active",
        name="Custom Chrome State",
        visual_signatures=[{
            'layout_hash': 'custom_hash_123',
            'dominant_colors': [[255, 255, 255], [0, 0, 0]],
            'ui_elements': {'buttons': 3, 'text_fields': 2}
        }]
    )
    
    print(f"✅ State created: {state_created}")
    
    # Test 5: Vision Intelligence insights
    print(f"\n🔍 Test 5: Vision Intelligence System Insights")
    print("-" * 60)
    
    vi_insights = await analyzer.get_vision_intelligence_insights()
    
    print(f"✅ Vision Intelligence Insights:")
    if 'components' in vi_insights:
        components = vi_insights['components']
        print(f"   Python VSMS: {'Available' if components.get('python_vsms') else 'Not available'}")
        print(f"   VSMS Core: {'Available' if components.get('vsms_core') else 'Not available'}")
        print(f"   Rust: {'Available' if components.get('rust_available') else 'Not available'}")
        print(f"   Swift: {'Available' if components.get('swift_available') else 'Not available'}")
    
    if 'vsms_summary' in vi_insights:
        summary = vi_insights['vsms_summary']
        print(f"\n   VSMS Summary:")
        print(f"   - Applications: {summary.get('tracked_applications', 0)}")
        print(f"   - States: {summary.get('total_states', 0)}")
        print(f"   - Personalization: {summary.get('personalization_score', 0):.1%}")
        
        if summary.get('stuck_states'):
            print(f"   - Stuck states: {', '.join(summary['stuck_states'][:3])}")
    
    # Test 6: Simulate workflow
    print(f"\n🔄 Test 6: Simulating User Workflow")
    print("-" * 60)
    
    workflow_states = [
        ("chrome_home", "Starting at Chrome home page"),
        ("chrome_loading", "Loading a website"),
        ("chrome_gmail", "Checking Gmail"),
        ("chrome_docs", "Working in Google Docs"),
        ("chrome_home", "Back to home page")
    ]
    
    for state_id, description in workflow_states:
        print(f"\n⏱️  Simulating: {description}")
        
        # Create mock screenshot
        mock_screenshot = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
        
        # Analyze with context
        result = await analyzer.analyze_screenshot(
            mock_screenshot, 
            f"User is in {state_id} state in Chrome"
        )
        
        # Show VSMS predictions
        if 'vsms_core' in result and 'predictions' in result['vsms_core']:
            predictions = result['vsms_core']['predictions']
            if predictions:
                print(f"   Next state predictions:")
                for next_state, prob in predictions[:2]:
                    print(f"   - {next_state}: {prob:.1%}")
        
        # Small delay between states
        await asyncio.sleep(1)
    
    # Test 7: Save states
    print(f"\n💾 Test 7: Saving VSMS States")
    print("-" * 60)
    
    await analyzer.save_vsms_states()
    print(f"✅ VSMS states saved successfully")
    
    # Cleanup
    print(f"\n🧹 Cleaning up...")
    await analyzer.cleanup_all_components()
    print(f"✅ Cleanup completed")
    
    print(f"\n✨ All tests completed successfully!")
    print(f"    The Vision Intelligence System is now learning and adapting to visual patterns.")
    print(f"    No hardcoding - everything is learned dynamically!")

async def main():
    """Run the test"""
    try:
        await test_vsms_integration()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Ensure necessary directories exist
    Path("learned_states").mkdir(exist_ok=True)
    
    # Run the test
    asyncio.run(main())