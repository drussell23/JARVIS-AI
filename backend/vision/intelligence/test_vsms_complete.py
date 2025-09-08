#!/usr/bin/env python3
"""
Test the complete Visual State Management System (VSMS)
Demonstrates the full intelligence foundation with all layers
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import VSMS components
from vsms_core import get_vsms, StateCategory, ModalType, ApplicationState
from state_detection_pipeline import VisualSignature, StateDetectionPipeline
from state_intelligence import StateVisit, TimeOfDay

async def simulate_application_usage():
    """Simulate user interaction with applications"""
    vsms = get_vsms()
    
    print("🚀 VSMS Complete Test - Simulating Application Usage")
    print("=" * 60)
    
    # Simulate Chrome browser usage
    app_id = "chrome"
    
    # Define some states for Chrome
    states = [
        ApplicationState(
            state_id="chrome_home",
            category=StateCategory.IDLE,
            name="New Tab Page"
        ),
        ApplicationState(
            state_id="chrome_loading",
            category=StateCategory.LOADING,
            name="Page Loading"
        ),
        ApplicationState(
            state_id="chrome_gmail",
            category=StateCategory.ACTIVE,
            name="Gmail"
        ),
        ApplicationState(
            state_id="chrome_docs",
            category=StateCategory.ACTIVE,
            name="Google Docs"
        ),
        ApplicationState(
            state_id="chrome_error",
            category=StateCategory.ERROR,
            name="Page Not Found"
        ),
        ApplicationState(
            state_id="chrome_download_dialog",
            category=StateCategory.MODAL,
            name="Download Dialog",
            is_modal=True,
            modal_type=ModalType.DIALOG
        )
    ]
    
    # Register states
    for state in states:
        vsms.create_state_definition(
            app_id=app_id,
            state_id=state.state_id,
            category=state.category,
            name=state.name,
            visual_signatures=[{
                'state_id': state.state_id,
                'layout_hash': f"hash_{state.state_id}",
                'dominant_colors': [[255, 255, 255], [0, 0, 0]],
                'ui_elements': {'buttons': 5, 'text_fields': 2}
            }]
        )
    
    print(f"✅ Registered {len(states)} states for Chrome")
    
    # Simulate user workflow
    workflow_sequence = [
        ("chrome_home", 5),      # Home for 5 seconds
        ("chrome_loading", 2),   # Loading for 2 seconds
        ("chrome_gmail", 120),   # Gmail for 2 minutes
        ("chrome_loading", 1),   # Quick load
        ("chrome_docs", 300),    # Docs for 5 minutes (stuck state)
        ("chrome_loading", 2),   # Loading
        ("chrome_error", 10),    # Error page
        ("chrome_home", 3),      # Back home
        ("chrome_download_dialog", 15),  # Download dialog (modal)
        ("chrome_home", 5),      # Home again
    ]
    
    print("\n📊 Simulating user workflow...")
    
    # Process each state in the workflow
    start_time = datetime.now()
    
    for i, (state_id, duration) in enumerate(workflow_sequence):
        # Create mock screenshot (in real usage, this would be actual screenshot)
        mock_screenshot = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
        
        # Simulate state detection
        print(f"\n⏱️  Time: +{(datetime.now() - start_time).seconds}s")
        
        # Process the observation
        result = await vsms.process_visual_observation(mock_screenshot, app_id)
        
        print(f"📍 Current State: {state_id}")
        print(f"   Detected: {result.get('detected_state', 'unknown')}")
        print(f"   Confidence: {result.get('confidence', 0):.2%}")
        
        # Show recommendations if available
        if 'recommendations' in result:
            recs = result['recommendations']
            if recs.get('warnings'):
                print("   ⚠️  Warnings:")
                for warning in recs['warnings']:
                    print(f"      - {warning['message']}")
            
            if recs.get('workflow_hint'):
                hint = recs['workflow_hint']
                print(f"   💡 Workflow detected: {' → '.join(hint['detected_workflow'])}")
            
            if recs.get('next_states'):
                print("   🔮 Predicted next states:")
                for state, prob in recs['next_states'][:3]:
                    print(f"      - {state}: {prob:.1%}")
        
        # Simulate time passing
        await asyncio.sleep(1)  # Speed up simulation
        
        # Force state update for testing (in real usage, detection would handle this)
        vsms.current_states[app_id] = state_id
        vsms.current_state_timestamps[app_id] = datetime.now() - timedelta(seconds=duration)
    
    print("\n" + "=" * 60)
    
    # Get and display insights
    print("\n📈 VSMS Insights:")
    insights = vsms.get_insights()
    
    print(f"\n🔍 Overall Statistics:")
    print(f"   - Applications tracked: {insights['tracked_applications']}")
    print(f"   - Total states: {insights['total_states']}")
    print(f"   - Anomalies detected: {insights['anomalies_detected']}")
    print(f"   - Personalization score: {insights['personalization_score']:.1%}")
    
    if insights.get('stuck_states'):
        print(f"\n⏳ Stuck States Detected:")
        for state in insights['stuck_states']:
            print(f"   - {state}")
    
    if insights.get('common_patterns'):
        print(f"\n🔄 Common Patterns:")
        for pattern, count in insights['common_patterns'][:5]:
            print(f"   - {pattern}: {count} times")
    
    # Get application-specific insights
    app_insights = vsms.get_application_insights(app_id)
    
    print(f"\n📱 Chrome Insights:")
    print(f"   - Total states: {app_insights['total_states']}")
    print(f"   - Current state: {app_insights['current_state']}")
    
    print(f"\n📊 Top States by Usage:")
    for state_info in app_insights['states'][:5]:
        print(f"   - {state_info['name']}: {state_info['detection_count']} times")
        if state_info['average_duration']:
            print(f"     Average duration: {state_info['average_duration']}")
    
    # Get intelligence insights
    intelligence = insights.get('intelligence', {})
    if intelligence:
        print(f"\n🧠 Intelligence Insights:")
        
        if intelligence.get('efficient_workflows'):
            print(f"   Efficient Workflows Found:")
            for workflow in intelligence['efficient_workflows'][:3]:
                print(f"   - {' → '.join(workflow['workflow'])}")
        
        if intelligence.get('improvement_suggestions'):
            print(f"\n💡 Suggestions:")
            for suggestion in intelligence['improvement_suggestions']:
                print(f"   - {suggestion['message']}")
                print(f"     Action: {suggestion['action']}")
    
    # Memory usage
    print(f"\n💾 Memory Usage:")
    memory = insights['memory_usage']
    for component, usage in memory.items():
        usage_mb = usage / (1024 * 1024)
        limit_mb = 50  # Each component has 50MB limit
        print(f"   - {component}: {usage_mb:.1f}MB / {limit_mb}MB ({usage_mb/limit_mb*100:.1f}%)")
    
    # Save state
    print(f"\n💾 Saving VSMS state...")
    vsms.state_history.save_to_disk()
    vsms._save_state_definitions()
    vsms.state_intelligence._save_intelligence_data()
    
    print(f"\n✅ VSMS test completed successfully!")


async def test_state_detection():
    """Test the state detection pipeline"""
    print("\n🔬 Testing State Detection Pipeline")
    print("=" * 60)
    
    pipeline = StateDetectionPipeline()
    
    # Create a test image
    test_image = np.ones((600, 800, 3), dtype=np.uint8) * 255  # White background
    
    # Add some UI elements (rectangles)
    import cv2
    cv2.rectangle(test_image, (50, 50), (750, 100), (200, 200, 200), -1)  # Header
    cv2.rectangle(test_image, (50, 150), (250, 550), (240, 240, 240), -1)  # Sidebar
    cv2.rectangle(test_image, (300, 150), (750, 550), (255, 255, 255), 2)  # Main content
    
    # Extract features
    features = await pipeline.extract_state_features(test_image)
    
    print(f"\n📋 Extracted Features:")
    print(f"   - Layout hash: {features.get('layout_hash', 'none')}")
    print(f"   - Element count: {features.get('element_count', 0)}")
    print(f"   - Dominant colors: {len(features.get('dominant_colors', []))} colors")
    print(f"   - UI elements: {features.get('ui_elements', {})}")
    print(f"   - Has modal: {features.get('has_modal', False)}")
    print(f"   - Brightness: {features.get('brightness', 0):.1f}")
    
    # Create some test signatures
    signatures = [
        VisualSignature(
            signature_type="layout",
            features={
                'state_id': 'test_state_1',
                'layout_hash': features['layout_hash']
            }
        ),
        VisualSignature(
            signature_type="color",
            features={
                'state_id': 'test_state_2',
                'dominant_colors': [[255, 255, 255], [200, 200, 200]]
            }
        )
    ]
    
    # Test ensemble detection
    detected_state, confidence, detection_features = await pipeline.detect_state_ensemble(
        test_image, 
        signatures
    )
    
    print(f"\n🎯 Detection Results:")
    print(f"   - Detected state: {detected_state}")
    print(f"   - Confidence: {confidence:.2%}")
    
    if 'detection_metadata' in detection_features:
        metadata = detection_features['detection_metadata']
        print(f"   - Strategies used: {', '.join(metadata['strategies_used'])}")
        print(f"   - Agreement level: {metadata['agreement_level']:.1%}")


async def main():
    """Run all tests"""
    print("🧪 VSMS Complete Test Suite")
    print("=" * 80)
    
    # Test 1: State Detection Pipeline
    await test_state_detection()
    
    # Add some spacing
    print("\n" * 2)
    
    # Test 2: Full Application Usage Simulation
    await simulate_application_usage()
    
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    # Create necessary directories
    Path("learned_states").mkdir(exist_ok=True)
    
    # Run tests
    asyncio.run(main())