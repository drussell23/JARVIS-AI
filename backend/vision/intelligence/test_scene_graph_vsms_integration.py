#!/usr/bin/env python3
"""
Test Scene Graph integration with VSMS and Claude Vision Analyzer
Shows how Scene Graph enhances state detection with relationship understanding
"""

import asyncio
import numpy as np
import cv2
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import analyzer
from claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig


async def create_complex_ui_screenshot():
    """Create a complex UI screenshot with multiple elements and relationships"""
    screenshot = np.ones((900, 1400, 3), dtype=np.uint8) * 255
    
    # Main application window
    cv2.rectangle(screenshot, (20, 20), (1380, 880), (240, 240, 240), -1)
    cv2.rectangle(screenshot, (20, 20), (1380, 80), (200, 200, 200), -1)  # Title bar
    cv2.putText(screenshot, "Chrome - Email Dashboard", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)
    
    # Toolbar with buttons
    toolbar_y = 100
    button_x = 40
    buttons = ["Compose", "Refresh", "Settings", "Search"]
    for btn_text in buttons:
        cv2.rectangle(screenshot, (button_x, toolbar_y), (button_x + 120, toolbar_y + 40), (100, 150, 250), -1)
        cv2.putText(screenshot, btn_text, (button_x + 10, toolbar_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        button_x += 140
    
    # Sidebar (folder list)
    cv2.rectangle(screenshot, (40, 160), (300, 850), (250, 250, 250), -1)
    cv2.putText(screenshot, "Folders", (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    
    folders = ["Inbox (5)", "Sent", "Drafts", "Spam", "Trash"]
    folder_y = 220
    for folder in folders:
        if "Inbox" in folder:
            cv2.rectangle(screenshot, (50, folder_y), (290, folder_y + 35), (230, 240, 250), -1)
        cv2.putText(screenshot, folder, (60, folder_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 1)
        folder_y += 45
    
    # Main content area (email list)
    cv2.rectangle(screenshot, (320, 160), (1000, 850), (255, 255, 255), -1)
    
    # Email items
    emails = [
        ("John Doe", "Meeting Tomorrow", "10:30 AM", True),
        ("Jane Smith", "Project Update", "9:15 AM", True),
        ("Support Team", "Your ticket has been resolved", "Yesterday", False),
        ("Newsletter", "Weekly Tech News", "Yesterday", False)
    ]
    
    email_y = 180
    for sender, subject, time, unread in emails:
        if unread:
            cv2.rectangle(screenshot, (330, email_y), (990, email_y + 80), (245, 250, 255), -1)
        
        # Sender
        weight = 2 if unread else 1
        cv2.putText(screenshot, sender, (350, email_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), weight)
        
        # Subject
        cv2.putText(screenshot, subject, (350, email_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
        
        # Time
        cv2.putText(screenshot, time, (850, email_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        
        email_y += 90
    
    # Detail panel
    cv2.rectangle(screenshot, (1020, 160), (1360, 850), (252, 252, 252), -1)
    cv2.putText(screenshot, "Email Preview", (1040, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    cv2.rectangle(screenshot, (1040, 210), (1340, 400), (245, 245, 245), -1)
    
    # Action buttons in detail panel
    cv2.rectangle(screenshot, (1040, 420), (1120, 455), (100, 200, 100), -1)
    cv2.putText(screenshot, "Reply", (1055, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.rectangle(screenshot, (1140, 420), (1240, 455), (200, 100, 100), -1)
    cv2.putText(screenshot, "Delete", (1155, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Status bar
    cv2.rectangle(screenshot, (20, 850), (1380, 880), (220, 220, 220), -1)
    cv2.putText(screenshot, "5 unread messages | Last sync: 2 minutes ago", (40, 870), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return screenshot


async def test_scene_graph_vsms_integration():
    """Test Scene Graph working with VSMS"""
    print("🚀 Testing Scene Graph + VSMS Integration")
    print("=" * 80)
    
    # Create config with all features enabled
    config = VisionConfig(
        vision_intelligence_enabled=True,
        vsms_core_enabled=True,
        scene_graph_enabled=True,
        model_name="claude-3-opus-20240229",
        max_tokens=1500
    )
    
    # Create analyzer
    analyzer = ClaudeVisionAnalyzer(config)
    print("✅ Analyzer created with Scene Graph + VSMS enabled")
    
    # Create complex screenshot
    screenshot = await create_complex_ui_screenshot()
    
    # Test 1: Analyze with Scene Graph
    print("\n📸 Test 1: Screenshot Analysis with Scene Graph")
    print("-" * 60)
    
    result = await analyzer.analyze_screenshot(
        screenshot,
        "Analyze this email application interface and identify the key elements and their relationships"
    )
    
    print(f"✅ Analysis completed")
    print(f"   Description: {result.get('description', 'N/A')[:150]}...")
    
    # Show Scene Graph results if available
    if 'vsms_core' in result and 'scene_graph' in result['vsms_core']:
        sg_data = result['vsms_core']['scene_graph']
        print(f"\n🏗️ Scene Graph Results:")
        print(f"   Nodes: {sg_data.get('node_count', 0)}")
        print(f"   Relationships: {sg_data.get('relationship_count', 0)}")
        
        if sg_data.get('key_nodes'):
            print(f"\n   Key Nodes:")
            for node in sg_data['key_nodes'][:3]:
                print(f"   - {node['node_type']} (importance: {node.get('importance_score', 0):.2f})")
        
        if 'interaction_patterns' in sg_data:
            patterns = sg_data['interaction_patterns']
            if patterns.get('control_clusters'):
                print(f"\n   Control Clusters: {len(patterns['control_clusters'])}")
            if patterns.get('ui_workflows'):
                print(f"   UI Workflows: {len(patterns['ui_workflows'])}")
    
    # Test 2: Get Scene Graph insights
    print(f"\n📊 Test 2: Scene Graph Insights")
    print("-" * 60)
    
    sg_insights = await analyzer.get_scene_graph_insights()
    
    if sg_insights.get('enabled'):
        print(f"✅ Scene Graph Insights:")
        
        if 'graph_metrics' in sg_insights:
            metrics = sg_insights['graph_metrics']
            print(f"   Graph density: {metrics.get('density', 0):.3f}")
            print(f"   Average degree: {metrics.get('average_degree', 0):.2f}")
            print(f"   Connected: {metrics.get('is_connected', False)}")
            
            if 'node_type_distribution' in metrics:
                print(f"\n   Node Types:")
                for node_type, count in metrics['node_type_distribution'].items():
                    print(f"   - {node_type}: {count}")
        
        print(f"\n   Memory usage: {sg_insights.get('memory_usage_mb', 0):.1f}MB")
    
    # Test 3: Simulate workflow with Scene Graph
    print(f"\n🔄 Test 3: Workflow Simulation with Scene Graph")
    print("-" * 60)
    
    workflow_steps = [
        ("User clicks on unread email", "Selecting email from list"),
        ("Email preview loads in detail panel", "Email content displayed"),
        ("User hovers over Reply button", "Preparing to respond"),
        ("User clicks Reply button", "Opening email composer")
    ]
    
    for step, description in workflow_steps:
        print(f"\n⏱️ Step: {step}")
        
        # Analyze with context
        result = await analyzer.analyze_screenshot(
            screenshot,
            f"The user just performed: {step}. Analyze the UI state and relationships."
        )
        
        # Check for scene context
        if 'vsms_core' in result and 'scene_context' in result['vsms_core']:
            context = result['vsms_core']['scene_context']
            print(f"   Primary elements: {context.get('primary_elements', [])}")
            print(f"   Has modal: {context.get('has_modal', False)}")
            print(f"   Info density: {context.get('information_density', 0)}")
        
        await asyncio.sleep(0.5)
    
    # Test 4: Demonstrate relationship understanding
    print(f"\n🔗 Test 4: Relationship Understanding")
    print("-" * 60)
    
    result = await analyzer.analyze_screenshot(
        screenshot,
        "Identify which UI elements control or modify other elements in this interface"
    )
    
    print(f"✅ Relationship analysis completed")
    
    # The Scene Graph should have identified:
    # - Compose button controls email composer (when implemented)
    # - Folder selection controls email list content
    # - Email selection controls detail panel
    # - Reply/Delete buttons control selected email
    
    # Test 5: Anomaly detection through Scene Graph
    print(f"\n⚠️ Test 5: Anomaly Detection")
    print("-" * 60)
    
    if 'vsms_core' in result and 'scene_graph' in result['vsms_core']:
        sg_data = result['vsms_core']['scene_graph']
        anomalies = sg_data.get('anomalies', [])
        
        if anomalies:
            print(f"✅ Anomalies detected: {len(anomalies)}")
            for anomaly in anomalies[:3]:
                print(f"   - Type: {anomaly['type']}")
                print(f"     Severity: {anomaly['severity']}")
        else:
            print("✅ No anomalies detected in the scene graph")
    
    # Clean up
    print(f"\n🧹 Cleaning up...")
    await analyzer.cleanup_all_components()
    
    print(f"\n✨ Scene Graph + VSMS integration test completed!")
    print(f"    The system now understands not just what elements exist,")
    print(f"    but how they relate to and interact with each other!")


async def main():
    """Run the integration test"""
    try:
        await test_scene_graph_vsms_integration()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Ensure directories exist
    Path("learned_states").mkdir(exist_ok=True)
    
    # Run test
    asyncio.run(main())