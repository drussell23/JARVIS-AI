#!/usr/bin/env python3
"""
Test script for Semantic Scene Graph integration
Demonstrates how Scene Graph enhances visual understanding with relationships
"""

import asyncio
import numpy as np
import cv2
from datetime import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import components
from semantic_scene_graph import get_scene_graph, NodeType, RelationshipType
from element_detector import ElementDetector


async def create_test_screenshot():
    """Create a mock screenshot with identifiable elements"""
    # Create blank canvas
    screenshot = np.ones((800, 1200, 3), dtype=np.uint8) * 255
    
    # Draw application window
    cv2.rectangle(screenshot, (50, 50), (1150, 750), (200, 200, 200), -1)
    cv2.rectangle(screenshot, (50, 50), (1150, 100), (180, 180, 180), -1)  # Title bar
    
    # Draw UI elements
    # Button 1
    cv2.rectangle(screenshot, (100, 150), (250, 190), (100, 150, 250), -1)
    cv2.putText(screenshot, "Save", (130, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Button 2
    cv2.rectangle(screenshot, (270, 150), (420, 190), (100, 150, 250), -1)
    cv2.putText(screenshot, "Cancel", (290, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Text field
    cv2.rectangle(screenshot, (100, 220), (500, 260), (240, 240, 240), -1)
    cv2.rectangle(screenshot, (100, 220), (500, 260), (150, 150, 150), 2)
    cv2.putText(screenshot, "Enter text here...", (110, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
    
    # Content area
    cv2.rectangle(screenshot, (100, 300), (700, 600), (250, 250, 250), -1)
    cv2.putText(screenshot, "Content Area", (350, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)
    
    # Sidebar
    cv2.rectangle(screenshot, (750, 150), (1100, 700), (230, 230, 230), -1)
    cv2.putText(screenshot, "Sidebar", (850, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    
    return screenshot


async def test_element_detection():
    """Test element detection"""
    print("\n🔍 Testing Element Detection")
    print("=" * 60)
    
    # Create test screenshot
    screenshot = await create_test_screenshot()
    
    # Detect elements
    detector = ElementDetector()
    elements = await detector.detect_elements(screenshot)
    
    print(f"✅ Detected {len(elements)} elements")
    
    # Show element types
    element_types = {}
    for elem in elements:
        elem_type = elem['type']
        element_types[elem_type] = element_types.get(elem_type, 0) + 1
    
    print("\n📊 Element Types:")
    for elem_type, count in element_types.items():
        print(f"   - {elem_type}: {count}")
    
    # Show some detected elements
    print("\n📋 Sample Detected Elements:")
    for elem in elements[:5]:
        bounds = elem['bounds']
        print(f"   - Type: {elem['type']}, Bounds: ({bounds['x']}, {bounds['y']}, {bounds['width']}x{bounds['height']})")
        if 'text' in elem.get('properties', {}):
            print(f"     Text: {elem['properties']['text']}")
    
    return screenshot, elements


async def test_scene_graph_building():
    """Test scene graph construction"""
    print("\n🏗️ Testing Scene Graph Building")
    print("=" * 60)
    
    # Get screenshot and elements
    screenshot, elements = await test_element_detection()
    
    # Get scene graph instance
    scene_graph = get_scene_graph()
    
    # Process the scene
    result = await scene_graph.process_scene(screenshot, elements)
    
    print(f"✅ Scene Graph built successfully")
    print(f"   - Nodes: {result['node_count']}")
    print(f"   - Relationships: {result['relationship_count']}")
    print(f"   - Memory usage: {result['memory_usage']['graph_structure'] / 1024:.1f}KB")
    
    # Show graph metrics
    metrics = result['graph_metrics']
    print(f"\n📈 Graph Metrics:")
    print(f"   - Density: {metrics['density']:.3f}")
    print(f"   - Average degree: {metrics['average_degree']:.2f}")
    print(f"   - Connected: {metrics['is_connected']}")
    print(f"   - Components: {metrics['component_count']}")
    
    # Show node type distribution
    print(f"\n📊 Node Type Distribution:")
    for node_type, count in metrics['node_type_distribution'].items():
        print(f"   - {node_type}: {count}")
    
    # Show key nodes
    print(f"\n🔑 Key Nodes (Top 3):")
    for node in result['key_nodes'][:3]:
        print(f"   - {node['node_type']} (ID: {node['node_id'][:8]}...)")
        print(f"     Importance: {node['importance_score']:.3f}")
        print(f"     Degree: {node['degree']}")
    
    return scene_graph, result


async def test_relationship_discovery():
    """Test relationship discovery"""
    print("\n🔗 Testing Relationship Discovery")
    print("=" * 60)
    
    scene_graph, _ = await test_scene_graph_building()
    
    # Get all relationships
    relationships = scene_graph.builder.relationships
    
    print(f"✅ Found {len(relationships)} relationships")
    
    # Count relationship types
    rel_types = {}
    for rel in relationships:
        rel_type = rel.relationship_type.name
        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
    
    print("\n📊 Relationship Types:")
    for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   - {rel_type}: {count}")
    
    # Show some specific relationships
    print("\n📋 Sample Relationships:")
    sample_rels = relationships[:5]
    for rel in sample_rels:
        source = scene_graph.get_node(rel.source_id)
        target = scene_graph.get_node(rel.target_id)
        if source and target:
            print(f"   - {source.node_type.name} --[{rel.relationship_type.name}]--> {target.node_type.name}")


async def test_graph_intelligence():
    """Test graph intelligence analysis"""
    print("\n🧠 Testing Graph Intelligence")
    print("=" * 60)
    
    scene_graph, result = await test_scene_graph_building()
    
    # Information flow analysis
    info_flow = result['information_flow']
    print(f"✅ Information Flow Paths: {info_flow}")
    
    # Interaction patterns
    patterns = result['interaction_patterns']
    print(f"\n🔄 Interaction Patterns:")
    
    if patterns.get('control_clusters'):
        print(f"   Control Clusters: {len(patterns['control_clusters'])}")
        for cluster in patterns['control_clusters']:
            print(f"   - Cluster size: {cluster['size']}, density: {cluster['density']:.3f}")
    
    if patterns.get('information_hubs'):
        print(f"   Information Hubs: {len(patterns['information_hubs'])}")
    
    if patterns.get('ui_workflows'):
        print(f"   UI Workflows detected: {len(patterns['ui_workflows'])}")
        for workflow in patterns['ui_workflows'][:2]:
            print(f"   - Workflow: {len(workflow)} steps")
    
    # Anomalies
    anomalies = result.get('anomalies', [])
    if anomalies:
        print(f"\n⚠️ Anomalies Detected: {len(anomalies)}")
        for anomaly in anomalies[:3]:
            print(f"   - Type: {anomaly['type']}, Severity: {anomaly['severity']}")


async def test_node_queries():
    """Test querying nodes and relationships"""
    print("\n🔍 Testing Node Queries")
    print("=" * 60)
    
    scene_graph, _ = await test_scene_graph_building()
    
    # Find all UI elements
    ui_elements = scene_graph.find_nodes_by_type(NodeType.UI_ELEMENT)
    print(f"✅ Found {len(ui_elements)} UI elements")
    
    # Find all information nodes
    info_nodes = scene_graph.find_nodes_by_type(NodeType.INFORMATION)
    print(f"✅ Found {len(info_nodes)} information nodes")
    
    # Test relationship queries
    if ui_elements:
        test_node = ui_elements[0]
        print(f"\n📋 Relationships for UI Element {test_node.node_id[:8]}...")
        
        relationships = scene_graph.get_relationships(test_node.node_id)
        print(f"   Total relationships: {len(relationships)}")
        
        # Show relationships by type
        for rel in relationships[:5]:
            if rel.source_id == test_node.node_id:
                target = scene_graph.get_node(rel.target_id)
                if target:
                    print(f"   - {rel.relationship_type.name} -> {target.node_type.name}")
            else:
                source = scene_graph.get_node(rel.source_id)
                if source:
                    print(f"   - {source.node_type.name} -> {rel.relationship_type.name}")


async def test_export():
    """Test graph export functionality"""
    print("\n💾 Testing Graph Export")
    print("=" * 60)
    
    scene_graph, _ = await test_scene_graph_building()
    
    # Export as JSON
    json_export = scene_graph.export_graph('json')
    
    print(f"✅ JSON Export:")
    print(f"   - Nodes: {len(json_export['nodes'])}")
    print(f"   - Relationships: {len(json_export['relationships'])}")
    print(f"   - Timestamp: {json_export['metadata']['timestamp']}")
    
    # Save to file
    export_path = Path("scene_graph_export.json")
    import json
    with open(export_path, 'w') as f:
        json.dump(json_export, f, indent=2)
    
    print(f"✅ Saved to: {export_path}")
    
    # Test GraphML export
    try:
        graphml = scene_graph.export_graph('graphml')
        print(f"✅ GraphML export: {len(graphml)} characters")
    except ImportError:
        print("⚠️ GraphML export requires NetworkX")


async def main():
    """Run all tests"""
    print("🧪 Semantic Scene Graph Test Suite")
    print("=" * 80)
    
    tests = [
        test_scene_graph_building,
        test_relationship_discovery,
        test_graph_intelligence,
        test_node_queries,
        test_export
    ]
    
    for test in tests:
        await test()
        print("\n" + "-" * 80 + "\n")
    
    print("✨ All tests completed successfully!")
    print("The Scene Graph system is now understanding relationships between visual elements!")


if __name__ == "__main__":
    asyncio.run(main())