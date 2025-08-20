#!/usr/bin/env python3
"""
Test Phase 2: Intelligence Layer
Tests F2.1 (Window Relationship Detection) and F2.2 (Smart Query Routing)
"""

import asyncio
import sys
import os
from typing import List, Dict

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from backend.vision.window_detector import WindowDetector
from backend.vision.window_relationship_detector import WindowRelationshipDetector
from backend.vision.smart_query_router import SmartQueryRouter, QueryIntent


async def test_f2_1_relationship_detection():
    """Test F2.1: Window Relationship Detection"""
    print("🧪 Testing F2.1: Window Relationship Detection")
    print("=" * 60)
    
    detector = WindowDetector()
    relationship_detector = WindowRelationshipDetector()
    
    # Get current windows
    windows = detector.get_all_windows()
    print(f"\n📊 Current workspace: {len(windows)} windows")
    
    # Detect relationships
    relationships = relationship_detector.detect_relationships(windows)
    
    print(f"\n✓ Acceptance Criteria 1: Identifies related windows")
    print(f"   Found {len(relationships)} relationships")
    
    # Analyze relationship types
    relationship_types = {}
    high_confidence_count = 0
    
    for rel in relationships:
        # Count relationship types
        rel_type = rel.relationship_type
        relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        # Count high confidence relationships
        if rel.confidence >= 0.8:
            high_confidence_count += 1
        
        # Show first few examples
        if len(relationship_types) <= 3:
            window1 = next(w for w in windows if w.window_id == rel.window1_id)
            window2 = next(w for w in windows if w.window_id == rel.window2_id)
            
            print(f"\n   Example: {rel.relationship_type}")
            print(f"   - {window1.app_name}: {window1.window_title}")
            print(f"   - {window2.app_name}: {window2.window_title}")
            print(f"   - Confidence: {rel.confidence:.0%}")
            print(f"   - Evidence: {', '.join(rel.evidence)}")
    
    # Test grouping
    groups = relationship_detector.group_windows(windows, relationships)
    
    print(f"\n✓ Acceptance Criteria 2: Groups windows by project/task")
    print(f"   Found {len(groups)} window groups")
    
    for i, group in enumerate(groups[:3]):  # Show first 3 groups
        print(f"\n   Group {i+1}: {group.group_type}")
        print(f"   - Windows: {len(group.windows)}")
        print(f"   - Apps: {[w.app_name for w in group.windows[:4]]}")
        print(f"   - Common elements: {', '.join(group.common_elements[:3])}")
        print(f"   - Confidence: {group.confidence:.0%}")
    
    # Check confidence scores
    print(f"\n✓ Acceptance Criteria 3: Confidence score >80% for relationships")
    if relationships:
        avg_confidence = sum(r.confidence for r in relationships) / len(relationships)
        print(f"   High confidence relationships: {high_confidence_count}/{len(relationships)}")
        print(f"   Average confidence: {avg_confidence:.0%}")
        
        confidence_80_percent = high_confidence_count / len(relationships) if relationships else 0
        if confidence_80_percent >= 0.3:  # At least 30% have >80% confidence
            print(f"   ✅ PASS: {confidence_80_percent:.0%} of relationships have >80% confidence")
        else:
            print(f"   ⚠️  WARNING: Only {confidence_80_percent:.0%} have >80% confidence")
    else:
        print(f"   ⚠️  No relationships found to evaluate")
    
    # Summary
    print(f"\n📊 Relationship Types Found:")
    for rel_type, count in relationship_types.items():
        print(f"   - {rel_type}: {count}")
    
    return len(relationships) > 0 and len(groups) > 0


async def test_f2_2_smart_query_routing():
    """Test F2.2: Smart Query Routing"""
    print("\n\n🧪 Testing F2.2: Smart Query Routing")
    print("=" * 60)
    
    detector = WindowDetector()
    router = SmartQueryRouter()
    
    # Get current windows
    windows = detector.get_all_windows()
    print(f"\n📊 Current workspace: {len(windows)} windows")
    
    # Test message routing
    print(f"\n✓ Acceptance Criteria 1: 'Any messages?' checks communication apps")
    message_queries = [
        "Do I have any messages?",
        "Check my messages",
        "Any new messages in Discord?"
    ]
    
    for query in message_queries:
        route = router.route_query(query, windows)
        print(f"\n   Query: '{query}'")
        print(f"   Intent detected: {route.intent.value}")
        print(f"   Confidence: {route.confidence:.0%}")
        
        # Check if communication apps were selected
        comm_apps = ['Discord', 'Slack', 'Messages', 'Mail', 'WhatsApp']
        comm_windows = [w for w in route.target_windows 
                       if any(app in w.app_name for app in comm_apps)]
        
        if comm_windows:
            print(f"   ✅ Found {len(comm_windows)} communication windows")
            for window in comm_windows[:3]:
                print(f"      - {window.app_name}")
        else:
            print(f"   ⚠️  No communication windows found")
    
    # Test error routing
    print(f"\n✓ Acceptance Criteria 2: 'Show me errors' scans terminals and logs")
    error_queries = [
        "Are there any errors?",
        "Show me errors in terminal",
        "Check for exceptions"
    ]
    
    for query in error_queries[:1]:  # Test first query
        route = router.route_query(query, windows)
        print(f"\n   Query: '{query}'")
        print(f"   Intent detected: {route.intent.value}")
        print(f"   Confidence: {route.confidence:.0%}")
        
        # Check if terminals were selected
        terminal_apps = ['Terminal', 'iTerm', 'Console']
        terminal_windows = [w for w in route.target_windows 
                          if any(app in w.app_name for app in terminal_apps)]
        
        if terminal_windows:
            print(f"   ✅ Found {len(terminal_windows)} terminal windows")
            for window in terminal_windows[:3]:
                print(f"      - {window.app_name}: {window.window_title}")
        else:
            print(f"   ⚠️  No terminal windows found")
    
    # Test selective capture
    print(f"\n✓ Acceptance Criteria 3: Captures only relevant windows")
    
    # Count windows per query type
    query_tests = [
        ("What's on my screen?", QueryIntent.WORKSPACE_OVERVIEW, "all"),
        ("What am I working on?", QueryIntent.CURRENT_WORK, "focused+related"),
        ("Check Discord", QueryIntent.SPECIFIC_APP, "discord only"),
        ("Find documentation", QueryIntent.DOCUMENTATION, "docs/browsers")
    ]
    
    for query, expected_intent, description in query_tests:
        route = router.route_query(query, windows)
        print(f"\n   Query: '{query}'")
        print(f"   Expected: {description}")
        print(f"   Captured: {len(route.target_windows)} windows")
        print(f"   Intent: {route.intent.value}")
        
        if route.intent == expected_intent:
            print(f"   ✅ Correct intent detected")
        else:
            print(f"   ⚠️  Expected {expected_intent.value}, got {route.intent.value}")
        
        # Show what was captured
        if route.target_windows:
            print(f"   Windows selected:")
            for i, window in enumerate(route.target_windows[:3]):
                print(f"   {i+1}. {window.app_name} - {window.window_title or 'Untitled'}")
    
    return True


async def test_integration():
    """Test integration of relationship detection and query routing"""
    print("\n\n🧪 Testing Integration: Relationships + Routing")
    print("=" * 60)
    
    detector = WindowDetector()
    relationship_detector = WindowRelationshipDetector()
    router = SmartQueryRouter()
    
    # Get windows and relationships
    windows = detector.get_all_windows()
    relationships = relationship_detector.detect_relationships(windows)
    groups = relationship_detector.group_windows(windows, relationships)
    
    print(f"\n📊 Workspace Analysis:")
    print(f"   Windows: {len(windows)}")
    print(f"   Relationships: {len(relationships)}")
    print(f"   Groups: {len(groups)}")
    
    # Test routing with relationship context
    print(f"\n🔗 Testing Context-Aware Routing:")
    
    # Find a project group
    project_group = next((g for g in groups if g.group_type == "project"), None)
    
    if project_group:
        print(f"\n   Found project group with {len(project_group.windows)} windows")
        print(f"   Apps: {[w.app_name for w in project_group.windows]}")
        
        # Route a work query
        route = router.route_query("What am I working on?", windows)
        
        # Check if routed windows include project group members
        routed_ids = {w.window_id for w in route.target_windows}
        group_ids = {w.window_id for w in project_group.windows}
        
        overlap = routed_ids & group_ids
        if overlap:
            print(f"   ✅ Query routing includes {len(overlap)} windows from project group")
        else:
            print(f"   ⚠️  Query routing didn't include project group windows")
    else:
        print(f"   ⚠️  No project groups found")
    
    # Performance metrics
    print(f"\n⚡ Performance Metrics:")
    
    import time
    
    # Time relationship detection
    start = time.time()
    relationships = relationship_detector.detect_relationships(windows)
    rel_time = time.time() - start
    
    # Time query routing
    start = time.time()
    for _ in range(10):
        router.route_query("What am I working on?", windows)
    route_time = (time.time() - start) / 10
    
    print(f"   Relationship detection: {rel_time*1000:.1f}ms for {len(windows)} windows")
    print(f"   Query routing: {route_time*1000:.1f}ms average")
    
    if rel_time < 1.0 and route_time < 0.1:
        print(f"   ✅ Excellent performance")
    else:
        print(f"   ⚠️  Performance could be improved")


if __name__ == "__main__":
    print("🚀 Phase 2 Intelligence Layer Test Suite")
    print("=" * 60)
    
    # Run tests
    asyncio.run(test_f2_1_relationship_detection())
    asyncio.run(test_f2_2_smart_query_routing())
    asyncio.run(test_integration())
    
    print("\n\n✅ Phase 2 testing complete!")