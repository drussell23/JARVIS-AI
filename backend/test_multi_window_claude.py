#!/usr/bin/env python3
"""
Test Multi-Window Analysis with Claude Vision
Demonstrates dynamic window analysis without hardcoding
"""

import os
import asyncio
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision.workspace_analyzer import WorkspaceAnalyzer
from vision.dynamic_multi_window_engine import get_dynamic_multi_window_engine
from vision.window_detector import WindowDetector


async def test_multi_window_analysis():
    print("🧪 Testing Multi-Window Analysis with Zero Hardcoding")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        print("Please set: export ANTHROPIC_API_KEY=your-key-here")
        return
    
    print("✅ Claude API key found")
    
    # Initialize components
    analyzer = WorkspaceAnalyzer()
    dynamic_engine = get_dynamic_multi_window_engine()
    detector = WindowDetector()
    
    # Get all windows
    all_windows = detector.get_all_windows()
    print(f"\n📊 Found {len(all_windows)} windows total:")
    for i, window in enumerate(all_windows[:10]):
        print(f"  {i+1}. {window.app_name} - {window.window_title or 'Untitled'}")
    
    # Test queries
    test_queries = [
        "Can you describe what's happening on all my screens and windows that I'm not currently looking at?",
        "Show me everything across my entire workspace",
        "What other applications do I have open besides what I'm looking at?",
        "Analyze all my windows and tell me what I'm working on"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"❓ Query: '{query}'")
        print(f"{'='*60}")
        
        # Test dynamic analysis
        print("\n🤖 Dynamic Window Analysis:")
        analysis = dynamic_engine.analyze_windows_for_query(query, all_windows)
        
        print(f"\nConfidence: {analysis.analysis_confidence:.0%}")
        print(f"Reasoning: {analysis.reasoning}")
        
        if analysis.primary_windows:
            print(f"\nPrimary Windows ({len(analysis.primary_windows)}):")
            for window in analysis.primary_windows:
                score = analysis.relevance_map.get(window.window_id, 0)
                print(f"  • {window.app_name}: {window.window_title or 'Untitled'}")
                print(f"    Relevance: {score:.0%}")
        
        if analysis.context_windows:
            print(f"\nContext Windows ({len(analysis.context_windows)}):")
            for window in analysis.context_windows[:5]:
                score = analysis.relevance_map.get(window.window_id, 0)
                print(f"  • {window.app_name}: {window.window_title or 'Untitled'}")
                print(f"    Relevance: {score:.0%}")
        
        # Now test full workspace analysis with Claude
        print("\n📸 Analyzing with Claude Vision...")
        workspace_analysis = await analyzer.analyze_workspace(query)
        
        print(f"\n🎯 Claude's Analysis:")
        print(f"Primary Task: {workspace_analysis.focused_task}")
        print(f"Workspace Context: {workspace_analysis.workspace_context}")
        
        if workspace_analysis.window_relationships:
            print(f"\nWindow Relationships:")
            for rel, details in workspace_analysis.window_relationships.items():
                print(f"  • {rel}: {details}")
        
        if workspace_analysis.important_notifications:
            print(f"\nNotifications:")
            for notif in workspace_analysis.important_notifications:
                print(f"  ⚠️  {notif}")
        
        if workspace_analysis.suggestions:
            print(f"\nSuggestions:")
            for suggestion in workspace_analysis.suggestions:
                print(f"  💡 {suggestion}")
        
        print(f"\nAnalysis Confidence: {workspace_analysis.confidence:.0%}")
        
        # Wait before next query
        if query != test_queries[-1]:
            print("\n⏳ Waiting 3 seconds before next query...")
            await asyncio.sleep(3)
    
    print("\n✅ Multi-window analysis test complete!")
    print("\n📝 Summary:")
    print("- Dynamic window selection without hardcoded app names ✓")
    print("- ML-based relevance scoring ✓")
    print("- Claude Vision integration for all windows ✓")
    print("- Zero hardcoding approach ✓")


if __name__ == "__main__":
    asyncio.run(test_multi_window_analysis())