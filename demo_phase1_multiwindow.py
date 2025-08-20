#!/usr/bin/env python3
"""
JARVIS Multi-Window Awareness - Phase 1 Demo
Shows the MVP functionality in action
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.vision.window_detector import WindowDetector
from backend.vision.multi_window_capture import MultiWindowCapture
from backend.vision.jarvis_workspace_integration import JARVISWorkspaceIntelligence


async def demo_phase1():
    """Demonstrate Phase 1 multi-window capabilities"""
    
    print("🚀 JARVIS Multi-Window Awareness - Phase 1 Demo")
    print("=" * 60)
    print("\nWelcome to the world's first Workspace Intelligence Agent!")
    print("\nThis demo shows JARVIS understanding your entire workspace,")
    print("not just a single window.\n")
    
    # Initialize components
    detector = WindowDetector()
    capture_system = MultiWindowCapture()
    workspace_intel = JARVISWorkspaceIntelligence()
    
    # Demo 1: Window Detection
    print("📊 Demo 1: Real-Time Window Detection")
    print("-" * 40)
    
    windows = detector.get_all_windows()
    print(f"\nJARVIS detects {len(windows)} open windows:")
    
    for i, window in enumerate(windows[:5]):
        focus_marker = "🎯" if window.is_focused else "  "
        print(f"{focus_marker} {i+1}. {window.app_name}: {window.window_title or 'Untitled'}")
    
    if len(windows) > 5:
        print(f"   ... and {len(windows) - 5} more")
    
    # Demo 2: Workspace Summary
    print("\n\n📋 Demo 2: Workspace Analysis")
    print("-" * 40)
    
    summary = detector.get_workspace_summary()
    print(f"\nWorkspace Summary:")
    print(f"• Total Windows: {summary['total_windows']}")
    print(f"• Active Applications: {len(summary['applications'])}")
    
    if summary['categorized_windows']:
        print("\nWindow Categories:")
        for category, count in summary['categorized_windows'].items():
            print(f"  • {category.capitalize()}: {count} windows")
    
    # Demo 3: Multi-Window Capture
    print("\n\n📸 Demo 3: Multi-Window Capture")
    print("-" * 40)
    print("\nCapturing workspace (this takes 1-2 seconds)...")
    
    captures = await capture_system.capture_multiple_windows()
    print(f"\n✅ Captured {len(captures)} windows:")
    
    for capture in captures:
        resolution = f"{capture.image.shape[1]}x{capture.image.shape[0]}"
        scale = f"{capture.resolution_scale:.0%}"
        print(f"  • {capture.window_info.app_name}: {resolution} @ {scale} scale")
    
    # Demo 4: Natural Language Queries
    print("\n\n💬 Demo 4: Natural Language Understanding")
    print("-" * 40)
    print("\nNow let's ask JARVIS about your workspace...\n")
    
    queries = [
        "What am I working on?",
        "Do I have any messages?",
        "What windows are open?",
        "Describe my workspace"
    ]
    
    for query in queries:
        print(f"🎤 You: {query}")
        response = await workspace_intel.handle_workspace_command(query)
        print(f"🤖 JARVIS: {response}")
        print()
        await asyncio.sleep(1)
    
    # Demo 5: Focus Change Detection
    print("\n📍 Demo 5: Real-Time Focus Tracking")
    print("-" * 40)
    print("\nJARVIS will monitor window changes for 5 seconds.")
    print("Try switching between windows!\n")
    
    async def monitor_demo():
        previous_windows = detector.get_all_windows()
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < 5:
            changes = detector.detect_window_changes(previous_windows)
            
            if changes['focus_changed'] and changes['current_focus']:
                print(f"→ Focus changed to: {changes['current_focus'].app_name}")
            
            if changes['opened']:
                print(f"→ New window opened!")
            
            if changes['closed']:
                print(f"→ Window closed!")
            
            previous_windows = detector.get_all_windows()
            await asyncio.sleep(0.1)
    
    await monitor_demo()
    
    # Summary
    print("\n\n🎉 Phase 1 Demo Complete!")
    print("=" * 60)
    print("\n✅ What JARVIS Can Now Do:")
    print("  • Detect all open windows across your workspace")
    print("  • Track which window has focus in real-time")
    print("  • Capture multiple windows efficiently")
    print("  • Understand natural language queries about your workspace")
    print("  • Provide context-aware responses")
    
    print("\n🚀 Coming in Phase 2:")
    print("  • Window relationship detection")
    print("  • Smart query routing based on context")
    print("  • Workflow pattern learning")
    
    print("\n💡 Try these commands with JARVIS:")
    print('  • "Hey JARVIS, what am I working on?"')
    print('  • "Do I have any messages?"')
    print('  • "What windows are open?"')
    print('  • "Describe my current workspace"')


if __name__ == "__main__":
    asyncio.run(demo_phase1())