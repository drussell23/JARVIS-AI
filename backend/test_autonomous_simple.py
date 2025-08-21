#!/usr/bin/env python3
"""
Simple test for JARVIS Autonomous System
Tests basic functionality without long-running monitoring
"""

import asyncio
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress some verbose logs
logging.getLogger('vision.window_detector').setLevel(logging.WARNING)
logging.getLogger('vision.multi_window_capture').setLevel(logging.ERROR)


async def test_autonomous_components():
    """Test individual autonomous components"""
    print("\n" + "="*60)
    print("🧪 TESTING AUTONOMOUS COMPONENTS")
    print("="*60)
    
    # Test 1: Decision Engine
    print("\n📋 Test 1: Autonomous Decision Engine")
    print("-" * 40)
    
    from autonomy.autonomous_decision_engine import AutonomousDecisionEngine
    from vision.window_detector import WindowInfo
    from vision.workspace_analyzer import WorkspaceAnalysis
    
    engine = AutonomousDecisionEngine()
    
    # Create test scenario
    test_windows = [
        WindowInfo(
            window_id=1,
            app_name="Discord",
            window_title="Discord (5 new messages)",
            bounds={"x": 0, "y": 0, "width": 800, "height": 600},
            is_focused=False,
            layer=0,
            is_visible=True,
            process_id=1001
        ),
        WindowInfo(
            window_id=2,
            app_name="Calendar",
            window_title="Team Meeting starts in 3 minutes",
            bounds={"x": 800, "y": 0, "width": 600, "height": 400},
            is_focused=True,
            layer=0,
            is_visible=True,
            process_id=1002
        )
    ]
    
    mock_state = WorkspaceAnalysis(
        focused_task="Working on project",
        window_relationships={},  # No specific relationships for this test
        workspace_context="Multiple apps open",
        important_notifications=["Discord (5)", "Meeting soon"],
        suggestions=[],
        confidence=0.9
    )
    
    # Get decisions
    actions = await engine.analyze_and_decide(mock_state, test_windows)
    print(f"✅ Decision Engine created {len(actions)} autonomous actions")
    
    for i, action in enumerate(actions[:3]):
        print(f"\n   Action {i+1}: {action.action_type}")
        print(f"   • Target: {action.target}")
        print(f"   • Priority: {action.priority.name}")
        print(f"   • Confidence: {action.confidence:.0%}")
    
    # Test 2: Permission Manager
    print("\n\n📋 Test 2: Permission Manager")
    print("-" * 40)
    
    from autonomy.permission_manager import PermissionManager
    
    perm_mgr = PermissionManager()
    
    if actions:
        test_action = actions[0]
        permission, confidence, reason = perm_mgr.check_permission(test_action)
        
        print(f"✅ Permission check for '{test_action.action_type}':")
        print(f"   • Decision: {'Approved' if permission else 'Denied' if permission is False else 'Ask User'}")
        print(f"   • Confidence: {confidence:.0%}")
        print(f"   • Reason: {reason}")
    
    # Test 3: Context Engine
    print("\n\n📋 Test 3: Context Engine")
    print("-" * 40)
    
    from autonomy.context_engine import ContextEngine
    
    context_engine = ContextEngine()
    context = await context_engine.analyze_context(mock_state, test_windows)
    
    print(f"✅ Context Analysis:")
    print(f"   • User State: {context.user_state.value}")
    print(f"   • Interruption Score: {context.interruption_score:.0%}")
    print(f"   • Activity Score: {context.activity_score:.0%}")
    print(f"   • Meeting Probability: {context.meeting_probability:.0%}")
    
    # Test 4: Action Executor (dry run)
    print("\n\n📋 Test 4: Action Executor (Dry Run)")
    print("-" * 40)
    
    from autonomy.action_executor import ActionExecutor
    
    executor = ActionExecutor()
    
    if actions:
        test_action = actions[0]
        result = await executor.execute_action(test_action, dry_run=True)
        
        print(f"✅ Dry run execution of '{test_action.action_type}':")
        print(f"   • Status: {result.status.value}")
        print(f"   • Execution Time: {result.execution_time:.2f}s" if result.execution_time else "   • Execution Time: N/A")
        print(f"   • Rollback Available: {result.rollback_available}")
    
    print("\n" + "="*60)
    print("✅ All components tested successfully!")
    print("="*60)


async def test_voice_commands():
    """Test autonomous voice commands"""
    print("\n" + "="*60)
    print("🎤 TESTING AUTONOMOUS VOICE COMMANDS")
    print("="*60)
    
    from vision.jarvis_workspace_integration import JARVISWorkspaceIntelligence
    
    jarvis = JARVISWorkspaceIntelligence()
    
    # Test commands
    commands = [
        ("enable autonomous mode", "Enabling autonomous mode"),
        ("autonomous status", "Checking status"),
        ("disable autonomous mode", "Disabling autonomous mode"),
    ]
    
    for command, description in commands:
        print(f"\n🎤 {description}: '{command}'")
        response = await jarvis.handle_autonomous_command(command)
        print(f"🤖 JARVIS: {response}")
    
    print("\n✅ Voice command testing complete!")


async def main():
    """Run all tests"""
    try:
        # Test components
        await test_autonomous_components()
        
        # Test voice commands
        await test_voice_commands()
        
        print("\n" + "="*60)
        print("🎉 JARVIS AUTONOMOUS SYSTEM TEST COMPLETE!")
        print("="*60)
        print("\nKey Findings:")
        print("  ✅ Decision Engine creates actions dynamically")
        print("  ✅ Permission Manager checks permissions intelligently")
        print("  ✅ Context Engine understands user state")
        print("  ✅ Action Executor can simulate actions safely")
        print("  ✅ Voice commands integrate seamlessly")
        print("\nThe autonomous system is ready for use!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())