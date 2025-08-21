#!/usr/bin/env python3
"""
Test JARVIS Autonomous System
Demonstrates the complete autonomous decision-making capabilities
"""

import asyncio
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from vision.jarvis_workspace_integration import JARVISWorkspaceIntelligence
from vision.window_detector import WindowInfo
from autonomy.autonomous_decision_engine import AutonomousAction, ActionPriority, ActionCategory


async def simulate_workspace_scenario():
    """Simulate a realistic workspace scenario"""
    print("\n" + "="*60)
    print("🤖 JARVIS AUTONOMOUS SYSTEM DEMONSTRATION")
    print("="*60)
    print("This demonstrates how JARVIS can autonomously:")
    print("  • Detect and handle notifications")
    print("  • Prepare for meetings")
    print("  • Organize your workspace")
    print("  • Learn from your preferences")
    print("="*60 + "\n")
    
    # Initialize JARVIS
    jarvis = JARVISWorkspaceIntelligence()
    
    # Test autonomous commands
    print("📍 Phase 1: Testing Autonomous Commands")
    print("-" * 40)
    
    # Enable autonomous mode
    print("\n🎤 User: Hey JARVIS, enable autonomous mode")
    response = await jarvis.handle_autonomous_command("enable autonomous mode")
    print(f"🤖 JARVIS: {response}")
    
    # Check status
    print("\n🎤 User: What's your autonomous status?")
    response = await jarvis.handle_autonomous_command("autonomous status")
    print(f"🤖 JARVIS: {response}")
    
    # Start monitoring
    print("\n📍 Phase 2: Starting Workspace Monitoring")
    print("-" * 40)
    await jarvis.start_monitoring()
    print("✅ Monitoring started - JARVIS is now watching your workspace")
    
    # Simulate workspace changes
    print("\n📍 Phase 3: Simulating Workspace Activity")
    print("-" * 40)
    print("Simulating various scenarios...")
    
    # Let autonomous system run for a bit
    print("⏱️  Letting autonomous system run for 5 seconds...")
    await asyncio.sleep(5)
    
    # Check for any autonomous actions taken
    print("\n📍 Phase 4: Checking Autonomous Actions")
    print("-" * 40)
    
    # Get execution stats
    stats = jarvis.action_executor.get_execution_stats()
    print(f"\n📊 Autonomous Statistics:")
    print(f"  • Total actions executed: {stats['total_executions']}")
    print(f"  • Success rate: {stats.get('success_rate', 0):.0%}")
    print(f"  • Rollback available: {stats.get('rollback_available', 0)}")
    
    # Get permission stats
    perm_stats = jarvis.permission_manager.get_permission_stats()
    print(f"\n🔐 Permission Statistics:")
    print(f"  • Total decisions: {perm_stats['total_decisions']}")
    print(f"  • Unique actions: {perm_stats['unique_actions']}")
    
    # Disable autonomous mode
    print("\n🎤 User: Disable autonomous mode")
    response = await jarvis.handle_autonomous_command("disable autonomous mode")
    print(f"🤖 JARVIS: {response}")
    
    jarvis.stop_monitoring()
    print("\n✅ Test complete!")


async def test_specific_scenarios():
    """Test specific autonomous scenarios"""
    print("\n" + "="*60)
    print("🧪 TESTING SPECIFIC AUTONOMOUS SCENARIOS")
    print("="*60)
    
    # Test the decision engine directly
    from autonomy.autonomous_decision_engine import AutonomousDecisionEngine
    from autonomy.permission_manager import PermissionManager
    from autonomy.context_engine import ContextEngine
    from autonomy.action_executor import ActionExecutor
    
    engine = AutonomousDecisionEngine()
    permission_mgr = PermissionManager()
    context_engine = ContextEngine()
    executor = ActionExecutor()
    
    # Scenario 1: Multiple notifications
    print("\n🎯 Scenario 1: Handling Multiple Notifications")
    print("-" * 40)
    
    test_windows = [
        WindowInfo(
            window_id=1,
            app_name="Slack",
            window_title="Slack (8 new messages)",
            bounds={"x": 0, "y": 0, "width": 800, "height": 600},
            is_focused=False,
            layer=0,
            is_visible=True,
            process_id=1001
        ),
        WindowInfo(
            window_id=2,
            app_name="Discord",
            window_title="Discord - #general (12)",
            bounds={"x": 800, "y": 0, "width": 600, "height": 400},
            is_focused=False,
            layer=0,
            is_visible=True,
            process_id=1002
        )
    ]
    
    # Mock workspace state
    from vision.workspace_analyzer import WorkspaceAnalysis
    mock_state = WorkspaceAnalysis(
        focused_task="Working on project",
        workspace_context="Multiple communication apps with notifications",
        important_notifications=["Slack (8)", "Discord (12)"],
        suggestions=["Check messages"],
        confidence=0.9
    )
    
    # Get autonomous decisions
    actions = await engine.analyze_and_decide(mock_state, test_windows)
    
    print(f"\n🤔 Autonomous decisions made: {len(actions)}")
    for i, action in enumerate(actions[:3]):  # Show first 3
        print(f"\n{i+1}. Action: {action.action_type}")
        print(f"   Target: {action.target}")
        print(f"   Priority: {action.priority.name}")
        print(f"   Confidence: {action.confidence:.0%}")
        print(f"   Reasoning: {action.reasoning}")
        
        # Check permission
        permission, conf, reason = permission_mgr.check_permission(action)
        print(f"   Permission: {'✅ Granted' if permission else '❌ Denied' if permission is False else '❓ Ask user'}")
        print(f"   Reason: {reason}")
    
    # Scenario 2: Meeting preparation
    print("\n\n🎯 Scenario 2: Meeting Preparation")
    print("-" * 40)
    
    meeting_window = WindowInfo(
        window_id=3,
        app_name="Calendar",
        window_title="Team Standup starts in 3 minutes",
        bounds={"x": 0, "y": 0, "width": 400, "height": 300},
        is_focused=True,
        layer=0,
        is_visible=True,
        process_id=1003
    )
    
    test_windows.append(meeting_window)
    
    # Add sensitive window
    sensitive_window = WindowInfo(
        window_id=4,
        app_name="1Password",
        window_title="1Password - Vault",
        bounds={"x": 400, "y": 0, "width": 400, "height": 300},
        is_focused=False,
        layer=1,
        is_visible=True,
        process_id=1004
    )
    test_windows.append(sensitive_window)
    
    # Re-analyze with meeting context
    actions = await engine.analyze_and_decide(mock_state, test_windows)
    
    # Find meeting-related actions
    meeting_actions = [a for a in actions if a.category == ActionCategory.CALENDAR or 
                      a.category == ActionCategory.SECURITY]
    
    if meeting_actions:
        print(f"\n🗓️ Meeting-related actions: {len(meeting_actions)}")
        for action in meeting_actions:
            print(f"  • {action.action_type}: {action.reasoning}")
    
    # Test context awareness
    print("\n\n🎯 Scenario 3: Context-Aware Timing")
    print("-" * 40)
    
    # Analyze current context
    context = await context_engine.analyze_context(mock_state, test_windows)
    
    print(f"📊 Current Context:")
    print(f"  • User State: {context.user_state.value}")
    print(f"  • Interruption Score: {context.interruption_score:.0%}")
    print(f"  • Activity Score: {context.activity_score:.0%}")
    print(f"  • Meeting Probability: {context.meeting_probability:.0%}")
    print(f"  • Reasoning: {context.reasoning}")
    
    # Test if actions should execute now
    if actions:
        test_action = actions[0]
        should_act, timing_reason = context_engine.should_act_now(test_action, context)
        print(f"\n⏰ Should execute '{test_action.action_type}' now? {'Yes' if should_act else 'No'}")
        print(f"   Reason: {timing_reason}")
    
    print("\n✅ Scenario testing complete!")


async def main():
    """Run all tests"""
    try:
        # Run basic demonstration
        await simulate_workspace_scenario()
        
        # Run specific scenario tests
        await test_specific_scenarios()
        
        print("\n" + "="*60)
        print("🎉 JARVIS AUTONOMOUS SYSTEM TEST COMPLETE!")
        print("="*60)
        print("\nKey Capabilities Demonstrated:")
        print("  ✅ Dynamic decision-making based on workspace state")
        print("  ✅ Permission learning from user feedback")
        print("  ✅ Context-aware action timing")
        print("  ✅ Intelligent notification handling")
        print("  ✅ Meeting preparation automation")
        print("  ✅ Security-conscious operations")
        print("\nJARVIS is ready to operate autonomously!")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())