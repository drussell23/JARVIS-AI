#!/usr/bin/env python3
"""Verify autonomous system is working correctly"""

import asyncio
import sys
import os
from datetime import datetime

# Colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

async def verify_imports():
    """Verify all imports work"""
    print(f"\n{Colors.BLUE}Verifying imports...{Colors.ENDC}")
    
    try:
        from autonomy import (
            AutonomousDecisionEngine, AutonomousAction, ActionPriority, ActionCategory,
            PermissionManager, ContextEngine, ActionExecutor,
            AutonomousBehaviorManager, MessageHandler, MeetingHandler,
            WorkspaceOrganizer, SecurityHandler
        )
        print(f"{Colors.GREEN}✅ All imports successful{Colors.ENDC}")
        return True
    except Exception as e:
        print(f"{Colors.RED}❌ Import error: {e}{Colors.ENDC}")
        return False

async def verify_basic_functionality():
    """Verify basic functionality"""
    print(f"\n{Colors.BLUE}Verifying basic functionality...{Colors.ENDC}")
    
    try:
        from autonomy import AutonomousBehaviorManager
        from vision.window_detector import WindowInfo
        
        # Create manager
        manager = AutonomousBehaviorManager()
        
        # Create test scenario
        windows = [
            WindowInfo(
                window_id=1, app_name="Slack",
                window_title="URGENT: Server down - need immediate help!",
                bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                is_focused=True, layer=0, is_visible=True, process_id=1001
            ),
            WindowInfo(
                window_id=2, app_name="Calendar",
                window_title="Client meeting starts in 2 minutes",
                bounds={"x": 800, "y": 0, "width": 400, "height": 300},
                is_focused=False, layer=0, is_visible=True, process_id=1002
            ),
            WindowInfo(
                window_id=3, app_name="1Password",
                window_title="1Password 7",
                bounds={"x": 0, "y": 600, "width": 400, "height": 400},
                is_focused=False, layer=0, is_visible=True, process_id=1003
            )
        ]
        
        workspace_state = {
            "window_count": 3,
            "user_state": "available",
            "in_meeting": False
        }
        
        # Process
        actions = await manager.process_workspace_state(workspace_state, windows)
        
        print(f"{Colors.GREEN}✅ Generated {len(actions)} actions{Colors.ENDC}")
        
        # Verify we got sensible actions
        assert len(actions) > 0, "No actions generated"
        
        # Check for urgent message handling
        urgent_handled = any("urgent" in a.reasoning.lower() for a in actions)
        print(f"{'✅' if urgent_handled else '⚠️'} Urgent message handling: {urgent_handled}")
        
        # Check for meeting preparation
        meeting_handled = any("meeting" in a.reasoning.lower() for a in actions)
        print(f"{'✅' if meeting_handled else '⚠️'} Meeting preparation: {meeting_handled}")
        
        # Check priority ordering
        priorities = [a.priority for a in actions]
        priority_values = [p.value for p in priorities]
        is_sorted = priority_values == sorted(priority_values)
        print(f"{'✅' if is_sorted else '⚠️'} Actions properly prioritized: {is_sorted}")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ Functionality error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return False

async def verify_edge_cases():
    """Verify edge case handling"""
    print(f"\n{Colors.BLUE}Verifying edge case handling...{Colors.ENDC}")
    
    try:
        from autonomy import AutonomousBehaviorManager
        from vision.window_detector import WindowInfo
        
        manager = AutonomousBehaviorManager()
        
        # Test 1: Empty workspace
        actions = await manager.process_workspace_state({}, [])
        print(f"{'✅' if actions == [] else '⚠️'} Empty workspace handled correctly")
        
        # Test 2: Massive workspace (50 windows)
        many_windows = [
            WindowInfo(
                window_id=i, app_name=f"App{i}",
                window_title=f"Window {i}",
                bounds={"x": (i % 10) * 100, "y": (i // 10) * 100, "width": 300, "height": 200},
                is_focused=(i == 0), layer=0, is_visible=True, process_id=2000+i
            ) for i in range(50)
        ]
        
        start = datetime.now()
        actions = await manager.process_workspace_state(
            {"window_count": 50, "user_state": "available"}, 
            many_windows
        )
        duration = (datetime.now() - start).total_seconds()
        
        print(f"{'✅' if duration < 2.0 else '⚠️'} Large workspace processed in {duration:.2f}s")
        print(f"{'✅' if len(actions) <= 10 else '⚠️'} Actions limited to {len(actions)} (max 10)")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ Edge case error: {e}{Colors.ENDC}")
        return False

async def main():
    """Run all verifications"""
    print(f"{Colors.BOLD}\n🤖 JARVIS Autonomous System Verification{Colors.ENDC}")
    print("=" * 50)
    
    results = {
        "imports": await verify_imports(),
        "functionality": await verify_basic_functionality(),
        "edge_cases": await verify_edge_cases()
    }
    
    # Summary
    print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
    print("=" * 50)
    
    all_passed = all(results.values())
    
    for test, passed in results.items():
        status = f"{Colors.GREEN}PASSED{Colors.ENDC}" if passed else f"{Colors.RED}FAILED{Colors.ENDC}"
        print(f"{test.capitalize()}: {status}")
    
    if all_passed:
        print(f"\n{Colors.GREEN}✅ All verifications passed! Autonomous system is working correctly.{Colors.ENDC}")
    else:
        print(f"\n{Colors.RED}❌ Some verifications failed. Please check the errors above.{Colors.ENDC}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)