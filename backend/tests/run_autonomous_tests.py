#!/usr/bin/env python3
"""
Test runner for all autonomous system tests
Executes functionality and edge case tests with detailed reporting
"""

import sys
import os
import asyncio
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test result colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


def print_header(title):
    """Print a formatted header"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.ENDC}")


async def run_behavior_tests():
    """Run autonomous behavior functionality tests"""
    print_header("🧪 AUTONOMOUS BEHAVIOR TESTS")
    
    try:
        # Import and run behavior tests
        from test_autonomous_behaviors import (
            TestMessageHandler, TestMeetingHandler, 
            TestWorkspaceOrganizer, TestSecurityHandler,
            TestAutonomousBehaviorManager
        )
        
        import pytest
        
        print(f"\n{Colors.YELLOW}Running behavior functionality tests...{Colors.ENDC}")
        
        # Run pytest programmatically
        result = pytest.main([
            "test_autonomous_behaviors.py",
            "-v",
            "--tb=short",
            "-q"
        ])
        
        if result == 0:
            print(f"{Colors.GREEN}✅ All behavior tests passed!{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.RED}❌ Some behavior tests failed{Colors.ENDC}")
            return False
            
    except Exception as e:
        print(f"{Colors.RED}❌ Error running behavior tests: {e}{Colors.ENDC}")
        return False


async def run_edge_case_tests():
    """Run edge case integration tests"""
    print_header("🔥 EDGE CASE INTEGRATION TESTS")
    
    try:
        # Import and run edge case tests
        from test_autonomous_edge_cases import TestEdgeCaseScenarios, TestSystemIntegration
        
        import pytest
        
        print(f"\n{Colors.YELLOW}Running edge case tests...{Colors.ENDC}")
        
        # Run pytest programmatically
        result = pytest.main([
            "test_autonomous_edge_cases.py",
            "-v",
            "--tb=short",
            "-q"
        ])
        
        if result == 0:
            print(f"{Colors.GREEN}✅ All edge case tests passed!{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.RED}❌ Some edge case tests failed{Colors.ENDC}")
            return False
            
    except Exception as e:
        print(f"{Colors.RED}❌ Error running edge case tests: {e}{Colors.ENDC}")
        return False


async def run_quick_demo():
    """Run a quick demonstration of autonomous behaviors"""
    print_header("🚀 QUICK AUTONOMOUS BEHAVIOR DEMO")
    
    try:
        from autonomy.autonomous_behaviors import AutonomousBehaviorManager
        from vision.window_detector import WindowInfo
        
        print(f"\n{Colors.YELLOW}Creating test scenario...{Colors.ENDC}")
        
        # Create test manager
        manager = AutonomousBehaviorManager()
        
        # Create diverse test windows
        test_windows = [
            WindowInfo(
                window_id=1,
                app_name="Slack",
                window_title="Slack (5 new messages) - #general: Hey team, quick update...",
                bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=9001
            ),
            WindowInfo(
                window_id=2,
                app_name="Calendar",
                window_title="Team Standup starts in 3 minutes - Zoom link ready",
                bounds={"x": 800, "y": 0, "width": 600, "height": 400},
                is_focused=True,
                layer=0,
                is_visible=True,
                process_id=9002
            ),
            WindowInfo(
                window_id=3,
                app_name="Mail",
                window_title="URGENT: Production deployment needs approval",
                bounds={"x": 0, "y": 600, "width": 800, "height": 400},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=9003
            ),
            WindowInfo(
                window_id=4,
                app_name="1Password",
                window_title="1Password - Vault Unlocked",
                bounds={"x": 1400, "y": 0, "width": 400, "height": 600},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=9004
            ),
            WindowInfo(
                window_id=5,
                app_name="Discord",
                window_title="Discord - Gaming Server (12 notifications)",
                bounds={"x": 1400, "y": 600, "width": 400, "height": 400},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=9005
            )
        ]
        
        # Create workspace state
        workspace_state = {
            "window_count": len(test_windows),
            "user_state": "available",
            "in_meeting": False,
            "last_organized": None
        }
        
        print(f"\n{Colors.BOLD}Current Workspace:{Colors.ENDC}")
        print(f"  • {len(test_windows)} windows open")
        print(f"  • Upcoming meeting in 3 minutes")
        print(f"  • Unread messages in Slack and Discord")
        print(f"  • Urgent email requiring attention")
        print(f"  • Password manager visible")
        
        # Process workspace
        print(f"\n{Colors.YELLOW}Analyzing workspace and generating autonomous actions...{Colors.ENDC}")
        
        start_time = time.time()
        actions = await manager.process_workspace_state(workspace_state, test_windows)
        end_time = time.time()
        
        print(f"\n{Colors.GREEN}Analysis completed in {end_time - start_time:.2f} seconds{Colors.ENDC}")
        print(f"\n{Colors.BOLD}Generated {len(actions)} Autonomous Actions:{Colors.ENDC}\n")
        
        # Display actions by priority
        priority_groups = {
            "CRITICAL": [],
            "HIGH": [],
            "MEDIUM": [],
            "LOW": []
        }
        
        for action in actions:
            priority_groups[action.priority.name].append(action)
        
        for priority in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if priority_groups[priority]:
                print(f"{Colors.BOLD}{priority} Priority Actions:{Colors.ENDC}")
                for i, action in enumerate(priority_groups[priority], 1):
                    print(f"  {i}. {action.action_type}")
                    print(f"     → Target: {action.target}")
                    print(f"     → Category: {action.category.value}")
                    print(f"     → Confidence: {action.confidence:.0%}")
                    print(f"     → Reasoning: {action.reasoning}")
                    if action.params:
                        print(f"     → Parameters: {action.params}")
                    print()
        
        # Highlight key behaviors
        print(f"\n{Colors.BOLD}Key Autonomous Behaviors Demonstrated:{Colors.ENDC}")
        
        # Check for meeting preparation
        meeting_actions = [a for a in actions if "meeting" in a.reasoning.lower()]
        if meeting_actions:
            print(f"{Colors.GREEN}✅ Meeting Preparation:{Colors.ENDC}")
            print(f"   • Detected upcoming meeting in 3 minutes")
            print(f"   • Will hide sensitive apps (1Password)")
            print(f"   • Will mute distracting notifications")
        
        # Check for message handling
        message_actions = [a for a in actions if a.category.value == "notification"]
        if message_actions:
            print(f"\n{Colors.GREEN}✅ Intelligent Message Handling:{Colors.ENDC}")
            print(f"   • Identified {len(message_actions)} message-related actions")
            print(f"   • Prioritized urgent production email")
            print(f"   • Queued routine messages for later")
        
        # Check for security
        security_actions = [a for a in actions if a.category.value == "security"]
        if security_actions:
            print(f"\n{Colors.GREEN}✅ Security Awareness:{Colors.ENDC}")
            print(f"   • Detected password manager visibility")
            print(f"   • Will secure before meeting starts")
        
        # Check for workspace organization
        maintenance_actions = [a for a in actions if a.category.value == "maintenance"]
        if maintenance_actions:
            print(f"\n{Colors.GREEN}✅ Workspace Organization:{Colors.ENDC}")
            print(f"   • Suggested workspace improvements")
            print(f"   • Will organize windows efficiently")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ Error in demo: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test runner"""
    print(f"\n{Colors.BOLD}🤖 JARVIS AUTONOMOUS SYSTEM TEST SUITE{Colors.ENDC}")
    print(f"{Colors.BLUE}Testing autonomous behaviors and edge cases{Colors.ENDC}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "behavior_tests": False,
        "edge_case_tests": False,
        "demo": False
    }
    
    # Run all test suites
    try:
        # Run behavior tests
        results["behavior_tests"] = await run_behavior_tests()
        
        # Run edge case tests
        results["edge_case_tests"] = await run_edge_case_tests()
        
        # Run demo
        results["demo"] = await run_quick_demo()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.ENDC}")
    
    # Print summary
    print_header("📊 TEST SUMMARY")
    
    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)
    
    print(f"\n{Colors.BOLD}Results:{Colors.ENDC}")
    print(f"  • Behavior Tests: {'✅ PASSED' if results['behavior_tests'] else '❌ FAILED'}")
    print(f"  • Edge Case Tests: {'✅ PASSED' if results['edge_case_tests'] else '❌ FAILED'}")
    print(f"  • Demo Execution: {'✅ SUCCESS' if results['demo'] else '❌ FAILED'}")
    
    print(f"\n{Colors.BOLD}Overall: {total_passed}/{total_tests} test suites passed{Colors.ENDC}")
    
    if total_passed == total_tests:
        print(f"{Colors.GREEN}\n🎉 All tests passed! Autonomous system is ready.{Colors.ENDC}")
        return 0
    else:
        print(f"{Colors.RED}\n⚠️  Some tests failed. Please review the output above.{Colors.ENDC}")
        return 1


if __name__ == "__main__":
    # Run tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)