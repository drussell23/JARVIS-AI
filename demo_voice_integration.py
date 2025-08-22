#!/usr/bin/env python3
"""
Voice Integration Demo for JARVIS
Demonstrates the comprehensive voice capabilities
"""

import asyncio
import os
import sys
import logging
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from autonomy.voice_integration import VoiceIntegrationSystem, VoiceInteractionType, ApprovalResponse
from autonomy.autonomous_decision_engine import AutonomousAction, ActionPriority
from autonomy.notification_intelligence import IntelligentNotification, NotificationContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VoiceIntegrationDemo:
    """Demonstration of JARVIS Voice Integration capabilities"""
    
    def __init__(self):
        self.voice_system = None
        
    async def run_demo(self):
        """Run the complete voice integration demonstration"""
        print("\n" + "="*80)
        print("🎯 JARVIS Voice Integration System - Comprehensive Demo")
        print("="*80)
        
        try:
            # Initialize system
            await self._initialize_system()
            
            # Demo 1: Voice Announcements
            await self._demo_voice_announcements()
            
            # Demo 2: Natural Voice Communication
            await self._demo_natural_communication()
            
            # Demo 3: Voice-based Approvals
            await self._demo_voice_approvals()
            
            # Demo 4: Intelligent Notification Announcements
            await self._demo_notification_announcements()
            
            # Demo 5: Contextual Voice Responses
            await self._demo_contextual_responses()
            
            # Show system statistics
            await self._show_statistics()
            
            # Clean shutdown
            await self._shutdown_system()
            
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
            if self.voice_system:
                await self.voice_system.stop_voice_integration()
        except Exception as e:
            print(f"❌ Demo error: {e}")
            logger.error(f"Demo error: {e}")
            
    async def _initialize_system(self):
        """Initialize the voice integration system"""
        print("\n🚀 Initializing JARVIS Voice Integration System...")
        
        # Check for API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
            print("Please set your Claude API key: export ANTHROPIC_API_KEY='your-key-here'")
            sys.exit(1)
            
        # Initialize voice system
        self.voice_system = VoiceIntegrationSystem(api_key)
        
        # Start the system
        await self.voice_system.start_voice_integration()
        
        print("✅ Voice Integration System initialized and running!")
        await asyncio.sleep(2)
        
    async def _demo_voice_announcements(self):
        """Demonstrate voice announcement capabilities"""
        print("\n📢 Demo 1: Voice Announcements")
        print("-" * 50)
        
        # Queue various types of announcements
        announcements = [
            {
                "content": "New message from the development team about the deployment",
                "urgency": 0.8,
                "context": "work_notification",
                "description": "High urgency work notification"
            },
            {
                "content": "Calendar reminder: Team meeting starts in 15 minutes",
                "urgency": 0.7,
                "context": "meeting_reminder", 
                "description": "Meeting reminder"
            },
            {
                "content": "System update completed successfully",
                "urgency": 0.3,
                "context": "system_status",
                "description": "Low urgency system status"
            },
            {
                "content": "Battery level is getting low - 15% remaining",
                "urgency": 0.6,
                "context": "system_alert",
                "description": "Medium urgency system alert"
            }
        ]
        
        for i, announcement in enumerate(announcements, 1):
            print(f"  📤 Queuing announcement {i}: {announcement['description']}")
            await self.voice_system.announcement_system.queue_announcement(
                content=announcement["content"],
                urgency=announcement["urgency"],
                context=announcement["context"]
            )
            await asyncio.sleep(1)
            
        # Let announcements process
        print("  ⏳ Processing announcements...")
        await asyncio.sleep(10)
        print("✅ Voice announcements demo completed!")
        
    async def _demo_natural_communication(self):
        """Demonstrate natural voice communication"""
        print("\n🎤 Demo 2: Natural Voice Communication")
        print("-" * 50)
        
        # Simulate various voice commands
        voice_commands = [
            {
                "command": "What's my schedule looking like today?",
                "confidence": 0.95,
                "description": "Schedule inquiry"
            },
            {
                "command": "Can you help me organize my workspace?",
                "confidence": 0.87,
                "description": "Workspace assistance request"
            },
            {
                "command": "I need to focus for the next two hours",
                "confidence": 0.92,
                "description": "Focus mode request"
            },
            {
                "command": "What are some suggestions for improving productivity?",
                "confidence": 0.88,
                "description": "Productivity advice request"
            }
        ]
        
        for i, cmd in enumerate(voice_commands, 1):
            print(f"  🗣️  Processing command {i}: {cmd['description']}")
            print(f"      User: \"{cmd['command']}\"")
            
            response = await self.voice_system.process_voice_command(
                command=cmd["command"],
                confidence=cmd["confidence"]
            )
            
            print(f"      JARVIS: \"{response}\"")
            print()
            await asyncio.sleep(2)
            
        print("✅ Natural communication demo completed!")
        
    async def _demo_voice_approvals(self):
        """Demonstrate voice-based approval system"""
        print("\n✅ Demo 3: Voice-based Approvals")
        print("-" * 50)
        
        # Create test actions requiring approval
        test_actions = [
            {
                "action": AutonomousAction(
                    action_type="close_inactive_apps",
                    target="system",
                    params={"apps": ["Safari", "TextEdit"]},
                    priority=ActionPriority.MEDIUM,
                    confidence=0.6,  # Below threshold, needs approval
                    category="system_optimization",
                    reasoning="Close inactive applications to free up system resources"
                ),
                "description": "Close inactive applications"
            },
            {
                "action": AutonomousAction(
                    action_type="organize_desktop",
                    target="desktop",
                    params={"method": "by_type"},
                    priority=ActionPriority.LOW,
                    confidence=0.65,  # Below threshold, needs approval
                    category="workspace_organization",
                    reasoning="Organize desktop files by type to improve workspace clarity"
                ),
                "description": "Organize desktop files"
            },
            {
                "action": AutonomousAction(
                    action_type="schedule_break_reminder",
                    target="calendar",
                    params={"interval": 30},
                    priority=ActionPriority.LOW,
                    confidence=0.5,  # Below threshold, needs approval
                    category="wellness",
                    reasoning="Schedule break reminders to maintain productivity and health"
                ),
                "description": "Schedule break reminders"
            }
        ]
        
        for i, action_info in enumerate(test_actions, 1):
            print(f"  🤔 Requesting approval {i}: {action_info['description']}")
            
            approval = await self.voice_system.request_approval(
                action=action_info["action"],
                timeout=10  # Shorter timeout for demo
            )
            
            print(f"      Approval result: {approval.value}")
            
            if approval == ApprovalResponse.APPROVED:
                print(f"      ✅ Action would be executed: {action_info['action'].action_type}")
            else:
                print(f"      ❌ Action cancelled or denied")
                
            print()
            await asyncio.sleep(2)
            
        print("✅ Voice approvals demo completed!")
        
    async def _demo_notification_announcements(self):
        """Demonstrate intelligent notification announcements"""
        print("\n🔔 Demo 4: Intelligent Notification Announcements")
        print("-" * 50)
        
        # Create simulated intelligent notifications
        notifications = [
            IntelligentNotification(
                source_window_id="slack_main",
                app_name="Slack",
                detected_text=["New message from @john.doe", "The server migration is complete"],
                visual_elements={"badge": {"count": 1}},
                context=NotificationContext.WORK_UPDATE,
                confidence=0.9,
                urgency_score=0.7
            ),
            IntelligentNotification(
                source_window_id="calendar_app",
                app_name="Calendar",
                detected_text=["Meeting in 5 minutes", "Daily standup with development team"],
                visual_elements={"banner": {"type": "reminder"}},
                context=NotificationContext.MEETING_REMINDER,
                confidence=0.95,
                urgency_score=0.9
            ),
            IntelligentNotification(
                source_window_id="system_alert",
                app_name="System Preferences",
                detected_text=["Software update available", "macOS Sonoma 14.1.2"],
                visual_elements={"notification": {"type": "system"}},
                context=NotificationContext.SYSTEM_ALERT,
                confidence=0.8,
                urgency_score=0.4
            )
        ]
        
        for i, notification in enumerate(notifications, 1):
            print(f"  🔔 Announcing notification {i}: {notification.context.value}")
            print(f"      From: {notification.app_name}")
            print(f"      Content: {' | '.join(notification.detected_text)}")
            
            announcement_id = await self.voice_system.announce_notification(notification)
            print(f"      Announcement ID: {announcement_id}")
            print()
            await asyncio.sleep(3)
            
        print("✅ Notification announcements demo completed!")
        
    async def _demo_contextual_responses(self):
        """Demonstrate contextual voice responses"""
        print("\n🧠 Demo 5: Contextual Voice Responses")
        print("-" * 50)
        
        # Simulate different contexts and show how responses adapt
        contextual_scenarios = [
            {
                "scenario": "Early morning (low energy)",
                "commands": [
                    "Good morning JARVIS",
                    "How should I start my day?"
                ],
                "context_description": "User just started their day"
            },
            {
                "scenario": "High workload period",
                "commands": [
                    "I'm feeling overwhelmed with tasks",
                    "Help me prioritize my work"
                ],
                "context_description": "User experiencing high cognitive load"
            },
            {
                "scenario": "End of day wrap-up",
                "commands": [
                    "What did I accomplish today?",
                    "Prepare tomorrow's agenda"
                ],
                "context_description": "User ending their work day"
            }
        ]
        
        for i, scenario in enumerate(contextual_scenarios, 1):
            print(f"  🎭 Scenario {i}: {scenario['scenario']}")
            print(f"      Context: {scenario['context_description']}")
            
            for command in scenario["commands"]:
                print(f"      User: \"{command}\"")
                
                response = await self.voice_system.process_voice_command(
                    command=command,
                    confidence=0.9
                )
                
                print(f"      JARVIS: \"{response}\"")
                await asyncio.sleep(2)
                
            print()
            
        print("✅ Contextual responses demo completed!")
        
    async def _show_statistics(self):
        """Show voice system statistics"""
        print("\n📊 Voice System Statistics")
        print("-" * 50)
        
        stats = self.voice_system.get_voice_statistics()
        
        print(f"  Total Interactions: {stats.get('total_interactions', 0)}")
        print(f"  Recent Interactions: {stats.get('recent_interactions', 0)}")
        print(f"  Announcement Queue Size: {stats.get('announcement_queue_size', 0)}")
        print(f"  Conversation Active: {stats.get('conversation_active', False)}")
        print(f"  System Active: {stats.get('system_active', False)}")
        
        interaction_types = stats.get('interaction_types', {})
        if interaction_types:
            print("\n  Interaction Types:")
            for interaction_type, count in interaction_types.items():
                print(f"    {interaction_type}: {count}")
                
        print("\n✅ Statistics displayed!")
        
    async def _shutdown_system(self):
        """Gracefully shutdown the voice system"""
        print("\n🛑 Shutting down Voice Integration System...")
        
        if self.voice_system:
            await self.voice_system.stop_voice_integration()
            
        print("✅ Voice Integration System shut down gracefully!")
        print("\n" + "="*80)
        print("🎯 Voice Integration Demo completed successfully!")
        print("="*80)


async def main():
    """Main demo function"""
    demo = VoiceIntegrationDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())