#!/usr/bin/env python3
"""Test real-time monitoring interaction"""

import asyncio
import os
import sys

# Add backend to path
backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

async def test_real_time_monitoring():
    """Test the proactive real-time monitoring features"""
    print("🎯 Testing Proactive JARVIS Monitoring System")
    print("=" * 80)
    
    # Test 1: Create mock continuous analyzer with enhanced capabilities
    print("\n1. Setting up enhanced mock continuous analyzer...")
    from vision.continuous_screen_analyzer import MemoryAwareScreenAnalyzer
    
    # Create an enhanced mock vision handler
    class MockVisionHandler:
        def __init__(self):
            self.screen_state = "coding"  # Start with coding workflow
            
        async def capture_screen(self):
            return {"success": True, "screenshot": f"mock_image_{self.screen_state}"}
        
        async def describe_screen(self, params):
            # Simulate different responses based on query and state
            query = params.get('query', '').lower()
            
            if 'application' in query:
                if self.screen_state == "coding":
                    return type('', (), {'success': True, 'description': 'VS Code with Python file open'})()
                elif self.screen_state == "error":
                    return type('', (), {'success': True, 'description': 'VS Code showing syntax error'})()
                elif self.screen_state == "research":
                    return type('', (), {'success': True, 'description': 'Chrome with multiple tabs'})()
                    
            return type('', (), {'success': True, 'description': f'Desktop showing {self.screen_state} workflow'})()
    
    mock_handler = MockVisionHandler()
    analyzer = MemoryAwareScreenAnalyzer(mock_handler, update_interval=3.0)
    
    # Test 2: Initialize proactive interaction handler
    print("\n2. Initializing proactive interaction handler...")
    from vision.real_time_interaction_handler import RealTimeInteractionHandler
    
    # Create notification callback for testing
    notifications_received = []
    workflows_detected = []
    opportunities_found = []
    
    async def test_notification_callback(notification):
        print(f"\n💬 JARVIS: {notification['message']}")
        print(f"   Type: {notification.get('type', 'notification')}")
        print(f"   Priority: {notification['priority']}")
        if 'data' in notification:
            if 'workflow' in notification['data']:
                print(f"   Workflow: {notification['data']['workflow']}")
                workflows_detected.append(notification['data']['workflow'])
            if 'opportunity' in notification['data']:
                print(f"   Opportunity: {notification['data']['opportunity'].get('type')}")
                opportunities_found.append(notification['data']['opportunity'])
        notifications_received.append(notification)
    
    # Create mock vision analyzer for Claude integration
    class MockVisionAnalyzer:
        async def analyze_screenshot(self, screenshot, prompt):
            # Simulate intelligent responses based on prompt
            prompt_lower = prompt.lower()
            
            if "proactive" in prompt_lower and "greeting" in prompt_lower:
                return {
                    'message': "I see you're working in VS Code on a Python project. I'll be here watching for opportunities to help - whether it's catching errors, suggesting optimizations, or providing relevant information as you code. Just continue working naturally, and I'll chime in when I spot something useful."
                }
            elif "workflow" in prompt_lower and "identify" in prompt_lower:
                return {
                    'data': {
                        'analysis': {
                            'workflow_type': 'coding',
                            'confidence': 0.9,
                            'indicators': ['VS Code open', 'Python file', 'active typing'],
                            'current_phase': 'implementation',
                            'potential_blockers': []
                        }
                    }
                }
            elif "opportunities" in prompt_lower:
                return {
                    'data': {
                        'analysis': {
                            'opportunities': [
                                {
                                    'type': 'workflow_tip',
                                    'description': 'User has been typing the same pattern multiple times',
                                    'assistance': 'I could suggest a code snippet or shortcut',
                                    'confidence': 0.87,
                                    'urgency': 'medium',
                                    'natural_message': "I noticed you're typing that same pattern repeatedly. Would you like me to show you a shortcut or create a snippet for that?"
                                }
                            ]
                        }
                    }
                }
            elif "farewell" in prompt_lower:
                return {
                    'message': "Great coding session! You made good progress on the Python module. I noticed you resolved that syntax error efficiently. Remember to commit your changes when you're ready. It's been helpful watching your workflow - I'm learning your patterns for next time."
                }
            elif "summary" in prompt_lower:
                return {
                    'message': "I've been monitoring your development workflow. You started with coding in VS Code, encountered and resolved a syntax error, then switched to Chrome for research. This is a common pattern I see when debugging - good problem-solving approach!"
                }
            else:
                # For general analysis prompts
                return {
                    'analysis': 'Monitoring screen activity',
                    'should_interact': False,
                    'confidence': 0.5
                }
        
        async def analyze_screenshot_async(self, screenshot, prompt, **kwargs):
            # Alias for analyze_screenshot
            return await self.analyze_screenshot(screenshot, prompt)
    
    mock_vision_analyzer = MockVisionAnalyzer()
    
    interaction_handler = RealTimeInteractionHandler(
        continuous_analyzer=analyzer,
        notification_callback=test_notification_callback,
        vision_analyzer=mock_vision_analyzer
    )
    
    # Test 3: Start proactive monitoring
    print("\n3. Starting proactive intelligent monitoring...")
    await analyzer.start_monitoring()
    await interaction_handler.start_interactive_monitoring()
    
    print("\n   ✅ Proactive monitoring activated! JARVIS is now intelligently observing...")
    
    # Test 4: Simulate workflow scenarios
    print("\n4. Simulating real workflow scenarios...")
    
    # Let initial observation happen
    await asyncio.sleep(2)
    
    # Simulate coding workflow
    print("\n   💻 Simulating active coding...")
    await analyzer._trigger_event('content_changed', {
        'change_type': 'text_input',
        'application': 'VS Code',
        'details': 'User typing Python code'
    })
    await asyncio.sleep(3)
    
    # Simulate error scenario
    print("\n   🐛 Simulating error detection...")
    mock_handler.screen_state = "error"
    await analyzer._trigger_event('error_detected', {
        'error_context': 'Syntax error in Python code',
        'line': 42,
        'error': "IndentationError: unexpected indent"
    })
    await asyncio.sleep(3)
    
    # Simulate context switch
    print("\n   🔄 Simulating context switch to research...")
    mock_handler.screen_state = "research"
    await analyzer._trigger_event('app_changed', {
        'old_app': 'VS Code',
        'new_app': 'Chrome',
        'context': 'User switched to browser, possibly researching'
    })
    await asyncio.sleep(3)
    
    # Test 5: Get comprehensive statistics
    print("\n5. Getting proactive monitoring statistics...")
    stats = interaction_handler.get_interaction_stats()
    print(f"   Monitoring duration: {stats['monitoring_duration']:.1f} seconds")
    print(f"   Notifications sent: {stats['notifications_sent']}")
    print(f"   Screen changes observed: {stats['screen_changes_observed']}")
    print(f"   Learning data:")
    print(f"      - Workflows observed: {stats['learning_data']['workflows_observed']}")
    print(f"      - Interactions tracked: {stats['learning_data']['interactions_tracked']}")
    
    # Test 6: Test proactive summary
    print("\n6. Getting intelligent screen summary...")
    summary = await interaction_handler.provide_screen_summary()
    print(f"   Summary: {summary}")
    
    # Wait to see proactive interactions
    print("\n7. Observing proactive assistance (15 seconds)...")
    print("   (JARVIS will proactively offer help based on what it observes)")
    await asyncio.sleep(15)
    
    # Test 8: Stop monitoring
    print("\n8. Stopping proactive monitoring...")
    await interaction_handler.stop_interactive_monitoring()
    await analyzer.stop_monitoring()
    
    # Comprehensive summary
    print("\n" + "=" * 80)
    print("📊 Proactive Monitoring Test Summary:")
    print(f"   Total notifications: {len(notifications_received)}")
    print(f"   Workflows detected: {set(workflows_detected) if workflows_detected else 'None'}")
    print(f"   Assistance opportunities: {len(opportunities_found)}")
    
    print("\n   Notification Timeline:")
    for i, notif in enumerate(notifications_received):
        msg = notif['message']
        if len(msg) > 70:
            msg = msg[:67] + "..."
        print(f"   {i+1}. [{notif['priority']}] {msg}")
    
    print("\n✅ Proactive real-time monitoring test completed!")
    print("   JARVIS demonstrated intelligent, context-aware assistance capabilities")

if __name__ == "__main__":
    asyncio.run(test_real_time_monitoring())