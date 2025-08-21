#!/usr/bin/env python3
"""
Test script to verify WhatsApp notification detection is working
Tests the fix for notification queries being routed to workspace intelligence
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice.jarvis_agent_voice import JARVISAgentVoice
from vision.smart_query_router import SmartQueryRouter, QueryIntent
from vision.window_detector import WindowDetector

async def test_notification_routing():
    """Test that notification queries are properly routed"""
    print("🔧 Testing Notification Detection Fix")
    print("=" * 50)
    
    # Test 1: Check voice command detection
    print("\n1️⃣ Testing Voice Command Detection:")
    jarvis = JARVISAgentVoice()
    
    test_queries = [
        "do i have any notifications from whatsapp",
        "any notifications from whatsapp",
        "check whatsapp notifications",
        "whatsapp notifications",
        "do i have any messages",  # This was already working
        "any notifications"
    ]
    
    for query in test_queries:
        is_system = jarvis._is_system_command(query)
        print(f"   '{query}' -> System Command: {is_system} ✅" if is_system else f"   '{query}' -> System Command: {is_system} ❌")
    
    # Test 2: Check query routing
    print("\n2️⃣ Testing Query Router:")
    router = SmartQueryRouter()
    detector = WindowDetector()
    
    # Get current windows
    windows = detector.get_all_windows()
    print(f"\n   Found {len(windows)} windows")
    
    # Find WhatsApp if open
    whatsapp_windows = [w for w in windows if 'whatsapp' in w.app_name.lower()]
    if whatsapp_windows:
        print(f"   ✅ WhatsApp is open: {whatsapp_windows[0].app_name}")
    else:
        print("   ⚠️  WhatsApp not open - test may show no WhatsApp windows")
    
    # Test routing
    test_query = "do i have any notifications from whatsapp"
    route = router.route_query(test_query, windows)
    
    print(f"\n   Query: '{test_query}'")
    print(f"   Intent: {route.intent.value}")
    print(f"   Confidence: {route.confidence:.0%}")
    print(f"   Reasoning: {route.reasoning}")
    print(f"   Target windows: {len(route.target_windows)}")
    
    for i, window in enumerate(route.target_windows[:3]):
        print(f"   {i+1}. {window.app_name} - {window.window_title or 'Untitled'}")
    
    # Test 3: Verify intent detection
    print("\n3️⃣ Testing Intent Detection:")
    
    notification_queries = [
        ("any notifications from whatsapp", QueryIntent.NOTIFICATIONS),
        ("do i have any messages", QueryIntent.MESSAGES),
        ("check whatsapp", QueryIntent.SPECIFIC_APP),
        ("any notifications", QueryIntent.NOTIFICATIONS)
    ]
    
    for query, expected_intent in notification_queries:
        route = router.route_query(query, windows)
        match = route.intent == expected_intent
        print(f"   '{query}' -> {route.intent.value} {'✅' if match else '❌ (expected ' + expected_intent.value + ')'}")
    
    # Test 4: Test workspace command handling
    print("\n4️⃣ Testing Workspace Command Handling:")
    
    # This simulates what happens when JARVIS processes the command
    workspace_command = "do i have any notifications from whatsapp"
    response = await jarvis._handle_system_command(workspace_command)
    
    print(f"   Command: '{workspace_command}'")
    print(f"   Response preview: {response[:200]}...")
    
    if "don't have access" in response.lower():
        print("   ❌ Still getting 'no access' response - fix may need restart")
    else:
        print("   ✅ Workspace intelligence triggered correctly!")
    
    print("\n" + "=" * 50)
    print("✅ Notification detection test complete!")
    print("\n💡 If tests pass but JARVIS still says 'no access':")
    print("   1. Restart JARVIS (Ctrl+C and run start_system.py)")
    print("   2. Make sure WhatsApp is open")
    print("   3. Try: 'Hey JARVIS, do I have any notifications from WhatsApp?'")

if __name__ == "__main__":
    asyncio.run(test_notification_routing())