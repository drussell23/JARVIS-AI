#!/usr/bin/env python3
"""
Demo script for JARVIS Intelligent Command Routing
Shows how the system routes commands without hardcoding
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'swift_bridge'))

from python_bridge import IntelligentCommandRouter, SWIFT_AVAILABLE


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


async def demo():
    """Demonstrate intelligent routing"""
    print(f"{Colors.BOLD}🤖 JARVIS Intelligent Command Routing Demo{Colors.ENDC}")
    print("=" * 50)
    
    if SWIFT_AVAILABLE:
        print(f"{Colors.CYAN}✅ Using Swift NLP classifier{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}⚠️  Using Python fallback (Swift unavailable){Colors.ENDC}")
    
    print("\nThis shows how commands are routed intelligently:")
    print("- No hardcoded keywords")
    print("- Uses linguistic analysis")
    print("- Learns from usage\n")
    
    # Initialize router
    router = IntelligentCommandRouter()
    
    # Demo commands
    commands = [
        ("close whatsapp", "Should route to SYSTEM (action verb)"),
        ("what's on my screen", "Should route to VISION (question)"),
        ("open safari", "Should route to SYSTEM (launch action)"),
        ("show me discord", "Should route to VISION (visual request)"),
    ]
    
    for command, explanation in commands:
        handler, details = await router.route_command(command)
        
        print(f"\n📝 Command: '{Colors.BOLD}{command}{Colors.ENDC}'")
        print(f"   {explanation}")
        print(f"   → Routed to: {Colors.GREEN}{handler.upper()}{Colors.ENDC}")
        print(f"   → Confidence: {details['confidence']:.1%}")
        print(f"   → Reasoning: {details['reasoning']}")
        
        await asyncio.sleep(0.5)  # Pause for readability
    
    print(f"\n{Colors.BOLD}Key Insights:{Colors.ENDC}")
    print("• 'close whatsapp' → SYSTEM (executes action)")
    print("• 'what's in whatsapp' → VISION (analyzes screen)")
    print("• No hardcoded app names or commands!")
    print("• Classifier learns and improves over time")
    
    print(f"\n{Colors.GREEN}✅ Intelligent routing active!{Colors.ENDC}")


if __name__ == "__main__":
    asyncio.run(demo())